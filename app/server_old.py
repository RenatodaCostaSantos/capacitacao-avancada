# ==========================================
# TELECOM ANOMALY — FASTAPI SERVER
# Execute: uvicorn server:app --reload --port 8000
# ==========================================

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.pipeline import (
    PipelineConfig, load_model,
    create_telecom_features, predict,
    TrainResult, PredictionResult
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ==========================================
# STATE GLOBAL
# ==========================================

class AppState:
    model_result: Optional[TrainResult] = None
    connected_clients: list[WebSocket] = []
    alert_history: list[dict] = []
    stats: dict = {
        "total_predictions": 0,
        "total_anomalies":   0,
        "total_critical":    0,
        "uptime_start":      time.time(),
    }

state = AppState()

# ==========================================
# LIFESPAN — carrega modelo na inicialização
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PipelineConfig()
    if Path(config.model_path).exists():
        log.info("Carregando modelo salvo...")
        state.model_result = load_model(config)
        log.info("✅ Modelo pronto para inferência.")
    else:
        log.warning("⚠️  Modelo não encontrado. Execute train.py primeiro.")
    yield
    log.info("Servidor encerrado.")

# ==========================================
# APP
# ==========================================

app = FastAPI(
    title="Telecom Anomaly Detection API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# serve o frontend estático
if Path("frontend").exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# ==========================================
# SCHEMAS
# ==========================================

class TelecomSample(BaseModel):
    """Payload de uma amostra em tempo real (POST /predict)"""
    rx_speed:          float = 0.0
    tx_speed:          float = 0.0
    rx_total_bytes:    float = 0.0
    tx_total_bytes:    float = 0.0
    received_packets:  float = 0.0
    dropped_packets:   float = 0.0
    buffer_overruns:   float = 0.0
    parse_errors:      float = 0.0
    # features derivadas (opcional — se não enviar, serão 0)
    throughput:        Optional[float] = None
    packet_rate:       Optional[float] = None
    packet_loss_rate:  Optional[float] = None
    byte_rate:         Optional[float] = None
    jitter:            Optional[float] = None

    def to_feature_dict(self) -> dict:
        d = self.model_dump()
        if d["throughput"] is None:
            d["throughput"] = d["rx_speed"] + d["tx_speed"]
        # valores derivados que dependem de janela temporal
        # quando enviados como None, ficam 0 (modelo lida com isso)
        for k in ["packet_rate", "packet_loss_rate", "byte_rate", "jitter"]:
            if d[k] is None:
                d[k] = 0.0
        return d


# ==========================================
# HELPERS
# ==========================================

def _check_model():
    if state.model_result is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo não carregado. Execute train.py e reinicie o servidor."
        )

def _build_event(pred: PredictionResult, sample: dict) -> dict:
    """Monta o evento que vai para o WebSocket e para o histórico."""
    state.stats["total_predictions"] += 1
    if pred.is_anomaly:
        state.stats["total_anomalies"] += 1
    if pred.anomaly_level == "critical":
        state.stats["total_critical"] += 1

    event = {
        "ts":            time.time(),
        "score":         pred.score,
        "is_anomaly":    pred.is_anomaly,
        "level":         pred.anomaly_level,
        "confidence":    pred.confidence,
        "throughput":    round(sample.get("throughput", 0), 2),
        "rx_speed":      round(sample.get("rx_speed", 0), 2),
        "tx_speed":      round(sample.get("tx_speed", 0), 2),
        "packet_loss":   round(sample.get("packet_loss_rate", 0), 4),
        "jitter":        round(sample.get("jitter", 0), 2),
        "buffer_overruns": int(sample.get("buffer_overruns", 0)),
        "stats":         dict(state.stats),
    }

    if pred.is_anomaly:
        alert = {
            "ts":      event["ts"],
            "level":   pred.anomaly_level,
            "score":   pred.score,
            "message": _alert_message(pred, sample),
        }
        state.alert_history.append(alert)
        state.alert_history = state.alert_history[-200:]   # ring buffer
        event["alert"] = alert

    return event


def _alert_message(pred: PredictionResult, sample: dict) -> str:
    tp  = sample.get("throughput", 0)
    pl  = sample.get("packet_loss_rate", 0)
    jit = sample.get("jitter", 0)
    buf = sample.get("buffer_overruns", 0)

    if tp < 20:
        return f"Queda crítica de throughput: {tp:.1f} Mb/s"
    if pl > 15:
        return f"Packet loss elevado: {pl:.1f} pkt/s"
    if jit > 30:
        return f"Jitter instável: {jit:.1f} ms"
    if buf > 5:
        return f"Buffer overruns: {buf} eventos"
    return f"Anomalia detectada — score {pred.score:.4f}"


async def _broadcast(event: dict):
    """Envia evento para todos os clientes WebSocket conectados."""
    dead = []
    payload = json.dumps(event)
    for ws in state.connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        state.connected_clients.remove(ws)


# ==========================================
# REST ENDPOINTS
# ==========================================

@app.get("/")
async def root():
    if Path("frontend/index.html").exists():
        return FileResponse("frontend/index.html")
    return {"status": "ok", "docs": "/docs"}


@app.get("/health")
async def health():
    return {
        "status":         "ok",
        "model_loaded":   state.model_result is not None,
        "clients":        len(state.connected_clients),
        "uptime_s":       round(time.time() - state.stats["uptime_start"]),
        **state.stats
    }


@app.get("/model/info")
async def model_info():
    _check_model()
    r = state.model_result
    return {
        "features":       r.feature_cols,
        "n_features":     len(r.feature_cols),
        "threshold":      r.threshold,
        "contamination":  r.model.contamination,
        "n_estimators":   r.model.n_estimators,
    }


@app.post("/predict")
async def predict_endpoint(sample: TelecomSample):
    """
    POST uma amostra, recebe predição imediata.
    Também faz broadcast para todos os clientes WebSocket.
    """
    _check_model()
    feat_dict = sample.to_feature_dict()
    pred      = predict(feat_dict, state.model_result)
    event     = _build_event(pred, feat_dict)
    await _broadcast(event)
    return {
        "score":       pred.score,
        "is_anomaly":  pred.is_anomaly,
        "level":       pred.anomaly_level,
        "confidence":  pred.confidence,
    }


@app.get("/alerts")
async def get_alerts(limit: int = 50):
    return {"alerts": list(reversed(state.alert_history))[:limit]}


@app.get("/stats")
async def get_stats():
    total = state.stats["total_predictions"]
    anom  = state.stats["total_anomalies"]
    return {
        **state.stats,
        "anomaly_rate": round(anom / max(total, 1), 4),
        "uptime_s":     round(time.time() - state.stats["uptime_start"]),
    }


# ==========================================
# WEBSOCKET — streaming em tempo real
# ==========================================

'''@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.connected_clients.append(websocket)
    log.info(f"Cliente WS conectado. Total: {len(state.connected_clients)}")

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            if data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "ts": time.time()}))
                continue

            if data.get("type") == "sample":
                _check_model()
                sample = TelecomSample(**data["payload"])
                feat_dict = sample.to_feature_dict()
                pred      = predict(feat_dict, state.model_result)
                event     = _build_event(pred, feat_dict)
                await _broadcast(event)

    except WebSocketDisconnect:
        state.connected_clients.remove(websocket)
        log.info(f"Cliente WS desconectado. Total: {len(state.connected_clients)}")
    except Exception as e:
        log.error(f"WS error: {e}")
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)'''


# no endpoint WebSocket, adicionar tratamento de ping explícito
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    state.connected_clients.append(websocket)

    try:
        while True:
            try:
                # timeout de 60s em vez de travar indefinidamente
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                # manda ping manual para manter vivo
                await websocket.send_text('{"type":"ping"}')
                continue

            data = json.loads(raw)
            if data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "ts": time.time()}))
                continue

            if data.get("type") == "sample":
                _check_model()
                sample    = TelecomSample(**data["payload"])
                feat_dict = sample.to_feature_dict()
                pred      = predict(feat_dict, state.model_result)
                event     = _build_event(pred, feat_dict)
                await _broadcast(event)

    except WebSocketDisconnect:
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)
    except Exception as e:
        log.error(f"WS error: {e}")
        if websocket in state.connected_clients:
            state.connected_clients.remove(websocket)


# ==========================================
# STREAM SIMULATOR (para testes sem dados reais)
# Ativa via: GET /simulate/start  |  /simulate/stop
# ==========================================

_sim_task: Optional[asyncio.Task] = None

@app.get("/simulate/start")
async def simulate_start(interval_ms: int = 500):
    global _sim_task
    if _sim_task and not _sim_task.done():
        return {"status": "already_running"}
    _sim_task = asyncio.create_task(_simulate_loop(interval_ms / 1000))
    return {"status": "started", "interval_ms": interval_ms}


@app.get("/simulate/stop")
async def simulate_stop():
    global _sim_task
    if _sim_task:
        _sim_task.cancel()
    return {"status": "stopped"}


async def _simulate_loop(interval: float):
    """Gera amostras sintéticas e faz broadcast via WebSocket."""
    _check_model()
    rng = np.random.default_rng(42)
    tick = 0
    log.info(f"Simulação iniciada (interval={interval}s)")

    while True:
        tick += 1
        is_anomaly = rng.random() < 0.08

        if is_anomaly:
            typ = rng.integers(0, 3)
            if typ == 0:    # queda de sinal
                rx, tx = rng.uniform(3, 12), rng.uniform(2, 10)
                pl, jit, buf = rng.uniform(20, 50), rng.uniform(40, 100), int(rng.integers(5, 20))
            elif typ == 1:  # spike de banda
                rx, tx = rng.uniform(90, 140), rng.uniform(80, 120)
                pl, jit, buf = rng.uniform(1, 5), rng.uniform(5, 20), int(rng.integers(10, 30))
            else:           # jitter/perda moderada
                rx, tx = rng.uniform(30, 55), rng.uniform(25, 45)
                pl, jit, buf = rng.uniform(10, 30), rng.uniform(25, 70), int(rng.integers(3, 10))
        else:
            rx  = rng.uniform(40, 55)
            tx  = rng.uniform(35, 50)
            pl  = rng.uniform(0, 2)
            jit = rng.uniform(0, 6)
            buf = int(rng.integers(0, 2))

        sample_dict = {
            "rx_speed":         float(rx),
            "tx_speed":         float(tx),
            "rx_total_bytes":   float(rng.uniform(1e6, 5e6)),
            "tx_total_bytes":   float(rng.uniform(1e6, 4e6)),
            "received_packets": float(rng.uniform(100, 500)),
            "dropped_packets":  float(pl * 5),
            "buffer_overruns":  float(buf),
            "parse_errors":     float(rng.integers(0, 3)),
            "throughput":       float(rx + tx),
            "packet_rate":      float(rng.uniform(10, 60)),
            "packet_loss_rate": float(pl),
            "byte_rate":        float(rng.uniform(1e4, 5e4)),
            "jitter":           float(jit),
        }

        pred  = predict(sample_dict, state.model_result)
        event = _build_event(pred, sample_dict)
        event["simulated"] = True
        await _broadcast(event)
        await asyncio.sleep(interval)
