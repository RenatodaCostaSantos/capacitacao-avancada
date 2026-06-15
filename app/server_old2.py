# ==========================================
# TELECOM ANOMALY — FASTAPI SERVER (fixed)
# uvicorn server:app --host 0.0.0.0 --port 8000
# ==========================================

import asyncio
import json
import time
import logging
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.pipeline import (
    PipelineConfig, load_model,
    predict, TrainResult, PredictionResult
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ==========================================
# CLIENTE WS — fila individual por conexão
# ==========================================

class WSClient:
    def __init__(self, ws: WebSocket):
        self.ws    = ws
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.alive = True

    async def sender(self):
        """Task separada — consome a fila e envia sem bloquear o receiver."""
        while self.alive:
            try:
                payload = await asyncio.wait_for(self.queue.get(), timeout=30.0)
                await self.ws.send_text(payload)
            except asyncio.TimeoutError:
                try:
                    await self.ws.send_text('{"type":"ping"}')
                except Exception:
                    self.alive = False
                    break
            except Exception:
                self.alive = False
                break

    def enqueue(self, payload: str):
        """Non-blocking — descarta se fila cheia (evita backpressure)."""
        try:
            self.queue.put_nowait(payload)
        except asyncio.QueueFull:
            pass


# ==========================================
# STATE GLOBAL
# ==========================================

class AppState:
    model_result: Optional[TrainResult] = None
    clients: list[WSClient] = []
    alert_history: list[dict] = []
    stats: dict = None

    def reset_stats(self):
        self.stats = {
            "total_predictions": 0,
            "total_anomalies":   0,
            "total_critical":    0,
            "uptime_start":      time.time(),
        }

state = AppState()
state.reset_stats()


# ==========================================
# LIFESPAN
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    config = PipelineConfig()
    if Path(config.model_path).exists():
        log.info("Carregando modelo...")
        state.model_result = load_model(config)
        log.info("Modelo pronto.")
    else:
        log.warning("Modelo nao encontrado. Execute train.py primeiro.")
    yield


app = FastAPI(title="Telecom Anomaly API", version="2.0.0", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if Path("frontend").exists():
    app.mount("/static", StaticFiles(directory="frontend"), name="static")


# ==========================================
# SCHEMAS
# ==========================================

class TelecomSample(BaseModel):
    rx_speed:          float = 0.0
    tx_speed:          float = 0.0
    rx_total_bytes:    float = 0.0
    tx_total_bytes:    float = 0.0
    received_packets:  float = 0.0
    dropped_packets:   float = 0.0
    buffer_overruns:   float = 0.0
    parse_errors:      float = 0.0
    throughput:        Optional[float] = None
    packet_rate:       Optional[float] = None
    packet_loss_rate:  Optional[float] = None
    byte_rate:         Optional[float] = None
    jitter:            Optional[float] = None

    def to_feature_dict(self) -> dict:
        d = self.model_dump()
        if d["throughput"] is None:
            d["throughput"] = d["rx_speed"] + d["tx_speed"]
        for k in ["packet_rate", "packet_loss_rate", "byte_rate", "jitter"]:
            if d[k] is None:
                d[k] = 0.0
        return d


# ==========================================
# HELPERS
# ==========================================

def _check_model():
    if state.model_result is None:
        raise HTTPException(status_code=503, detail="Modelo nao carregado.")


def _build_event(pred: PredictionResult, sample: dict) -> dict:
    state.stats["total_predictions"] += 1
    if pred.is_anomaly:
        state.stats["total_anomalies"] += 1
    if pred.anomaly_level == "critical":
        state.stats["total_critical"] += 1

    event = {
        "ts":              time.time(),
        "score":           pred.score,
        "is_anomaly":      pred.is_anomaly,
        "level":           pred.anomaly_level,
        "confidence":      pred.confidence,
        "throughput":      round(sample.get("throughput", 0), 2),
        "rx_speed":        round(sample.get("rx_speed", 0), 2),
        "tx_speed":        round(sample.get("tx_speed", 0), 2),
        "packet_loss":     round(sample.get("packet_loss_rate", 0), 4),
        "jitter":          round(sample.get("jitter", 0), 2),
        "buffer_overruns": int(sample.get("buffer_overruns", 0)),
        "stats":           dict(state.stats),
    }

    if pred.is_anomaly:
        alert = {
            "ts":      event["ts"],
            "level":   pred.anomaly_level,
            "score":   pred.score,
            "message": _alert_message(pred, sample),
        }
        state.alert_history.append(alert)
        state.alert_history = state.alert_history[-200:]
        event["alert"] = alert

    return event


def _alert_message(pred: PredictionResult, sample: dict) -> str:
    tp  = sample.get("throughput", 0)
    pl  = sample.get("packet_loss_rate", 0)
    jit = sample.get("jitter", 0)
    buf = sample.get("buffer_overruns", 0)
    if tp < 20:
        return f"Queda critica de throughput: {tp:.1f} Mb/s"
    if pl > 15:
        return f"Packet loss elevado: {pl:.1f} pkt/s"
    if jit > 30:
        return f"Jitter instavel: {jit:.1f} ms"
    if buf > 5:
        return f"Buffer overruns: {buf} eventos"
    return f"Anomalia detectada - score {pred.score:.4f}"

#%%
def _convert_to_serializable(obj):
    """Converte tipos NumPy para tipos Python nativos JSON-serializáveis."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(v) for v in obj]
    return obj


def _broadcast(event: dict):
    """Enfileira para todos os clientes — completamente non-blocking."""
    event = _convert_to_serializable(event)
    payload = json.dumps(event)
    dead = [c for c in state.clients if not c.alive]
    for c in dead:
        state.clients.remove(c)
    for c in state.clients:
        c.enqueue(payload)


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
        "status":       "ok",
        "model_loaded": state.model_result is not None,
        "clients":      len(state.clients),
        "uptime_s":     round(time.time() - state.stats["uptime_start"]),
        **state.stats,
    }


@app.get("/model/info")
async def model_info():
    _check_model()
    r = state.model_result
    return {
        "features":      r.feature_cols,
        "n_features":    len(r.feature_cols),
        "threshold":     r.threshold,
        "contamination": r.model.contamination,
        "n_estimators":  r.model.n_estimators,
    }


@app.get("/model/metrics")
async def model_metrics():
    _check_model()
    r = state.model_result
    return {
        "accuracy":         r.metrics.get("accuracy", 0),
        "precision":        r.metrics.get("precision", 0),
        "recall":           r.metrics.get("recall", 0),
        "f1":               r.metrics.get("f1", 0),
        "threshold":        r.metrics.get("threshold", 0),
        "n_features":       r.metrics.get("n_features", 0),
        "n_train_samples":  r.metrics.get("n_train_samples", 0),
        "confusion_matrix": r.metrics.get("confusion_matrix", [[0,0],[0,0]]),
    }


@app.post("/predict")
async def predict_endpoint(sample: TelecomSample):
    _check_model()
    feat_dict = sample.to_feature_dict()
    pred      = predict(feat_dict, state.model_result)
    event     = _build_event(pred, feat_dict)
    _broadcast(event)
    return {"score": pred.score, "is_anomaly": pred.is_anomaly,
            "level": pred.anomaly_level, "confidence": pred.confidence}


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
# WEBSOCKET — fila por cliente
# ==========================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    client = WSClient(websocket)
    state.clients.append(client)
    log.info(f"WS conectado. Clientes: {len(state.clients)}")

    sender_task = asyncio.create_task(client.sender())

    try:
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=45.0)
            except asyncio.TimeoutError:
                client.enqueue('{"type":"ping"}')
                continue

            data = json.loads(raw)

            if data.get("type") == "ping":
                client.enqueue(json.dumps({"type": "pong", "ts": time.time()}))
                continue

            if data.get("type") == "sample":
                _check_model()
                feat_dict = TelecomSample(**data["payload"]).to_feature_dict()
                pred      = predict(feat_dict, state.model_result)
                event     = _build_event(pred, feat_dict)
                _broadcast(event)

    except WebSocketDisconnect:
        log.info("WS desconectado normalmente.")
    except Exception as e:
        log.warning(f"WS encerrado: {e}")
    finally:
        client.alive = False
        sender_task.cancel()
        if client in state.clients:
            state.clients.remove(client)
        log.info(f"Clientes restantes: {len(state.clients)}")


# ==========================================
# SIGNAL PARA PRODUCER
# ==========================================

@app.get("/signal/start-streaming")
async def signal_start_streaming():
    """Envia sinal 'start_streaming' para todos os clientes WebSocket (producers aguardando)."""
    signal_msg = json.dumps({"type": "start_streaming"})
    for client in state.clients:
        client.enqueue(signal_msg)
    log.info(f"Sinal de início enviado para {len(state.clients)} clientes")
    return {"status": "signal_sent", "clients": len(state.clients)}


# ==========================================
# SIMULADOR SINTETICO
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
    _check_model()
    rng = np.random.default_rng(42)
    log.info(f"Simulacao sintetica iniciada ({interval}s/sample)")

    while True:
        is_anomaly = rng.random() < 0.08
        if is_anomaly:
            typ = rng.integers(0, 3)
            if typ == 0:
                rx, tx, pl, jit, buf = rng.uniform(3,12), rng.uniform(2,10), rng.uniform(20,50), rng.uniform(40,100), int(rng.integers(5,20))
            elif typ == 1:
                rx, tx, pl, jit, buf = rng.uniform(90,140), rng.uniform(80,120), rng.uniform(1,5), rng.uniform(5,20), int(rng.integers(10,30))
            else:
                rx, tx, pl, jit, buf = rng.uniform(30,55), rng.uniform(25,45), rng.uniform(10,30), rng.uniform(25,70), int(rng.integers(3,10))
        else:
            rx, tx, pl, jit, buf = rng.uniform(40,55), rng.uniform(35,50), rng.uniform(0,2), rng.uniform(0,6), int(rng.integers(0,2))

        sample = {
            "rx_speed": float(rx), "tx_speed": float(tx),
            "rx_total_bytes": float(rng.uniform(1e6,5e6)),
            "tx_total_bytes": float(rng.uniform(1e6,4e6)),
            "received_packets": float(rng.uniform(100,500)),
            "dropped_packets": float(pl * 5),
            "buffer_overruns": float(buf),
            "parse_errors": float(rng.integers(0,3)),
            "throughput": float(rx+tx),
            "packet_rate": float(rng.uniform(10,60)),
            "packet_loss_rate": float(pl),
            "byte_rate": float(rng.uniform(1e4,5e4)),
            "jitter": float(jit),
        }
        pred  = predict(sample, state.model_result)
        event = _build_event(pred, sample)
        event["simulated"] = True
        _broadcast(event)
        await asyncio.sleep(interval)

# %%
