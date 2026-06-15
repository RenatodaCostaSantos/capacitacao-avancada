# ==========================================
# TELECOM ANOMALY — FASTAPI SERVER
# uvicorn server:app --host 0.0.0.0 --port 8000
# ==========================================

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.pipeline import (
    PipelineConfig, PredictionResult, TrainResult,
    clean_dataframe, create_telecom_features, load_files, load_model, predict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


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
    model_result:  Optional[TrainResult] = None
    clients:       list[WSClient]        = []
    alert_history: list[dict]            = []
    stats:         dict                  = None

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
        log.warning("Modelo não encontrado. Execute train.py primeiro.")
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
    rx_tx_ratio:       Optional[float] = None
    byte_per_packet:   Optional[float] = None
    speed_delta:       Optional[float] = None
    throughput_change: Optional[float] = None
    rx_speed_mean_10:  Optional[float] = None
    rx_speed_std_10:   Optional[float] = None

    def to_feature_dict(self) -> dict:
        d = self.model_dump()
        if d["throughput"] is None:
            d["throughput"] = d["rx_speed"] + d["tx_speed"]
        for k in [
            "packet_rate", "packet_loss_rate", "byte_rate", "jitter",
            "rx_tx_ratio", "byte_per_packet", "speed_delta",
            "throughput_change", "rx_speed_mean_10", "rx_speed_std_10",
        ]:
            if d[k] is None:
                d[k] = 0.0
        return d


# ==========================================
# HELPERS
# ==========================================

def _check_model():
    if state.model_result is None:
        raise HTTPException(status_code=503, detail="Modelo não carregado.")


def _convert_to_serializable(obj):
    """Converte tipos NumPy para tipos Python nativos JSON-serializáveis."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(v) for v in obj]
    return obj


def _broadcast(event: dict):
    """Enfileira para todos os clientes — non-blocking."""
    payload = json.dumps(_convert_to_serializable(event))
    dead    = [c for c in state.clients if not c.alive]
    for c in dead:
        state.clients.remove(c)
    for c in state.clients:
        c.enqueue(payload)


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
        return f"Queda crítica de throughput: {tp:.1f} Mb/s"
    if pl > 15:
        return f"Packet loss elevado: {pl:.1f} pkt/s"
    if jit > 30:
        return f"Jitter instável: {jit:.1f} ms"
    if buf > 5:
        return f"Buffer overruns: {buf} eventos"
    return f"Degradação de link detectada — score {pred.score:.4f}"


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
        "confusion_matrix": r.metrics.get("confusion_matrix", [[0, 0], [0, 0]]),
    }


@app.post("/predict")
async def predict_endpoint(sample: TelecomSample):
    _check_model()
    feat_dict = sample.to_feature_dict()
    pred      = predict(feat_dict, state.model_result)
    event     = _build_event(pred, feat_dict)
    _broadcast(event)
    return {
        "score":      pred.score,
        "is_anomaly": pred.is_anomaly,
        "level":      pred.anomaly_level,
        "confidence": pred.confidence,
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
# SIMULADOR — DADOS REAIS DOS CSV
# ==========================================

# Mesmos arquivos do train.py
_CSV_FILES = {
    "normal": [
        r"data_sem_falha\carbonZ_2018-07-18-16-37-39_1_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-07-30-16-39-00_3_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-09-11-14-16-55_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-09-11-14-41-38_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-10-05-14-34-20_1_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-09-11-15-05-11_2_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-10-05-14-37-22_1_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-10-05-15-52-12_1_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-10-05-15-52-12_2_no_failure-diagnostics.csv",
        r"data_sem_falha\carbonZ_2018-10-18-11-08-24_no_failure-diagnostics.csv",
    ],
    "failure": [
        r"Data_com_falha\carbonZ_2018-07-18-15-53-31_1_engine_failure-diagnostics.csv",
        r"Data_com_falha\carbonZ_2018-07-18-15-53-31_2_engine_failure-diagnostics.csv",
        r"Data_com_falha\carbonZ_2018-07-30-16-39-00_1_engine_failure-diagnostics.csv",
        r"Data_com_falha\carbonZ_2018-07-30-16-39-00_2_engine_failure-diagnostics.csv",
        r"Data_com_falha\carbonZ_2018-09-11-11-56-30_engine_failure-diagnostics.csv",
        r"Data_com_falha\carbonZ_2018-09-11-14-22-07_1_engine_failure-diagnostics.csv",
        r"Data_com_falha\carbonZ_2018-09-11-14-22-07_2_engine_failure-diagnostics.csv",
    ],
}

_FEATURE_COLS = [
    "received_packets", "dropped_packets",
    "rx_total_bytes",   "tx_total_bytes",
    "rx_speed",         "tx_speed",
    "throughput",       "packet_rate",
    "packet_loss_rate", "byte_rate",
    "jitter",           "buffer_overruns",
    "parse_errors",     "rx_tx_ratio",
    "byte_per_packet",  "speed_delta",
    "throughput_change","rx_speed_mean_10",
    "rx_speed_std_10",
]

_sim_task: Optional[asyncio.Task] = None
_sim_df:   Optional[pd.DataFrame] = None


def _load_sim_dataframe() -> pd.DataFrame:
    """Carrega e prepara os CSV reais uma única vez (cache em memória)."""
    global _sim_df
    if _sim_df is not None:
        return _sim_df

    log.info("Carregando CSV para simulação...")
    df_n_raw, df_f_raw = load_files(_CSV_FILES)

    df_n = create_telecom_features(clean_dataframe(df_n_raw))
    df_f = create_telecom_features(clean_dataframe(df_f_raw))

    df_n["label"] = 0
    df_f["label"] = 1

    df = pd.concat([df_n, df_f], ignore_index=True)
    if "time_s" in df.columns:
        df = df.sort_values("time_s").reset_index(drop=True)

    log.info(
        f"CSV prontos: {len(df)} amostras | "
        f"Normal: {(df.label==0).sum()} | "
        f"Failure: {(df.label==1).sum()}"
    )
    _sim_df = df
    return _sim_df


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
        _sim_task = None
    return {"status": "stopped"}


@app.get("/simulate/status")
async def simulate_status():
    running = _sim_task is not None and not _sim_task.done()
    return {"running": running}


async def _simulate_loop(interval: float):
    """Reproduz os CSV reais linha a linha, em loop contínuo."""
    _check_model()

    try:
        df = await asyncio.to_thread(_load_sim_dataframe)
    except Exception as e:
        log.error(f"Erro ao carregar CSV: {e}")
        return

    feat_cols = [c for c in _FEATURE_COLS if c in df.columns]
    log.info(f"Simulação iniciada — {len(df)} amostras | {interval}s/amostra")

    while True:
        for _, row in df.iterrows():
            sample = {col: float(row.get(col, 0)) for col in feat_cols}
            sample = {k: v if np.isfinite(v) else 0.0 for k, v in sample.items()}

            pred  = predict(sample, state.model_result)
            event = _build_event(pred, sample)
            event["simulated"] = True
            _broadcast(event)

            await asyncio.sleep(interval)

        log.info("Dataset completo — reiniciando do início...")
