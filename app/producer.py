# ==========================================
# TELECOM ANOMALY — PRODUCER
# Lê os CSV reais linha a linha e envia
# para o servidor via WebSocket, simulando
# ingestão de dados em tempo real.
#
# Execute: python producer.py
# Execute com loop: python producer.py --loop
# ==========================================

import argparse
import asyncio
import json
import logging
import time

import numpy as np
import pandas as pd
import websockets

from core.pipeline import clean_dataframe, create_telecom_features, load_files

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# Mesmos arquivos do train.py
FILES = {
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

# Todas as features geradas pelo pipeline (deve bater com FEATURE_COLS do pipeline.py)
FEATURE_COLS = [
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


def prepare_dataframe() -> pd.DataFrame:
    """Carrega, limpa e extrai features de todos os CSV."""
    log.info("Carregando CSV...")
    df_n_raw, df_f_raw = load_files(FILES)

    df_n = create_telecom_features(clean_dataframe(df_n_raw))
    df_f = create_telecom_features(clean_dataframe(df_f_raw))

    df_n["label"] = 0
    df_f["label"] = 1

    df = pd.concat([df_n, df_f], ignore_index=True)
    df = df.sort_values("time_s").reset_index(drop=True)

    log.info(
        f"Total: {len(df)} amostras | "
        f"Normal: {(df.label==0).sum()} | "
        f"Failure: {(df.label==1).sum()}"
    )
    return df


async def stream(
    ws_url:       str,
    interval_s:   float,
    loop:         bool,
    ping_timeout: int = 30,
):
    """Envia o dataset linha a linha via WebSocket."""
    df        = prepare_dataframe()
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    log.info(f"Conectando em {ws_url} ...")

    while True:
        try:
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=ping_timeout,
                close_timeout=10,
            ) as ws:
                log.info("✅ WebSocket conectado. Iniciando streaming...")

                sent = 0
                t0   = time.time()

                for _, row in df.iterrows():
                    payload = {col: float(row.get(col, 0)) for col in feat_cols}
                    payload = {k: v if np.isfinite(v) else 0.0 for k, v in payload.items()}

                    await ws.send(json.dumps({"type": "sample", "payload": payload}))
                    sent += 1

                    if sent % 100 == 0:
                        rate = sent / (time.time() - t0)
                        log.info(f"  Enviadas {sent}/{len(df)} | {rate:.1f} amostras/s")

                    await asyncio.sleep(interval_s)

                log.info(f"✅ Dataset completo ({sent} amostras enviadas).")

                if not loop:
                    break
                log.info("Reiniciando (loop=True)...")

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            log.warning(f"Conexão perdida: {e}. Reconectando em 3s...")
            await asyncio.sleep(3)


async def post_stream(api_url: str, interval_s: float):
    """Alternativa via HTTP POST /predict (sem WebSocket)."""
    import httpx

    df        = prepare_dataframe()
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]

    log.info(f"Enviando via POST para {api_url}/predict ...")
    async with httpx.AsyncClient(timeout=5) as client:
        for _, row in df.iterrows():
            payload = {col: float(row.get(col, 0)) for col in feat_cols}
            payload = {k: v if np.isfinite(v) else 0.0 for k, v in payload.items()}

            try:
                r      = await client.post(f"{api_url}/predict", json=payload)
                result = r.json()
                log.info(f"  score={result.get('score', 0):.4f}  level={result.get('level', '?')}")
            except Exception as e:
                log.warning(f"  Erro POST: {e}")

            await asyncio.sleep(interval_s)


# ==========================================
# ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom Anomaly — Producer")
    parser.add_argument("--host",         default="localhost",  help="Host do servidor")
    parser.add_argument("--port",         default=8000, type=int)
    parser.add_argument("--interval",     default=0.5,  type=float,
                        help="Segundos entre amostras (default 0.5s)")
    parser.add_argument("--loop",         action="store_true",
                        help="Reiniciar dataset ao terminar")
    parser.add_argument("--mode",         choices=["ws", "http"], default="ws",
                        help="Protocolo: ws (WebSocket) ou http (POST)")
    parser.add_argument("--ping-timeout", default=30, type=int,
                        help="Timeout ping WebSocket em segundos")
    args = parser.parse_args()

    if args.mode == "ws":
        asyncio.run(stream(
            ws_url=f"ws://{args.host}:{args.port}/ws",
            interval_s=args.interval,
            loop=args.loop,
            ping_timeout=args.ping_timeout,
        ))
    else:
        asyncio.run(post_stream(
            api_url=f"http://{args.host}:{args.port}",
            interval_s=args.interval,
        ))
