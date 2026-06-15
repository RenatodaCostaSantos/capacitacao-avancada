# Telecom Anomaly Detection — Pipeline Completo

Sistema de detecção de anomalias em tempo real com IsolationForest,
FastAPI, WebSocket e dashboard live.

---

## Estrutura

```
telecom_anomaly/
├── core/
│   └── pipeline.py      ← treino, features, predict (teu código original)
├── frontend/
│   └── index.html       ← dashboard live (abre no browser)
├── models/              ← gerado após treino
│   ├── isolation_forest.joblib
│   ├── scaler.joblib
│   └── threshold.joblib
├── data_pre/            ← teus CSV aqui
├── train.py             ← STEP 1: treinar e salvar modelo
├── server.py            ← STEP 2: API FastAPI + WebSocket
├── producer.py          ← STEP 3: streaming dos CSV reais
└── requirements.txt
```

---

## Setup

```bash
# 1. instalar dependências
pip install -r requirements.txt

# 2. colocar os CSV em data_pre/
```

---

## Uso — 3 terminais

### Terminal 1 — Treinar o modelo
```bash
python train.py
```
Isso salva os arquivos em `models/`. Só precisa rodar uma vez.

---

### Terminal 2 — Subir o servidor
```bash
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```
Acesse o dashboard em: **http://localhost:8000**
Documentação da API: **http://localhost:8000/docs**

---

### Terminal 3 — Opção A: streaming dos CSV reais
```bash
# envia 10 amostras por segundo via WebSocket
python producer.py --interval 0.1

# mais lento, para debug
python producer.py --interval 1.0

# loop infinito (reinicia o dataset ao terminar)
python producer.py --interval 0.1 --loop

# alternativa via HTTP POST
python producer.py --mode http --interval 0.5
```

### Terminal 3 — Opção B: simulação sintética (sem CSV)
```
No próprio dashboard, clique em "▶ Iniciar simulação"
Ou via API: GET http://localhost:8000/simulate/start?interval_ms=500
```

---

## Endpoints da API

| Método    | Endpoint              | Descrição                              |
|-----------|-----------------------|----------------------------------------|
| GET       | /                     | Dashboard HTML                         |
| GET       | /health               | Status do servidor + modelo            |
| GET       | /model/info           | Features, threshold, parâmetros        |
| POST      | /predict              | Predição de 1 amostra (JSON)           |
| GET       | /alerts               | Histórico de alertas                   |
| GET       | /stats                | Métricas da sessão                     |
| GET       | /simulate/start       | Inicia gerador sintético               |
| GET       | /simulate/stop        | Para gerador                           |
| WebSocket | /ws                   | Stream bidirecional                    |

---

## Exemplo POST /predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rx_speed": 8.5,
    "tx_speed": 6.2,
    "rx_total_bytes": 1500000,
    "tx_total_bytes": 1200000,
    "received_packets": 250,
    "dropped_packets": 45,
    "buffer_overruns": 12,
    "parse_errors": 2
  }'
```

Resposta:
```json
{
  "score": -0.2341,
  "is_anomaly": true,
  "level": "critical",
  "confidence": 0.87
}
```

---

## Protocolo WebSocket

### Enviar amostra
```json
{ "type": "sample", "payload": { "rx_speed": 45.0, "tx_speed": 38.5, ... } }
```

### Receber evento
```json
{
  "ts": 1719000000.0,
  "score": -0.23,
  "is_anomaly": true,
  "level": "critical",
  "confidence": 0.87,
  "throughput": 83.5,
  "rx_speed": 45.0,
  "tx_speed": 38.5,
  "packet_loss": 2.1,
  "jitter": 12.4,
  "buffer_overruns": 3,
  "alert": {
    "ts": 1719000000.0,
    "level": "critical",
    "score": -0.23,
    "message": "Queda crítica de throughput: 12.3 Mb/s"
  }
}
```

---

## Níveis de alerta

| Level      | Condição                          |
|------------|-----------------------------------|
| `normal`   | score >= threshold                |
| `warning`  | score < threshold, margem < 0.15  |
| `critical` | score < threshold, margem >= 0.15 |
