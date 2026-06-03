# LLM Evaluation Platform

Local-first platform for evaluating and monitoring a retrieval-augmented LLM application. A FastAPI RAG service answers questions over a vector store and publishes every question/answer trace to Kafka. An async consumer picks up each trace, scores it with RAGAS, BERTScore and an LLM judge, and writes the results to Postgres. A DistilBERT judge — fine-tuned with LoRA and tracked in MLflow — scores stored traces in batch, a drift detector watches the score distributions, and when they move it triggers a retraining DAG in Airflow. Prometheus and Grafana sit on top so you can see quality, latency and drift over time.

---

## Stack

- **API:** FastAPI + Uvicorn
- **Async messaging:** Apache Kafka (aiokafka)
- **Generation:** Ollama (llama3.2:3b)
- **Vector store:** ChromaDB
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **Online scoring:** RAGAS (faithfulness, answer relevancy, context recall), BERTScore, LLM-as-judge
- **Offline judge:** DistilBERT fine-tuned with LoRA (PEFT + Transformers)
- **Experiment tracking & registry:** MLflow
- **Primary database:** PostgreSQL + SQLAlchemy + Alembic
- **Cache:** Redis
- **Annotation:** Label Studio
- **Drift detection:** Kolmogorov–Smirnov test over rolling windows
- **Orchestration:** Apache Airflow
- **Containerization:** Docker Compose
- **Observability:** Prometheus + Pushgateway, Grafana
- **Quality:** ruff, black, mypy, pytest, pre-commit

---

## Setup

```bash
git clone https://github.com/Nazar570/llm-eval-platform.git
cd llm-eval-platform

python3 -m venv .venv
source .venv/bin/activate

pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r training/requirements.txt
```

Copy the example env. The defaults already point at the local stack, so there are no keys to fill in:

```bash
cp .env.example .env
```

---

## Services

The whole stack runs through Docker Compose. The first start pulls the Ollama model (~2 GB), so give it a minute.

### Start services

```bash
make up
```

```text
# Postgres, Kafka, MLflow, ChromaDB, Ollama,
# Redis, Label Studio, Airflow, Prometheus, Grafana
```

```bash
make ps
make logs
make down
```

Apply database migrations against the local Postgres:

```bash
make migrate
```

---

## The evaluation loop

One command per phase, all runnable from the host:

```bash
python -m training.dataset_builder
# build data/eval_dataset.jsonl from Label Studio + weak labels

python -m training.train
# fine-tune the DistilBERT judge, log the run to MLflow

python -m training.evaluate
# score the holdout, enforce the F1 quality gate

python -m training.register_model
# push the model to the MLflow registry and Postgres

python -m training.batch_eval
# score unscored traces with the registered judge

python -m training.eval_report
# write data/eval_report.md
```

---

## Endpoints

### UIs

| Service | URL |
|----------|-----|
| RAG API docs | http://localhost:8001/docs |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8081 *(admin / admin)* |
| Grafana | http://localhost:3000 *(admin / admin)* |
| Prometheus | http://localhost:9090 |
| Label Studio | http://localhost:8080 |

### API

| Method | Endpoint | Description |
|----------|----------|-------------|
| GET | `/health` | liveness check |
| GET | `/metrics` | Prometheus metrics |
| POST | `/query` | ask a question and emit a trace |

Example request:

```bash
curl -X POST http://localhost:8001/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is retrieval augmented generation?",
    "ground_truth": "RAG grounds answers in retrieved context."
  }'
```

---

## Pipeline flow

```text
Question
    │
    ├─→ retrieve (ChromaDB)
    │
    ├─→ generate (Ollama)
    │
    └─→ publish trace to Kafka
                     │
                     ▼
eval consumer → RAGAS + BERTScore + LLM judge
                     │
                     ▼
                 Postgres
                     │
                     ▼
DistilBERT judge (LoRA, tracked in MLflow)
                     │
                     ▼
          batch scores in Postgres
                     │
                     ▼
         drift detector (KS test)
                     │
                     ▼
   triggers Airflow retrain DAG
                     │
                     ▼
     new registered model version
```

---

## Tests

```bash
make test
```

Run the full quality gate the same way CI does:

```bash
make check
```

```text
# ruff + black --check + mypy + pytest
```

---

## Observability

Every service exposes a Prometheus endpoint — the RAG app on 8001, the eval consumer on 8002 and the drift detector on 8003. Prometheus scrapes them plus the Pushgateway that batch jobs report to, and Grafana reads from Prometheus to chart score distributions, drift events and batch-eval throughput. Each MLflow run keeps the training params, metrics and the model artifact, and every registered version is mirrored into the `model_versions` table so the eval report can show the registry history without calling MLflow.