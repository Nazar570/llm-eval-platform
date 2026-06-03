import logging
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pendulum
import yaml
from airflow.models.dag import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

DAG_ID = "eval_pipeline"
TRAINING_MODULE_ROOT = "training"
TRAINING_PYTHON = os.getenv("TRAINING_PYTHON", "/opt/training-venv/bin/python")
DEFAULT_SETTINGS_PATH = "/opt/airflow/config/settings.yaml"
DEFAULT_SCHEDULE_INTERVAL_HOURS = 6


def _load_schedule_interval() -> timedelta:
    settings_path = Path(os.getenv("EVAL_CONFIG_PATH", DEFAULT_SETTINGS_PATH))
    hours = DEFAULT_SCHEDULE_INTERVAL_HOURS
    try:
        with settings_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        batch_eval = payload.get("batch_eval") or {}
        hours = int(batch_eval.get("schedule_interval_hours", DEFAULT_SCHEDULE_INTERVAL_HOURS))
    except (OSError, ValueError, TypeError):
        logger.exception("eval_pipeline failed to load schedule from %s, using default", settings_path)
    return timedelta(hours=hours)


def _log_batch_start(**context: Any) -> None:
    logical_date = context["logical_date"]
    logger.info("eval_pipeline starting batch evaluation window logical_date=%s", logical_date)


def _log_batch_completion(**context: Any) -> None:
    logical_date = context["logical_date"]
    logger.info("eval_pipeline completed batch evaluation window logical_date=%s", logical_date)


default_args: dict[str, Any] = {
    "owner": "llm-eval-platform",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "depends_on_past": False,
}

with DAG(
    dag_id=DAG_ID,
    description="Scheduled batch evaluation of stored QA traces",
    default_args=default_args,
    schedule=_load_schedule_interval(),
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=3),
    tags=["llm-eval", "batch", "evaluation"],
) as dag:
    log_batch_start = PythonOperator(
        task_id="log_batch_start",
        python_callable=_log_batch_start,
    )

    run_batch_evaluation = BashOperator(
        task_id="run_batch_evaluation",
        bash_command=f"{TRAINING_PYTHON} -m {TRAINING_MODULE_ROOT}.batch_eval",
        append_env=True,
    )

    log_batch_completion = PythonOperator(
        task_id="log_batch_completion",
        python_callable=_log_batch_completion,
    )

    log_batch_start >> run_batch_evaluation >> log_batch_completion
