import logging
import os
from datetime import timedelta
from typing import Any

import pendulum
from airflow.models.dag import DAG
from airflow.models.param import Param
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule

logger = logging.getLogger(__name__)

DAG_ID = "retrain_trigger"
TRAINING_MODULE_ROOT = "training"
TRAINING_PYTHON = os.getenv("TRAINING_PYTHON", "/opt/training-venv/bin/python")


def _resolve_drifted_metrics(**context: Any) -> str:
    dag_run = context["dag_run"]
    conf = dag_run.conf or {}
    drifted_metrics = conf.get("drifted_metrics") or []
    triggered_at = conf.get("triggered_at") or pendulum.now("UTC").to_iso8601_string()
    if not drifted_metrics:
        logger.info("retrain_trigger received empty drifted_metrics, proceeding with full retrain")
    joined = ",".join(str(metric) for metric in drifted_metrics)
    task_instance = context["ti"]
    task_instance.xcom_push(key="drifted_metrics_csv", value=joined)
    task_instance.xcom_push(key="triggered_at", value=triggered_at)
    logger.info("retrain_trigger resolved drifted_metrics=%s triggered_at=%s", joined, triggered_at)
    return joined


def _log_completion(**context: Any) -> None:
    task_instance = context["ti"]
    drifted_metrics_csv = task_instance.xcom_pull(task_ids="resolve_drifted_metrics", key="drifted_metrics_csv")
    logger.info("retrain_trigger completed retrain cycle for metrics=%s", drifted_metrics_csv)


default_args: dict[str, Any] = {
    "owner": "llm-eval-platform",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

with DAG(
    dag_id=DAG_ID,
    description="Retrains the DistilBERT evaluator when drift is detected",
    default_args=default_args,
    schedule=None,
    start_date=pendulum.datetime(2024, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=6),
    tags=["llm-eval", "retrain", "drift"],
    params={
        "drifted_metrics": Param(default=[], type="array"),
        "triggered_at": Param(default="", type="string"),
    },
) as dag:
    resolve_drifted_metrics = PythonOperator(
        task_id="resolve_drifted_metrics",
        python_callable=_resolve_drifted_metrics,
    )

    build_dataset = BashOperator(
        task_id="build_dataset",
        bash_command=f"{TRAINING_PYTHON} -m {TRAINING_MODULE_ROOT}.dataset_builder",
        env={
            "DRIFTED_METRICS": "{{ ti.xcom_pull(task_ids='resolve_drifted_metrics', key='drifted_metrics_csv') }}",
            "TRIGGERED_AT": "{{ ti.xcom_pull(task_ids='resolve_drifted_metrics', key='triggered_at') }}",
        },
        append_env=True,
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"{TRAINING_PYTHON} -m {TRAINING_MODULE_ROOT}.train",
        env={
            "DRIFTED_METRICS": "{{ ti.xcom_pull(task_ids='resolve_drifted_metrics', key='drifted_metrics_csv') }}",
        },
        append_env=True,
    )

    evaluate_model = BashOperator(
        task_id="evaluate_model",
        bash_command=f"{TRAINING_PYTHON} -m {TRAINING_MODULE_ROOT}.evaluate",
        append_env=True,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=f"{TRAINING_PYTHON} -m {TRAINING_MODULE_ROOT}.register_model",
        append_env=True,
    )

    log_completion = PythonOperator(
        task_id="log_completion",
        python_callable=_log_completion,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    resolve_drifted_metrics >> build_dataset >> train_model >> evaluate_model >> register_model >> log_completion
