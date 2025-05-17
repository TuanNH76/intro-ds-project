import asyncio
from datetime import datetime, timedelta

from scripts.fetch_data import (
    process_mentions_data,
)

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


def run_process_mentions():
    execution_date = datetime(2025, 5, 1, 0, 0, 0)
    _ = asyncio.run(process_mentions_data(execution_date, 3600, 3600))


default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fetch_all_ner_count_dag",
    default_args=default_args,
    schedule_interval=None,  # every 3 hours
    catchup=False,
    tags=["news", "ner"],
) as dag:
    process_mentions_task = PythonOperator(
        task_id="process_mentions_all_time",
        python_callable=run_process_mentions,
    )
