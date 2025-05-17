# dags/price_fetch_dag.py
import asyncio
from datetime import timedelta

from scripts.fetch_data import fetch_price_coin

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago


def run_fetch_price(**_):
    asyncio.run(fetch_price_coin())

default_args = {
    "owner": "airflow",
    "start_date": days_ago(1),
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="price_fetch_dag",
    default_args=default_args,
    schedule_interval="0 */3 * * *",  # every 3 hours
    catchup=False,
    tags=["price", "coin"],
) as dag:
    fetch_price = PythonOperator(
        task_id="fetch_price",
        python_callable=run_fetch_price,
    )
