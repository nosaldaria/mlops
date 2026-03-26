from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator, BranchPythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime
import json
import os

def check_model_quality():
    with open('reports/metrics.json', 'r') as f:
        metrics = json.load(f)
    if metrics.get('f1', 0) > 0.60:
        return 'register_model'
    return 'stop_pipeline'

with DAG(
    'telco_churn_training',
    start_date=datetime(2026, 3, 26),
    schedule='@weekly',
    catchup=False
) as dag:

    prepare_data = BashOperator(
        task_id='prepare_data',
        bash_command='cd /Users/daria/PycharmProjects/mlops/mlops_lab_1 && dvc repro prepare'
    )

    train_model = BashOperator(
        task_id='train_model',
        bash_command='cd /Users/daria/PycharmProjects/mlops/mlops_lab_1 && python src/optimize.py'
    )

    branching = BranchPythonOperator(
        task_id='evaluate_model',
        python_callable=check_model_quality
    )

    register = BashOperator(
        task_id='register_model',
        bash_command='echo "Registering model in MLflow Registry..."'
    )

    stop = BashOperator(
        task_id='stop_pipeline',
        bash_command='echo "Model quality too low. Stopping."'
    )

    prepare_data >> train_model >> branching >> [register, stop]