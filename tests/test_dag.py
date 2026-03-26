import os
from airflow.models import DagBag


def test_dag_import():
    os.environ["AIRFLOW__CORE__SQL_ALCHEMY_CONN"] = "sqlite:////tmp/airflow.db"

    dag_path = os.path.join(os.path.dirname(__file__), "..", "dags")

    dag_bag = DagBag(dag_folder=dag_path, include_examples=False)

    assert len(dag_bag.import_errors) == 0, f"Помилки в DAG: {dag_bag.import_errors}"


def test_dag_exists():
    dag_path = os.path.join(os.path.dirname(__file__), "..", "dags")
    dag_bag = DagBag(dag_folder=dag_path, include_examples=False)
    assert 'telco_churn_training' in dag_bag.dags, "DAG 'telco_churn_training' не знайдено"