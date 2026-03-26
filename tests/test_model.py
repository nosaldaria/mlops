import os
import pandas as pd
import pytest
import json

def test_data_schema():
    data_path = "/Users/daria/PycharmProjects/mlops/mlops_lab_1/data/prepared/train.csv"

    assert os.path.exists(data_path), f"Файл даних {data_path} не знайдено!"

    df = pd.read_csv(data_path)

    required_cols = {"tenure", "MonthlyCharges", "TotalCharges", "Churn"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Відсутні колонки: {missing}"

    assert df["Churn"].notna().all(), "Цільова змінна 'Churn' містить пропуски!"

    assert df.shape[0] >= 100, "Замало даних для проведення експерименту!"

def test_artifacts_exist():
    assert os.path.exists("/Users/daria/PycharmProjects/mlops/mlops_lab_1/models/model.pkl"), "Артефакт моделі (model.pkl) не знайдено!"
    assert os.path.exists("/Users/daria/PycharmProjects/mlops/mlops_lab_1/reports/metrics.json"), "Файл метрик (metrics.json) не знайдено!"
    assert os.path.exists("/Users/daria/PycharmProjects/mlops/mlops_lab_1/reports/confusion_matrix.png"), "Візуалізація не знайдена!"


def test_quality_gate():
    threshold = float(os.getenv("F1_THRESHOLD", "0.50"))

    with open("/Users/daria/PycharmProjects/mlops/mlops_lab_1/reports/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    f1 = float(metrics["f1"])

    assert f1 >= threshold, f"Quality Gate не пройдено: F1 {f1:.4f} нижче порогу {threshold}"