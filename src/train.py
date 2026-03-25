import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn


def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def run_experiment(n_estimators, max_depth, learning_rate_tag):

    mlflow.set_tracking_uri("file:./mlruns")

    mlflow.set_experiment("Telco_Churn_Optimization")

    with mlflow.start_run():
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("developer", "Daria")
        mlflow.set_tag("iteration_type", learning_rate_tag)

        data_path = "/Users/daria/PycharmProjects/mlops/mlops_lab_1/data/raw/dataset.csv"

        X_train, X_test, y_train, y_test = load_and_preprocess(data_path)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Оцінка якості моделі [cite: 128]
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Логування гіперпараметрів та метрик [cite: 103, 107]
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Побудова та логування графіка важливості ознак (Feature Importance) [cite: 115]
        plt.figure(figsize=(10, 6))
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        importances.nlargest(10).plot(kind='barh')
        plt.title(f"Top 10 Features (Depth={max_depth})")
        plt.tight_layout()

        plot_path = "feature_importance.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)  # [cite: 96, 111]

        # Логування самої моделі [cite: 95, 109]
        mlflow.sklearn.log_model(model, "churn_model")

        print(f"Запуск успішний: Depth={max_depth}, F1={f1:.4f}")


if __name__ == "__main__":
    # Використання CLI аргументів для гнучкості запусків [cite: 114]
    parser = argparse.ArgumentParser(description="Train Telco Churn Model")
    parser.add_argument("--n_estimators", type=int, default=100, help="Кількість дерев")
    parser.add_argument("--max_depth", type=int, default=None, help="Максимальна глибина")
    parser.add_argument("--lr_tag", type=str, default="default", help="Тег для опису версії")

    args = parser.parse_args()

    run_experiment(args.n_estimators, args.max_depth, args.lr_tag)