import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import mlflow
import mlflow.sklearn
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

parser = argparse.ArgumentParser(description="Train RandomForest for Telco Churn with MLflow")

parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
parser.add_argument("--min_samples_split", type=int, default=2)
parser.add_argument("--min_samples_leaf", type=int, default=1)
parser.add_argument("--max_features", type=str, default="sqrt")

parser.add_argument("--dataset_version", type=str, default="v1")
parser.add_argument("--author", type=str, default="YourName")

args = parser.parse_args()

mlflow.set_tracking_uri("http://127.0.0.1:5000") 
mlflow.set_experiment("mlops_lab_1_churn")

df = pd.read_csv("/Users/daria/PycharmProjects/mlops/mlops_lab_1/data/raw/dataset.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.dropna()

target = "Churn"
X = df.drop(columns=[target, "customerID"])
y = df[target]

categorical_cols = X.select_dtypes(include=['object']).columns
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=42,
        n_jobs=-1
    )

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)

    print(f"Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    print(f"Training time: {training_time:.4f} seconds")

    mlflow.log_params(vars(args))

    mlflow.log_metrics({
        "train_accuracy": train_acc,
        "train_f1": train_f1,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "training_time": training_time
    })

    mlflow.set_tags({
        "author": args.author,
        "dataset_version": args.dataset_version,
        "model_type": "RandomForest"
    })

    mlflow.sklearn.log_model(model, "model")

    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10,6))
    sns.barplot(x=feature_importances, y=feature_importances.index)
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

print("Training completed and logged to MLflow!")
