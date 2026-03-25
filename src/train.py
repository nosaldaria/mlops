import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import joblib

def train(data_dir, model_dir):
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("DVC_Pipeline_Run")

    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    X_train = train_df.drop('Churn', axis=1)
    y_train = train_df['Churn']

    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)

        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.joblib")
        joblib.dump(model, model_path)

        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained, saved locally to {model_path} and logged to MLflow")

if __name__ == "__main__":
    train(sys.argv[1], sys.argv[2])