import pandas as pd
import optuna
import mlflow
import hydra
import joblib
import os
import json
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def load_data(cfg):
    train = pd.read_csv(cfg.data.train_path)
    test = pd.read_csv(cfg.data.test_path)
    return train.drop('Churn', axis=1), train['Churn'], test.drop('Churn', axis=1), test['Churn']


def objective(trial, cfg, X_train, y_train, X_test, y_test):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", cfg.random_forest.n_estimators.low,
                                          cfg.random_forest.n_estimators.high),
        "max_depth": trial.suggest_int("max_depth", cfg.random_forest.max_depth.low, cfg.random_forest.max_depth.high),
        "min_samples_split": trial.suggest_int("min_samples_split", cfg.random_forest.min_samples_split.low,
                                               cfg.random_forest.min_samples_split.high),
        "random_state": cfg.seed
    }

    with mlflow.start_run(nested=True, run_name=f"Trial_{trial.number}"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        score = f1_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("f1", score)

        return score


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, y_train, X_test, y_test = load_data(cfg)

    with mlflow.start_run(run_name="Optuna_Parent_Run"):
        sampler = optuna.samplers.TPESampler(
            seed=cfg.seed) if cfg.hpo.sampler == "tpe" else optuna.samplers.RandomSampler(seed=cfg.seed)

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: objective(trial, cfg, X_train, y_train, X_test, y_test), n_trials=cfg.hpo.n_trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1", study.best_value)

        best_model = RandomForestClassifier(**study.best_params, random_state=cfg.seed)
        best_model.fit(X_train, y_train)

        y_pred = best_model.predict(X_test)
        final_f1 = f1_score(y_test, y_pred)
        final_acc = accuracy_score(y_test, y_pred)

        os.makedirs("models", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        joblib.dump(best_model, "models/model.pkl")
        mlflow.sklearn.log_model(best_model, "model")

        metrics = {
            "f1": float(final_f1),
            "accuracy": float(final_acc)
        }
        with open("reports/metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title("Confusion Matrix - Best Model")
        plt.savefig("reports/confusion_matrix.png")
        mlflow.log_artifact("reports/confusion_matrix.png")

        print(f"Оптимізація завершена. Найкращий F1: {final_f1:.4f}")


if __name__ == "__main__":
    main()