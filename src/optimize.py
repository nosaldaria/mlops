import pandas as pd
import optuna
import mlflow
import hydra
import joblib
import os
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


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

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.sklearn.log_model(best_model, "model")

        print(f"Оптимізація завершена. Найкращий F1: {study.best_value:.4f}")


if __name__ == "__main__":
    main()