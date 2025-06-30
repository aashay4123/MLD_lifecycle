from zenml.steps import step, Output
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
import pandas as pd
import joblib
import mlflow
from typing import List, Tuple, Any
from sklearn.metrics import accuracy_score, f1_score, log_loss
from src.utils.scripts.monitor import monitor
from src.Stage_8_HP_tuning.HP_tuning_optuna import (
    hp_tuning_optuna_probabilistic_approach,
)
from src.Stage_8_HP_tuning.ensemble_selector_optuna import ensemble_selector_optuna
from src.Stage_7_Evaluation.Evaluate_register import evaluate_and_register
from src.Stage_10_Deploy.Deploy import deploy_model


@enable_mlflow
@step
@monitor(name="load_production_data")
def load_production_data() -> Output(train=pd.DataFrame, val=pd.DataFrame):
    train = pd.read_parquet("artifacts/final/production/train.parquet")
    val = pd.read_parquet("artifacts/final/production/val.parquet")
    return train, val


@enable_mlflow
@step
@monitor(name="optuna_hpo_step")
def tune_step(
    train: pd.DataFrame, val: pd.DataFrame
) -> Output(models=List[Tuple[str, Any]]):
    final_model, best_params, metrics = hp_tuning_optuna_probabilistic_approach(
        train, val
    )
    return [(best_params["model"], final_model)]


@enable_mlflow
@step
@monitor(name="optuna_ensemble_step")
def ensemble_step(
    train: pd.DataFrame, val: pd.DataFrame, models: List[Tuple[str, Any]]
) -> Output(model=Any, score=float):
    final_model, score = ensemble_selector_optuna(train, val, models, top_k=3)
    joblib.dump(final_model, "artifacts/final/final_model.joblib")
    mlflow.sklearn.log_model(final_model, artifact_path="final_ensemble_model")
    mlflow.log_metric("ensemble_score", score)
    return final_model, score


@enable_mlflow
@step
@monitor(name="evaluate_on_test")
def test_evaluator(model: Any) -> Output(metrics=dict):
    test = pd.read_parquet("artifacts/final/production/test.parquet")
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred, average="weighted"),
        "test_log_loss": log_loss(y_test, y_prob) if y_prob is not None else None,
    }

    for k, v in metrics.items():
        mlflow.log_metric(k, v)

    return metrics


@enable_mlflow
@pipeline(enable_cache=True)
def optuna_hpo_with_ensemble_pipeline(
    load_production_data, tune_step, ensemble_step, evaluate_and_register, deploy_model
):
    train, val = load_production_data()
    models = tune_step(train, val)
    model, score = ensemble_step(train, val, models)
    evaluate_and_register(model, val_metrics=score)
    deploy_model()


# Main
def OptunaHPO():
    optuna_hpo_with_ensemble_pipeline(
        load_production_data=load_production_data(),
        tune_step=tune_step(),
        ensemble_step=ensemble_step(),
        evaluate_and_register=evaluate_and_register(),
        deploy_model=deploy_model(),
    ).run()
