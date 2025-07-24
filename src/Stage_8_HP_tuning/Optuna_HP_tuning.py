from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import pandas as pd
from ensemble_selector_optuna import ensemble_selector_optuna
from HP_tuning_optuna import hp_tuning_optuna_probabilistic_approach
from zenml.integrations.mlflow.steps import enable_mlflow
from zenml.pipelines import pipeline
from zenml.steps import Output, step

from utils.monitor import monitor


@enable_mlflow
@step
@monitor(name="tune_model_optuna")
def tune_step(
    train: pd.DataFrame, val: pd.DataFrame
) -> Output(models=List[Tuple[str, Any]]):
    final_model, best_params, metrics = hp_tuning_optuna_probabilistic_approach(
        train, val
    )
    return [(best_params["model"], final_model)]


@enable_mlflow
@step
@monitor(name="ensemble_selector_optuna")
def ensemble_step(
    train: pd.DataFrame, val: pd.DataFrame, models: List[Tuple[str, Any]]
) -> Output(model=Any, score=float):
    final_model, score = ensemble_selector_optuna(train, val, models, top_k=3)
    joblib.dump(final_model, "artifacts/final_model.joblib")
    mlflow.sklearn.log_model(final_model, "final_ensemble_model")
    mlflow.log_metric("ensemble_score", score)
    return final_model, score


@pipeline
def full_hpo_ensemble_pipeline(
    ingest_data,
    inspect_data,
    split_data,
    impute_data,
    scale_transform,
    detect_outliers,
    feature_split,
    feature_construct,
    encode,
    tune_step,
    ensemble_step,
):
    data = ingest_data()
    checked = inspect_data(data)
    train, val, test = split_data(checked)
    train, val, test = impute_data(train, val, test)
    train, val, test = scale_transform(train, val, test)
    train, val, test = detect_outliers(train, val, test)
    train, val, test = feature_split(train, val, test)
    train, val, test = feature_construct(train, val, test)
    train, val, test = encode(train, val, test)
    models = tune_step(train, val)
    ensemble_step(train, val, models)
