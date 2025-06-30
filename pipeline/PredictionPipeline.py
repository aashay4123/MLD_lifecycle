# test_pipeline.py

from zenml.pipelines import pipeline
from zenml.steps import step, Output
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow

import pandas as pd
import mlflow
import os
import json
from typing import Tuple, Any
from sklearn.metrics import accuracy_score, f1_score

from utils.monitor import monitor


@step
@monitor(name="load_raw_test_data")
def load_test_data() -> Output(test=pd.DataFrame):
    return pd.read_parquet("artifacts/splits/test.parquet")


@step
@monitor(name="load_reference_data")
def load_reference_encoded() -> Output(ref=pd.DataFrame):
    return pd.read_parquet("artifacts/final_data/test_encoded.parquet")


@step
@monitor(name="load_all_components", log_result=True)
def load_transform_artifacts() -> (
    Output(imputer=Any, transformer=Any, encoder=Any, model=Any)
):
    imputer = mlflow.sklearn.load_model("artifacts:/imputer_model/production")
    transformer = mlflow.sklearn.load_model("artifacts:/scaler_model/production")
    encoder = mlflow.sklearn.load_model("artifacts:/encoder_model/production")
    baseline_model = mlflow.sklearn.load_model("artifacts:/baseline_model/production")
    # model = mlflow.sklearn.load_model("models:/best_model/production")
    return imputer, transformer, encoder, baseline_model


@enable_mlflow
@step
@monitor(name="apply_transforms_and_predict", log_result=True)
def apply_and_predict(
    test: pd.DataFrame,
    ref: pd.DataFrame,
    imputer: Any,
    transformer: Any,
    encoder: Any,
    model: Any,
) -> Output(predictions=pd.DataFrame, metrics=dict, match=bool):

    # Step 1: Impute
    test_i = imputer.transform(test)

    # Step 2: Transform/scale
    numeric = test_i.select_dtypes(include="number").columns.tolist()
    test_t = transformer.transform(test_i[numeric])
    test_i[numeric] = test_t

    # Step 3: Encode
    test_e = encoder.encode_test(test_i)

    # Step 4: Compare with reference
    match = test_e.equals(ref)
    if not match:
        test_e.to_csv("artifacts/results/retransformed_test.csv", index=False)
        ref.to_csv("artifacts/results/original_encoded_test.csv", index=False)
        mlflow.log_artifact("artifacts/results/retransformed_test.csv")
        mlflow.log_artifact("artifacts/results/original_encoded_test.csv")

    # Step 5: Predict
    if "target" in test_e.columns:
        X_test = test_e.drop(columns=["target"])
        y_true = test_e["target"]
    else:
        X_test = test_e
        y_true = None

    y_pred = model.predict(X_test)
    pred_df = pd.DataFrame({"prediction": y_pred})
    pred_df.to_csv("artifacts/results/predictions.csv", index=False)
    mlflow.log_artifact("artifacts/results/predictions.csv")

    metrics = {}
    if y_true is not None:
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1_score"] = f1_score(y_true, y_pred, average="weighted")
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})

    return pred_df, metrics, match


@pipeline(enable_cache=False)
def test_pipeline(
    load_test_data, load_reference_encoded, load_transform_artifacts, apply_and_predict
):
    test = load_test_data()
    ref = load_reference_encoded()
    imputer, transformer, encoder, model = load_transform_artifacts()
    apply_and_predict(test, ref, imputer, transformer, encoder, model)


def PredictionPipeline():
    test_pipeline(
        load_test_data=load_test_data(),
        load_reference_encoded=load_reference_encoded(),
        load_transform_artifacts=load_transform_artifacts(),
        apply_and_predict=apply_and_predict(),
    ).run()
