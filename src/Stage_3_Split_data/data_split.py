from typing import Tuple
import os
from typing_extensions import Annotated
import pandas as pd
import mlflow
from zenml import step
from zenml.steps import get_step_context
from src.utils.monitor import monitor
from configs import global_conf
from .BaselineModel import AutoBaseline
from .ThreeWaySplit import SplitThreeWay
import json

DATASET_TARGET_COLUMN_NAME = "label"


@step
@monitor(name="baseline_step", track_memory=True, track_input_size=True)
def baseline(train: pd.DataFrame, test: pd.DataFrame) -> None:
    if train.empty or test.empty:
        raise ValueError("Train or test data is empty. Cannot run baseline model.")

    baseline_model = AutoBaseline(target=DATASET_TARGET_COLUMN_NAME)
    baseline_results = baseline_model.run(train, test)

    baseline_report_dir = global_conf.BASELINE_REPORT_PATH
    os.makedirs(baseline_report_dir, exist_ok=True)

    baseline_report_path = os.path.join(baseline_report_dir, "baseline_metrics.json")
    with open(baseline_report_path, "w") as f:
        json.dump(baseline_results, f, indent=2)

    # MLflow logging
    with mlflow.start_run(run_name="baseline", nested=True):
        mlflow.log_param("baseline_type", "DummyClassifier")
        for metric, value in baseline_results.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(metric, value)
            else:
                mlflow.log_param(metric, str(value))
        mlflow.log_artifact(baseline_report_path)


@step
def data_splitter(
    data: pd.DataFrame,
    target: str = DATASET_TARGET_COLUMN_NAME,
    stratify: bool = True,
    oversample: bool = False,
    seed: int = 42,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    if data.empty:
        raise ValueError("Input data is empty. Cannot split data.")
    splitter = SplitThreeWay(
        data=data, stratify=stratify, seed=seed, oversample=oversample, target=target
    )
    train, test, val = splitter.split_data()

    # Save split summary as artifact
    split_summary = {
        "train_rows": len(train),
        "test_rows": len(test),
        "val_rows": len(val),
        "stratify": stratify,
        "oversample": oversample,
        "seed": seed,
    }
    split_report_dir = global_conf.SPLIT_PARQUET_PATH
    os.makedirs(split_report_dir, exist_ok=True)

    summary_path = os.path.join(split_report_dir, "split_summary.json")

    with open(summary_path, "w") as f:
        json.dump(split_summary, f, indent=2)

    # MLflow logging
    with mlflow.start_run(run_name="data_splitter", nested=True):
        mlflow.log_param("stratify", stratify)
        mlflow.log_param("oversample", oversample)
        mlflow.log_param("seed", seed)
        mlflow.log_metric("train_rows", len(train))
        mlflow.log_metric("test_rows", len(test))
        mlflow.log_metric("val_rows", len(val))
        mlflow.log_artifact(summary_path)

    return train, test, val
