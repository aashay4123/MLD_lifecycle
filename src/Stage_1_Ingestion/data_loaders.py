import os
import pandas as pd
import json
import mlflow
from zenml import step
from zenml.steps import get_step_context
from typing import Tuple
from .DataCollector import DataCollector
from .DataHealthCheck import DataHealthCheck
from configs import global_conf
from src.utils.monitor import monitor
import numpy as np

DATASET_TARGET_COLUMN_NAME = "label"

# DATALEAK by X wrt the target column if corr == 1 drops the column


def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


@step
@monitor(name="dataCheck_step", track_memory=True, track_input_size=True)
def dataCheck(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty. Cannot perform health check.")
    if DATASET_TARGET_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Target column '{DATASET_TARGET_COLUMN_NAME}' not found in DataFrame."
        )
    # Run health check
    health_check = DataHealthCheck(df)
    health_check.run_all()

    results = health_check.get_results()
    safe_results = convert_numpy_types(results)
    report_path = os.path.join("reports", "health_report", "data_health_report.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(safe_results, f, indent=2)

    # Log to MLflow
    with mlflow.start_run(run_name="data_health_check", nested=True):
        missing_frac = df.isnull().mean().mean()
        mlflow.log_metric("missing_fraction", missing_frac)
        mlflow.log_artifact(report_path)
        mlflow.log_text("\n".join(df.columns.tolist()[:10]), "columns.txt")

    return df


@step
def dataLoader(file: str = None, project: str = "Default") -> pd.DataFrame:
    if file is None:
        file = global_conf.CSV_PATH
        if not os.path.exists(file):
            raise ValueError(
                f"File '{file}' does not exist. Please provide a valid file path."
            )
    if not os.path.exists(file):
        raise ValueError(f"File '{file}' does not exist. Please check the file path.")

    dataCollector = DataCollector(suite_name=project)
    df = dataCollector.read_file(file)

    if df is None:
        raise ValueError(
            "DataFrame is None. Please check the file path or data source."
        )
    if df.empty:
        raise ValueError(
            "DataFrame is empty. Please check the file path or data source."
        )
    if DATASET_TARGET_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Target column '{DATASET_TARGET_COLUMN_NAME}' not found in DataFrame."
        )

    parquet_name = "data.parquet"
    os.makedirs(global_conf.RAW_PARQUET_PATH, exist_ok=True)
    parquet_path = os.path.join(global_conf.RAW_PARQUET_PATH, parquet_name)
    df.to_parquet(parquet_path, index=False)

    # Log to MLflow
    with mlflow.start_run(run_name="data_loader", nested=True):
        mlflow.log_param("project_name", project)
        mlflow.log_param("csv_path", file)
        mlflow.log_metric("num_rows", len(df))
        mlflow.log_metric("num_columns", df.shape[1])
        mlflow.log_artifact(parquet_path)

    return df
