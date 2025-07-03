from pathlib import Path
import os
import json
import pandas as pd
import mlflow
from zenml import step
from zenml.steps import StepContext
from typing import Tuple
from typing_extensions import Annotated
from .Outlier_Detection import OutlierDetector
from .Missing_Imputer import MissingImputer
from src.utils.PipelineReporter import PipelineReporter
from src.utils.monitor import monitor
from configs import global_conf


DATASET_TARGET_COLUMN_NAME = "label"


@step
@monitor(name="missing_imputer_step", track_memory=True, track_input_size=True)
def missing_imputer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    missing_imputer = MissingImputer(
        max_missing_frac_drop=0.8,
        knn_neighbors=2,
        knn_mice_max_rows=10,
        knn_mice_max_columns=3,
        var_ratio_cutoff=0.5,
        cov_change_cutoff=0.2,
        rare_freq_cutoff=0.1,
        random_state=0,
        verbose=False,
    )

    train_imp = missing_imputer.fit_transform(train)
    test_imp = missing_imputer.transform(test)
    val_imp = missing_imputer.transform(val)

    _, missing_imputer_path = missing_imputer.report_missing_imputer(
        train, train_imp)

    # MLflow logging
    with mlflow.start_run(run_name="missing_imputer", nested=True):
        mlflow.log_param("knn_neighbors", 2)
        # mlflow.log_metric("num_missing", missing_imputer.outlier_flags_.sum())
        mlflow.log_artifact(missing_imputer_path)

    return train_imp, test_imp, val_imp


@step
@monitor(name="outlier_detector_step", track_memory=True, track_input_size=True)
def outlier_detector(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    detector = OutlierDetector(
        outlier_threshold=5,
        robust_covariance=True,
        cap_outliers=None,  # winsorize
        model_family="linear",
        random_state=0,
        verbose=False,
    )
    train_clean = detector.fit_transform(train)
    test_clean = detector.transform(test)
    val_clean = detector.transform(val)

    report, report_path = detector.report_outlier_detector(
        "outlier_detector", detector)

    # MLflow logging
    with mlflow.start_run(run_name="outlier_detector", nested=True):
        mlflow.log_param("outlier_threshold", 5)
        mlflow.log_metric("num_outliers", detector.outlier_flags_.sum())
        mlflow.log_artifact(report_path)

    return train_clean, test_clean, val_clean
