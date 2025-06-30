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

    # Generate report
    reporter = PipelineReporter()
    reporter.register("missing_imputer", missing_imputer)
    report = reporter.generate_report("missing_imputer_report")

    preprocessor_report_dir = global_conf.PREPROCESSOR_REPORT_PATH
    os.makedirs(preprocessor_report_dir, exist_ok=True)

    missing_imputer_path = os.path.join(
        preprocessor_report_dir, "missing_imputer_report.json"
    )

    with open(missing_imputer_path, "w") as f:
        json.dump(report, f, indent=2)

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
        outlier_threshold=3,
        robust_covariance=True,
        cap_outliers=None,  # winsorize
        model_family="linear",
        random_state=0,
        verbose=False,
    )
    train_clean = detector.fit_transform(train)
    test_clean = detector.transform(test)
    val_clean = detector.transform(val)

    # Generate report
    reporter = PipelineReporter()
    reporter.register("outlier_detector", detector)
    report = reporter.generate_report("outlier_detector_report")

    preprocessor_report_dir = global_conf.PREPROCESSOR_REPORT_PATH
    os.makedirs(preprocessor_report_dir, exist_ok=True)

    outlier_detector_path = os.path.join(
        preprocessor_report_dir, "outlier_detector_report.json"
    )

    with open(outlier_detector_path, "w") as f:
        json.dump(report, f, indent=2)

    # MLflow logging
    with mlflow.start_run(run_name="outlier_detector", nested=True):
        mlflow.log_param("outlier_threshold", 3)
        mlflow.log_metric("num_outliers", detector.outlier_flags_.sum())
        mlflow.log_artifact(outlier_detector_path)

    return train_clean, test_clean, val_clean


@step
def fit_data_preprocessor_step(context: StepContext, train_df: pd.DataFrame) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Fit on training data: detects outliers, imputes, and saves both models."""
    outlier_model = OutlierDetector(verbose=True)
    cleaned_df = outlier_model.fit_transform(train_df)
    outlier_model_path = f"{context.artifact_uri}/outlier_model_state.pkl"
    outlier_model.save_state(outlier_model_path)
    context.log_artifact("outlier_model", outlier_model_path)

    imputer = MissingImputer()
    imputed_df = imputer.fit_transform(cleaned_df)
    imputer_model_path = f"{context.artifact_uri}/missing_model_state.pkl"
    imputer.save_state(imputer_model_path)
    context.log_artifact("missing_model", imputer_model_path)

    return imputed_df, Path(outlier_model_path), Path(imputer_model_path)


@step
def transform_data_preprocessor_step(
    input_df: pd.DataFrame, outlier_model_path: Path, missing_model_path: Path
) -> pd.DataFrame:
    """Apply saved outlier + imputer models to validation/test data."""
    outlier_model = OutlierDetector()
    outlier_model.load_state(str(outlier_model_path))
    df_cleaned = outlier_model.transform(input_df)

    imputer = MissingImputer()
    imputer.load_state(str(missing_model_path))
    df_final = imputer.transform(df_cleaned)

    return df_final
