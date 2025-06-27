import pandas as pd
from zenml import step
from .DataCollector import DataCollector
from .DataHealthCheck import DataHealthCheck
from src.utils.PipelineReporter import PipelineReporter


DATASET_TARGET_COLUMN_NAME = "label"


@step
def dataCheck(df: pd.DataFrame, reporter: PipelineReporter = None) -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError(
            "DataFrame is None or empty. Cannot perform health check.")
    if DATASET_TARGET_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Target column '{DATASET_TARGET_COLUMN_NAME}' not found in DataFrame.")

    health_check = DataHealthCheck(df)
    health_check.run_all_checks()
    # 3. Register the health report and charts
    reporter.register(name="DataHealthCheck", report=health_check.get_results(
    ), charts=health_check.get_chart_paths())

    return df


@step
def dataLoader(file: str = None, project: str = "Default") -> pd.DataFrame:
    dataCollector = DataCollector(suite_name=project)
    df = dataCollector.read_file(file)
    print(f"Data loaded from {file} with shape {df.shape}.")
    if df is None:
        raise ValueError(
            "DataFrame is None. Please check the file path or data source.")
    if df.empty:
        raise ValueError(
            "DataFrame is empty. Please check the file path or data source.")
    if DATASET_TARGET_COLUMN_NAME not in df.columns:
        raise ValueError(
            f"Target column '{DATASET_TARGET_COLUMN_NAME}' not found in DataFrame.")
    print(f"Data loaded successfully from {file} with shape {df.shape}.")
    return df
