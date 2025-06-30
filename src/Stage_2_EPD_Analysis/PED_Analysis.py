import os
import pandas as pd
import json
from configs import global_conf
import mlflow
from zenml import step
from zenml.steps import StepContext
from src.utils.monitor import monitor
from typing import Tuple, List
from src.Stage_2_EPD_Analysis.EDAnalyzer import EDAnalyzer
from src.Stage_2_EPD_Analysis.EPDA import UnifiedEPDA
from src.Stage_2_EPD_Analysis.PDAnalysis import ProbabilisticAnalysis


def serialize(obj):
    """Convert DataFrames/Series to JSON-serializable dicts/lists recursively."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="split")  # or "records"/"index"
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize(i) for i in obj]
    else:
        try:
            json.dumps(obj)  # test serializability
            return obj
        except (TypeError, OverflowError):
            return str(obj)  # fallback: string representation


@step
def EDAnalyze(df: pd.DataFrame, project: str = "Default") -> pd.DataFrame:

    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty. Cannot run ED Analysis.")
    context = StepContext.get_context()
    ed_analyze = EDAnalyzer(df)
    ed_analyze.run()

    # Save report/artifacts if available
    if hasattr(ed_analyze, "summary_report_path"):
        context.log_artifact("ed_report", ed_analyze.summary_report_path)

    with mlflow.start_run(run_name="ed_analyze", nested=True):
        mlflow.log_param("project", project)
        mlflow.log_metric("num_columns", len(df.columns))
        if hasattr(ed_analyze, "summary_report_path"):
            mlflow.log_artifact(ed_analyze.summary_report_path)

    return df


@step
def PDAnalyze(df: pd.DataFrame, project: str = "Default") -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty. Cannot run PD Analysis.")
    context = StepContext.get_context()

    pd_analyze = ProbabilisticAnalysis(df)
    pd_analyze.run()

    # Save report/artifacts if available
    if hasattr(pd_analyze, "summary_report_path"):
        context.log_artifact("pd_report", pd_analyze.summary_report_path)

    with mlflow.start_run(run_name="pd_analyze", nested=True):
        mlflow.log_param("project", project)
        mlflow.log_metric("num_columns", len(df.columns))
        if hasattr(pd_analyze, "summary_report_path"):
            mlflow.log_artifact(pd_analyze.summary_report_path)

    return df


@step
@monitor(name="unified_ped_analyze_step", track_memory=True, track_input_size=True)
def UnifiedPEDAnalyze(df: pd.DataFrame, project: str = "Default") -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty. Cannot run UnifiedPED Analysis.")

    unified_ped = UnifiedEPDA(df)
    report = unified_ped.run()
    eda_report_dir = global_conf.EPDA_REPORT_PATH
    os.makedirs(eda_report_dir, exist_ok=True)

    eda_path = os.path.join(eda_report_dir, "eda_summary.json")
    report_serializable = serialize(report)
    with open(eda_path, "w") as f:
        json.dump(report_serializable, f, indent=2)
    with mlflow.start_run(run_name="unified_ped_analyze", nested=True):
        mlflow.log_param("project", project)
        mlflow.log_artifact(eda_path)
        if hasattr(unified_ped, "generated_figures"):
            for fig_paths in unified_ped.generated_figures:
                mlflow.log_artifact(fig_paths)

    return df
