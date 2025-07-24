import os
import pandas as pd
import json
from configs import global_conf
import mlflow
from zenml import step
from zenml.steps import StepContext
from src.utils.monitor import monitor
from typing import Tuple, List
from src.Stage_2_EPD_Analysis.EDAnalyzer import EDAnalyze
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
def FullPEDPipeline(df: pd.DataFrame, project: str = "Default") -> pd.DataFrame:
    if df is None or df.empty:
        raise ValueError("DataFrame is None or empty. Cannot run Full PED Pipeline.")

    # 1️⃣ Run ED Analysis
    print(f"[DEBUG] EDAnalyzer type: {EDAnalyze}")
    print("[DEBUG] EDAnalyzer:", EDAnalyze)

    ed_analyze = EDAnalyze(df)
    ed_analyze.run_all()
    # ed_report_path = os.path.join(
    #     global_conf.EPDA_REPORT_PATH, "ed_summary.json")
    # os.makedirs(global_conf.EPDA_REPORT_PATH, exist_ok=True)
    # with open(ed_report_path, "w") as f:
    #     json.dump(serialize(ed_report), f, indent=2)

    # 2️⃣ Run Probabilistic Analysis
    pd_analyze = ProbabilisticAnalysis(df)
    pd_report = pd_analyze.run()
    pd_report_path = os.path.join(global_conf.EPDA_REPORT_PATH, "pd_summary.json")
    with open(pd_report_path, "w") as f:
        json.dump(serialize(pd_report), f, indent=2)

    # 3️⃣ Run Unified EPDA Analysis
    unified_ped = UnifiedEPDA(df)
    unified_report = unified_ped.run()
    unified_report_path = os.path.join(
        global_conf.EPDA_REPORT_PATH, "unified_epda_summary.json"
    )
    with open(unified_report_path, "w") as f:
        json.dump(serialize(unified_report), f, indent=2)

    # MLflow tracking
    with mlflow.start_run(run_name="full_ped_pipeline", nested=True):
        mlflow.log_param("project", project)
        mlflow.log_metric("num_columns", len(df.columns))
        # mlflow.log_artifact(ed_report_path)
        mlflow.log_artifact(pd_report_path)
        mlflow.log_artifact(unified_report_path)

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
        mlflow.log_param("eda_report_dir", eda_report_dir)
        mlflow.log_artifact(eda_path)

    return df
