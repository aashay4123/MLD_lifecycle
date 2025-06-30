from src.data_analysis.advanced_drift_monitor import DriftMonitor
from src.data_analysis.Probabilistic.probabilistic_analysis import ProbabilisticAnalysis
from pathlib import Path
from zenml.pipelines import pipeline
from zenml.steps import step, Output
import pandas as pd
import os
import json
import logging
import mlflow

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@step
def data_drift_check(
    train: pd.DataFrame, live: pd.DataFrame
) -> Output(drift_score=float, drift_report=dict):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train, current_data=live)

    results = report.as_dict()
    drift_score = results["metrics"][0]["result"]["dataset_drift_share"]
    logger.info(f"[Evidently] Drift score: {drift_score:.2f}")

    with open("artifacts/drift/evidently_data_drift.json", "w") as f:
        json.dump(results, f, indent=2)

    mlflow.log_artifact("artifacts/drift/evidently_data_drift.json")
    mlflow.log_metric("evidently_drift_score", drift_score)

    return drift_score, results


@step
def target_drift_check(
    train: pd.DataFrame, live: pd.DataFrame
) -> Output(target_drift_score=float):
    report = Report(metrics=[TargetDriftPreset()])
    report.run(reference_data=train, current_data=live)

    results = report.as_dict()
    drift_score = results["metrics"][0]["result"]["dataset_drift_share"]
    logger.info(f"[Evidently] Target Drift score: {drift_score:.2f}")

    with open("artifacts/drift/evidently_target_drift.json", "w") as f:
        json.dump(results, f, indent=2)

    mlflow.log_artifact("artifacts/drift/evidently_target_drift.json")
    mlflow.log_metric("evidently_target_drift", drift_score)

    return drift_score


@step
def probabilistic_monitor_step(data: pd.DataFrame) -> Output(alerts=list):
    pa = ProbabilisticAnalysis(data)
    alerts = []

    dist = pa.fit_all_distributions()
    for f, r in dist.items():
        if r["ks_stat"] > 0.2:
            alerts.append(f"KS drift: {f}")

    entropy = pa.shannon_entropy()
    for f, s in entropy.items():
        if s < 0.5:
            alerts.append(f"Entropy collapse: {f}")

    mi = pa.mutual_info_scores("target")
    for f, s in mi.items():
        if s < 0.01:
            alerts.append(f"Low MI: {f}")

    try:
        pa.copula_modeling()
    except Exception as e:
        alerts.append("Copula failed")

    for f in data.select_dtypes("number").columns:
        low, high = pa.predictive_intervals(f)
        if high - low < 0.1:
            alerts.append(f"Predictive collapse: {f}")

    imp = pa.feature_importance("target")
    for f, s in imp.items():
        if s < 0.005:
            alerts.append(f"Low feature importance: {f}")

    return alerts


@step
def data_drift_step(train: pd.DataFrame, test: pd.DataFrame) -> Output(score=float):
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=train, current_data=test)
    result = report.as_dict()

    score = result["metrics"][0]["result"]["dataset_drift_share"]
    Path("artifacts/drift").mkdir(parents=True, exist_ok=True)
    with open("artifacts/drift/data_drift.json", "w") as f:
        json.dump(result, f, indent=2)

    mlflow.log_artifact("artifacts/drift/data_drift.json")
    mlflow.log_metric("data_drift_score", score)

    return score


@step
def target_drift_step(train: pd.DataFrame, test: pd.DataFrame) -> Output(score=float):
    report = Report(metrics=[TargetDriftPreset()])
    report.run(reference_data=train, current_data=test)
    result = report.as_dict()

    score = result["metrics"][0]["result"]["dataset_drift_share"]
    with open("artifacts/drift/target_drift.json", "w") as f:
        json.dump(result, f, indent=2)

    mlflow.log_artifact("artifacts/drift/target_drift.json")
    mlflow.log_metric("target_drift_score", score)

    return score


@step
def advanced_drift_step(train: pd.DataFrame, test: pd.DataFrame) -> Output(alerts=list):
    cat = train.select_dtypes("object").columns.tolist()
    num = train.select_dtypes("number").columns.tolist()

    monitor = DriftMonitor(numeric_features=num, categorical_features=cat)
    monitor.fit_reference(ref_df=train, ref_target=train["target"])

    report, alerts = monitor.detect(test, test["target"])

    with open("artifacts/drift/advanced_drift.json", "w") as f:
        json.dump(report, f, indent=2)

    mlflow.log_artifact("artifacts/drift/advanced_drift.json")
    return alerts


@pipeline(enable_cache=False)
def drift_monitoring_pipeline(
    ingest_data,
    inspect_data,
    split_data,
    data_drift_step,
    target_drift_step,
    probabilistic_monitor_step,
    advanced_drift_step,
):
    df = ingest_data()
    df = inspect_data(df)
    train, val, test = split_data(df)

    data_drift_step(train, test)
    target_drift_step(train, test)
    probabilistic_monitor_step(test)
    advanced_drift_step(train, test)
