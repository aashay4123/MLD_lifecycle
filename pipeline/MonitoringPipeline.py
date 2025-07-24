from typing import List

from deploy_model import deploy_model
from drift_monitoring_pipeline import (
    drift_monitoring_pipeline,
)
from evaluate_and_register import evaluate_and_register
from optuna_hpo_with_ensemble_pipeline import optuna_hpo_with_ensemble_pipeline
from test_prediction_pipeline import test_prediction_pipeline

# === Step imports ===
from training_pipeline import training_pipeline
from zenml.pipelines import pipeline
from zenml.steps import step


@step
def check_drift_alerts(
    data_alerts: List[str],
    target_drift_score: float,
    performance_alerts: List[str],
    entropy_warnings: List[str],
) -> bool:
    print("🔍 Drift Alerts:", data_alerts)
    print("🎯 Target Drift Score:", target_drift_score)
    print("📉 Performance Alerts:", performance_alerts)
    print("🧪 Entropy Warnings:", entropy_warnings)

    # Custom logic to trigger retraining
    if (
        data_alerts
        or target_drift_score > 0.3
        or performance_alerts
        or entropy_warnings
    ):
        print("⚠️ Drift/Anomaly Detected: Retraining triggered.")
        return True
    print("✅ No major drift detected. Skipping retraining.")
    return False


@pipeline(enable_cache=False)
def MonitoringPipeline():
    # === Step 1: Base training pipeline ===
    training_pipeline()

    # === Step 2: Drift Monitoring + Validation ===
    monitoring_outputs = drift_monitoring_pipeline()

    # === Step 3: Conditional Check for Retraining ===
    retrain_needed = check_drift_alerts(
        data_alerts=monitoring_outputs["advanced_drift_step"],
        target_drift_score=monitoring_outputs["target_drift_step"],
        performance_alerts=monitoring_outputs["probabilistic_monitor_step"],
        entropy_warnings=monitoring_outputs["entropy_check_step"],
    )

    # === Step 4: Retrain with HPO + Ensemble Selection ===
    if retrain_needed:
        optuna_hpo_with_ensemble_pipeline()

    # === Step 5: Evaluate + Register model in MLflow ===
    evaluate_and_register()

    # === Step 6: Deploy model with ZenML ===
    deploy_model()

    # === Step 7: Run prediction tests if needed ===
    test_prediction_pipeline()
