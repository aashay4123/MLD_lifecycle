import joblib
import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss
from zenml.steps import Output, step


@step
def evaluate_and_register(model, val_metrics: dict) -> Output(metrics=dict):
    test = pd.read_parquet("artifacts/final/production/test.parquet")
    X_test = test.drop(columns=["target"])
    y_test = test["target"]

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    test_accuracy = accuracy_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred, average="weighted")
    test_logloss = log_loss(y_test, y_prob) if y_prob is not None else None

    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_f1", test_f1)
    if test_logloss is not None:
        mlflow.log_metric("test_log_loss", test_logloss)

    current_metrics = {
        "val_accuracy": val_metrics.get("accuracy"),
        "test_accuracy": test_accuracy,
        "test_f1": test_f1,
        "test_log_loss": test_logloss,
    }

    model_name = "final_ensemble_model"
    client = mlflow.tracking.MlflowClient()

    # === Compare with baseline run ===
    try:
        baseline_runs = mlflow.search_runs(filter_string='tags.type = "baseline"')
        if not baseline_runs.empty:
            best_baseline_acc = baseline_runs.sort_values(
                "metrics.test_accuracy", ascending=False
            ).iloc[0]["metrics.test_accuracy"]
            if test_accuracy > best_baseline_acc:
                print(
                    f"✅ New model beats baseline: {test_accuracy:.3f} vs {best_baseline_acc:.3f}"
                )
            else:
                print(
                    f"⚠️ Baseline better: {best_baseline_acc:.3f} vs {test_accuracy:.3f}"
                )
    except Exception as e:
        print("⚠️ Baseline comparison skipped:", e)

    # === Compare with last registered model ===
    try:
        latest_versions = client.get_latest_versions(
            model_name, stages=["Production", "Staging"]
        )
        best_prev_acc = 0.0
        for v in latest_versions:
            run = mlflow.get_run(v.run_id)
            acc = float(run.data.metrics.get("val_accuracy", 0))
            best_prev_acc = max(best_prev_acc, acc)

        new_val_acc = val_metrics.get("accuracy", 0)
        if new_val_acc > best_prev_acc:
            # Register and transition
            mlflow.sklearn.log_model(
                model, artifact_path=model_name, registered_model_name=model_name
            )
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            result = mlflow.register_model(model_uri, model_name)
            client.transition_model_version_stage(
                model_name,
                result.version,
                stage="Staging",
                archive_existing_versions=True,
            )
            print(f"✅ Model registered to Staging (v{result.version}) [val_acc ↑]")
        else:
            print(
                f"ℹ️ Model NOT registered (val_acc: {new_val_acc:.3f} <= {best_prev_acc:.3f})"
            )
    except Exception as e:
        print("⚠️ Registration failed:", e)

    return current_metrics
