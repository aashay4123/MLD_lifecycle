import importlib
import os
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional, Union

import mlflow
import yaml
from mlflow import ActiveRun
from mlflow.artifacts import download_artifacts
from mlflow.entities import ViewType
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:7000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "ML_Local")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT)
_client = MlflowClient()

# ─────────────────────────────────────────────
# Run Management
# ─────────────────────────────────────────────


# @contextmanager
def start_run(
    run_name: Optional[str] = None,
    experiment_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    nested: bool = False,
) -> Generator[ActiveRun, None, None]:
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    # with mlflow.start_run(run_name=run_name, nested=nested, tags=tags) as active_run:
    #     yield active_run
    # __enter__ returns the ActiveRun
    return mlflow.start_run(run_name=run_name, nested=nested, tags=tags)


end_run = mlflow.end_run

# ─────────────────────────────────────────────
# Logging Parameters, Metrics & Artifacts
# ─────────────────────────────────────────────


def log_params(params: Dict[str, Any]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    for k, v in metrics.items():
        mlflow.log_metric(k, v, step=step)


def log_artifacts(path: str, artifact_path: Optional[str] = None) -> None:
    """
    Logs a file or all files under a directory.
    """
    if os.path.isdir(path):
        mlflow.log_artifacts(path, artifact_path=artifact_path)
    else:
        mlflow.log_artifact(path, artifact_path=artifact_path)


def download_artifact(artifact_uri: str, dst_path: Optional[str] = None) -> str:
    """
    Download an artifact from the run’s artifact store.
    (MLflow 3.x dropped the run_id parameter here.)
    """
    return download_artifacts(artifact_uri=artifact_uri, dst_path=dst_path)


def list_artifacts(
    path: Optional[str] = None, run_id: Optional[str] = None
) -> List[str]:
    return [f.path for f in mlflow.list_artifacts(path or "", run_id=run_id)]


# ─────────────────────────────────────────────
# Autologging
# ─────────────────────────────────────────────
_AUTOLOGGERS = {
    "sklearn": "mlflow.sklearn",
    "xgboost": "mlflow.xgboost",
    "lightgbm": "mlflow.lightgbm",
    "pytorch": "mlflow.pytorch",
    "tensorflow": "mlflow.tensorflow",
    "sparkml": "mlflow.spark",
}


def enable_autologging(framework: str) -> None:
    module_name = _AUTOLOGGERS.get(framework)
    if not module_name:
        raise ValueError(f"No autologging support for {framework!r}")
    mod = importlib.import_module(module_name)
    mod.autolog()


# ─────────────────────────────────────────────
# Model Logging & Loading
# ─────────────────────────────────────────────


def log_sklearn_model(
    model: Any,
    artifact_path: str,
    conda_env: Optional[Dict[str, Any]] = None,
    registered_model_name: Optional[str] = None,
) -> None:
    import mlflow.sklearn

    mlflow.sklearn.log_model(
        sk_model=model, artifact_path=artifact_path, conda_env=conda_env
    )
    if registered_model_name:
        uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model(uri, registered_model_name)


def load_sklearn_model(model_uri: str) -> Any:
    import mlflow.sklearn

    return mlflow.sklearn.load_model(model_uri)


def log_xgboost_model(
    model: Any,
    artifact_path: str,
    conda_env: Optional[Dict[str, Any]] = None,
    registered_model_name: Optional[str] = None,
) -> None:
    import mlflow.xgboost

    mlflow.xgboost.log_model(
        booster=model, artifact_path=artifact_path, conda_env=conda_env
    )
    if registered_model_name:
        uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model(uri, registered_model_name)


def load_xgboost_model(model_uri: str) -> Any:
    import mlflow.xgboost

    return mlflow.xgboost.load_model(model_uri)


def log_lightgbm_model(
    model: Any,
    artifact_path: str,
    conda_env: Optional[Dict[str, Any]] = None,
    registered_model_name: Optional[str] = None,
) -> None:
    import mlflow.lightgbm

    mlflow.lightgbm.log_model(
        booster=model, artifact_path=artifact_path, conda_env=conda_env
    )
    if registered_model_name:
        uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model(uri, registered_model_name)


def load_lightgbm_model(model_uri: str) -> Any:
    import mlflow.lightgbm

    return mlflow.lightgbm.load_model(model_uri)


def log_pyfunc_model(
    python_model: Any,
    artifact_path: str,
    conda_env: Optional[Dict[str, Any]] = None,
    registered_model_name: Optional[str] = None,
) -> None:
    mlflow.pyfunc.log_model(
        python_model=python_model, artifact_path=artifact_path, conda_env=conda_env
    )
    if registered_model_name:
        uri = f"runs:/{mlflow.active_run().info.run_id}/{artifact_path}"
        mlflow.register_model(uri, registered_model_name)


def load_pyfunc_model(model_uri: str) -> Any:
    return mlflow.pyfunc.load_model(model_uri)


# ─────────────────────────────────────────────
# Model Registry Operations
# ─────────────────────────────────────────────


def register_model(model_uri: str, name: str) -> ModelVersion:
    return mlflow.register_model(model_uri=model_uri, name=name)


def _set_model_stage(
    name: str, version: Union[int, str], stage: str, archive_existing: bool
) -> None:
    _client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage=stage,
        archive_existing_versions=archive_existing,
    )


def promote_model(
    name: str, version: Union[int, str], stage: str = "Production"
) -> None:
    _set_model_stage(name, version, stage=stage, archive_existing=True)


def archive_model(name: str, version: Union[int, str]) -> None:
    _set_model_stage(name, version, stage="Archived", archive_existing=False)


# ─────────────────────────────────────────────
# Serving Helper
# ─────────────────────────────────────────────


def serve_model(
    model_uri: str,
    host: str = "127.0.0.1",
    port: int = 1234,
    workers: int = 1,
    no_conda: bool = True,
) -> subprocess.Popen:
    cmd = [
        "mlflow",
        "models",
        "serve",
        "--model-uri",
        model_uri,
        "--host",
        host,
        "--port",
        str(port),
        "--workers",
        str(workers),
        *(["--no-conda"] if no_conda else []),
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ─────────────────────────────────────────────
# Environment & Reproducibility
# ─────────────────────────────────────────────


def log_conda_env(conda_env_path: str = "conda.yaml") -> None:
    if os.path.exists(conda_env_path):
        mlflow.log_artifact(conda_env_path, artifact_path="env")
    else:
        reqs = subprocess.check_output(["pip", "freeze"]).decode()
        with open("requirements.txt", "w") as f:
            f.write(reqs)
        mlflow.log_artifact("requirements.txt", artifact_path="env")


# ─────────────────────────────────────────────
# Quick Inspection
# ─────────────────────────────────────────────


def list_model_flavors(model_uri: str) -> Dict[str, Dict]:
    try:
        mlmodel_yml = download_artifacts(f"{model_uri}/MLmodel")
        with open(mlmodel_yml, "r") as f:
            spec = yaml.safe_load(f)
        return spec.get("flavors", {})
    except Exception as e:
        raise MlflowException(f"Failed to read flavors from {model_uri}: {e}") from e


def list_runs(
    experiment_name: Optional[str] = None,
    filter_string: str = "",
    order_by: Optional[List[str]] = None,
    max_results: int = 50,
):
    exp_ids = None
    if experiment_name:
        exp = _client.get_experiment_by_name(experiment_name)
        if exp is None:
            raise ValueError(f"Experiment {experiment_name!r} not found")
        exp_ids = [exp.experiment_id]
    return _client.search_runs(
        experiment_ids=exp_ids,
        filter_string=filter_string,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=max_results,
        order_by=order_by or ["attributes.start_time desc"],
    )


__all__ = [
    "start_run",
    "end_run",
    "log_params",
    "log_metrics",
    "log_artifacts",
    "download_artifact",
    "list_artifacts",
    "enable_autologging",
    "log_sklearn_model",
    "load_sklearn_model",
    "log_xgboost_model",
    "load_xgboost_model",
    "log_lightgbm_model",
    "load_lightgbm_model",
    "log_pyfunc_model",
    "load_pyfunc_model",
    "register_model",
    "promote_model",
    "archive_model",
    "serve_model",
    "log_conda_env",
    "list_model_flavors",
    "list_runs",
]

# ───────────────────────────────────────────── Example Usage ─────────────────────────────────────────────

# from mlflow_utils import (
#     start_run, end_run,
#     log_params, log_metrics,
#     log_artifacts, download_artifact,
#     enable_autologging,
#     log_sklearn_model, load_sklearn_model,
#     register_model, promote_model,
#     log_conda_env,
#     list_model_flavors, list_runs,
#     serve_model,
# )
# from sklearn.linear_model import ElasticNet
# from sklearn.metrics import mean_squared_error, r2_score
# import os
# import time

# # 1) Optionally turn on autologging for sklearn
# enable_autologging("sklearn")

# # 2) Start a run (with an explicit name)
# with start_run(run_name="elasticnet_v1") as run:
#     # 2a) Log hyperparameters
#     alpha = 0.1
#     l1_ratio = 0.5
#     log_params({"alpha": alpha, "l1_ratio": l1_ratio})

#     # 2b) Train your model
#     model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
#     model.fit(X_train, y_train)  # assume X_train, y_train are defined

#     # 2c) Evaluate and log metrics
#     preds = model.predict(X_test)
#     rmse = mean_squared_error(y_test, preds, squared=False)
#     r2   = r2_score(y_test, preds)
#     log_metrics({"rmse": rmse, "r2": r2})

#     # 2d) Log your environment for reproducibility
#     log_conda_env("conda.yaml")

#     # 2e) Log any extra artifacts (e.g. figures, reports)
#     log_artifacts("reports/eda.html", artifact_path="reports")

#     # 2f) Log & register the model in the MLflow Model Registry
#     artifact_subdir = "model"
#     log_sklearn_model(
#         model,
#         artifact_path=artifact_subdir,
#         conda_env=None,
#         registered_model_name="ElasticNetENet"
#     )
#     mv = register_model(
#         model_uri=f"runs:/{run.info.run_id}/{artifact_subdir}",
#         name="ElasticNetENet"
#     )

#     # 2g) Promote the newly registered version to “Staging”
#     promote_model("ElasticNetENet", mv.version, stage="Staging")

# # 3) After the run context, you can list recent runs
# recent = list_runs(experiment_name=os.getenv("MLFLOW_EXPERIMENT"), max_results=5)
# for info in recent:
#     print(f"Run {info.info.run_id}: rmse={info.data.metrics.get('rmse')}")

# # 4) Inspect available flavors for the Production model
# flavors = list_model_flavors("models:/ElasticNetENet/Production")
# print("Available flavors:", flavors)

# # 5) Load the model and serve it locally
# loaded = load_sklearn_model("models:/ElasticNetENet/Production")
# print("Loaded model prediction:", loaded.predict(X_test[:3]))

# serve_proc = serve_model(
#     model_uri="models:/ElasticNetENet/Production",
#     host="0.0.0.0",
#     port=8080,
#     workers=2,
#     no_conda=True
# )
# print("Serving on port 8080 (pid:", serve_proc.pid, ")")

# # 6) Download an artifact from a past run
# local_path = download_artifact(
#     artifact_uri=f"runs:/{recent[0].info.run_id}/reports/eda.html",
#     dst_path="./downloaded"
# )
# print("Downloaded EDA report to:", local_path)

# # 7) Clean up
# time.sleep(2)
# serve_proc.terminate()
# end_run()
