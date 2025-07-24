# deploy_model.py

import mlflow
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import step


@step
def deploy_model() -> MLFlowDeploymentService:
    deployer = MLFlowModelDeployer.get_active_model_deployer()
    model_name = "final_ensemble_model"
    active_run = mlflow.active_run()
    model_uri = f"runs:/{active_run.info.run_id}/{model_name}"

    service = deployer.deploy_model(
        model_uri=model_uri, model_name=model_name, replace=True, timeout=60
    )
    return service
