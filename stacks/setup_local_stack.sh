#!/usr/bin/env bash

set -Eeo pipefail

poetry run zenml integration install sklearn xgboost lightgbm mlflow great_expectations evidently whylogs  -y

poetry run zenml data-validator register ge_validator --flavor=great_expectations
poetry run zenml data-validator register evidently_validator --flavor=evidently

poetry run zenml secret create mlflow_secret \
    --username=aashay4123 \
    --password=hashtag
    
poetry run zenml experiment-tracker register local_mlflow_tracker  --flavor=mlflow   --tracking_username={{mlflow_secret.username}} --tracking_password={{mlflow_secret.password}} --tracking_uri=http://localhost:7000

poetry run zenml model-deployer register local_mlflow_deployer  --flavor=mlflow

poetry run zenml stack register local_stack \
    -a default \
    -o default \
    -e local_mlflow_tracker \
    -d local_mlflow_deployer \
    -dv evidently_validator \

poetry run zenml stack set local_stack
