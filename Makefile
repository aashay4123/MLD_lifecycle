all: 
	make env 
	make install 
	make test

env:
	poetry env list || poetry env use python3.11

install:
	poetry install
	poetry run zenml integration install s3 sklearn mlflow -y

devup:
	poetry run zenml up
	poetry run mlflow server --backend-store-uri sqlite:///.zen/mlflow.db --default-artifact-root ./.zen/mlruns --host 127.0.0.1  --port 7000

train:
	poetry run train

test:
	poetry run test

tune:
	poetry run tune

fmt:
	poetry run black .

lint:
	poetry run flake8 src/
