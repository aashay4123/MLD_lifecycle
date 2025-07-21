# 📍 Dashboard and Endpoint URL Index

This file lists all local URLs, ports, and access points for various components of the MLD_LIFECYCLE system architecture. It is meant as a single reference index during development, debugging, and demonstration.

---

## ⚙️ Pipeline & Orchestration

| Component        | Tool       | Description                                     | Default URL / Path                  |
|------------------|------------|-------------------------------------------------|-------------------------------------|
| ZenML Dashboard  | ZenML      | View pipeline DAGs, stack configs, step outputs | http://localhost:8237              |
| ZenML CLI        | ZenML      | CLI to trigger, inspect, and manage pipelines   | `zenml pipeline run` or `zenml up` |

---

## 🧪 Experiment Tracking

| Component     | Tool    | Description                                     | Default URL / Path                  |
|---------------|---------|-------------------------------------------------|-------------------------------------|
| MLflow UI     | MLflow  | View all experiments, models, parameters, metrics | http://localhost:5000              |
| MLflow Artifacts | MLflow | Checkpoints, pickles, reports for each run     | `mlruns/` (local path)              |
| MLflow Registry | MLflow | Versioned models, staging/production tags      | http://localhost:5000/#/models     |

---

## 🎯 Hyperparameter Optimization

| Component         | Tool          | Description                                   | Default URL / Path                    |
|-------------------|---------------|-----------------------------------------------|---------------------------------------|
| Optuna Dashboard  | DeepCave      | Interactive UI for Optuna trials              | http://localhost:8501 (Streamlit app)|
| Optuna DB         | SQLite        | Trial logs and optimization history           | `optuna_trials.db`                    |

---

## 📊 Evaluation & Explainability

| Component          | Tool          | Description                                   | Default URL / Path                    |
|--------------------|---------------|-----------------------------------------------|---------------------------------------|
| SHAP Plots         | SHAP          | Global + Local explanation artifacts          | Stored in `reports/explainability/`   |
| LIME HTMLs         | LIME          | HTML-based local explanations                 | Stored in `reports/explainability/`   |
| Evaluation Metrics | Custom / Sklearn | ROC, PR, Confusion Matrix, MAE, etc        | Logged in MLflow                      |

---

## 🚀 Deployment Interface

| Component         | Tool          | Description                                 | Default URL / Path                     |
|-------------------|---------------|---------------------------------------------|----------------------------------------|
| FastAPI Endpoint  | FastAPI       | REST endpoint for prediction                | http://localhost:8000/docs             |
| Swagger UI        | FastAPI       | API schema, sample inputs, test console     | http://localhost:8000/docs             |
| Streamlit App     | Streamlit     | UI interface for interactive predictions    | http://localhost:8502                  |

---

## 📈 Monitoring & Drift Detection

| Component            | Tool         | Description                                  | Default URL / Path                     |
|----------------------|--------------|----------------------------------------------|----------------------------------------|
| Evidently Report     | Evidently    | HTML report on drift, stability, feature stats | Stored in `reports/drift/*.html`     |
| Drift Logs           | Custom       | JSON logs of feature distribution change     | `logs/drift_monitoring.json`           |

---

## 🔁 Retraining Logic

| Component          | Tool       | Description                                 | Default URL / Path                     |
|--------------------|------------|---------------------------------------------|----------------------------------------|
| Retrain Trigger    | Custom/CLI | Checks for drift, triggers retrain          | CLI: `make retrain`                    |
| Version Compare    | Custom     | Evaluation report comparing old vs new model | Stored in `reports/model_compare/`     |

---

## 🧰 Diagnostics & Validation

| Component          | Tool            | Description                                 | Default URL / Path                     |
|--------------------|-----------------|---------------------------------------------|----------------------------------------|
| Data Health Check  | Pandera / GE    | Schema and type validation report           | `reports/validation/schema_check.html` |
| EDA Summary        | Custom / ydata  | HTML profile of distributions & statistics  | `reports/eda/summary.html`             |
| Imputation Report  | Custom          | JSON/HTML report of imputation logic        | `reports/imputation/`                  |
| Preprocessor Logs  | Custom          | JSON summaries of transforms per stage      | `reports/preprocessing/*.json`         |

---

## 📦 Packaging & Infra

| Component         | Tool           | Description                                   | Default URL / Path                     |
|-------------------|----------------|-----------------------------------------------|----------------------------------------|
| Docker Compose    | Docker         | Orchestration of services (MLflow, Streamlit) | `docker-compose.yaml`                  |
| DVC Artifacts     | DVC (Local)    | Versioned data files and models               | `.dvc/`, `data/`, `artifacts/`         |
| Requirements File | Poetry/Pip     | Exact Python environment                     | `pyproject.toml`, `poetry.lock`        |

---

## 🧾 Version & Audit Trail

| Component         | Tool       | Description                                 | Default URL / Path                     |
|-------------------|------------|---------------------------------------------|----------------------------------------|
| GitHub Repo       | Git        | Code versioning, commits, branches           | https://github.com/<user>/MLD_lifecycle |
| Commit Info       | Git        | Used for SHA-tagging experiments             | Via `git rev-parse HEAD`               |
| Audit Metadata    | Custom     | Stored alongside reports in YAML/JSON       | `reports/*/metadata.json`              |

---

🧠 **Tips**:
- Most dashboards default to `localhost`; use `0.0.0.0` if running in Docker.
- Use `make devup` or `zenml up` to launch all services together.
- To expose URLs outside local network, consider tools like [ngrok](https://ngrok.com/).

