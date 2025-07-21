# 🐳 Dockerized MLD_LIFECYCLE Pipeline

This repo runs a **full local ML pipeline orchestration stack** inside containers:

---

## 🚀 Services Included

| Service   | Description                               | Port   |
| --------- | ----------------------------------------- | ------ |
| MLflow    | Experiment tracking + artifact store      | `5000` |
| ZenML UI  | Pipeline orchestration layer              | `9000` |
| FastAPI   | Model serving with REST API               | `8000` |
| Streamlit | Frontend visualization app                | `8501` |
| Optuna UI | Hyperparameter optimization dashboard     | `8080` |
| Evidently | Drift monitoring reports (runs on demand) | -      |

---

## 🧱 Volume Mounts

| Container | Volume Path                  | Description                   |
| --------- | ---------------------------- | ----------------------------- |
| mlflow    | `./mlruns:/app/mlruns`       | Tracks models/artifacts       |
| zenml     | `./.zen:/root/.zen`          | ZenML metadata & stack config |
| fastapi   | `./src:/app/src`             | Serving scripts + models      |
| streamlit | `./reports:/app/reports`     | HTML & JSON reports           |
| optuna    | `./optuna.db:/app/optuna.db` | Trial database for dashboard  |
| evidently | `./reports:/app/reports`     | Uses report folder artifacts  |

---

## 🛠️ Running the Stack

```bash
# 1. Build and launch all containers
docker-compose up --build

# 2. Access services
http://localhost:5000     # MLflow UI
http://localhost:9000     # ZenML Dashboard
http://localhost:8000     # FastAPI inference
http://localhost:8501     # Streamlit UI
http://localhost:8080     # Optuna Dashboard
```

---

## 🧪 Development Tips

- Modify source in `src/` — it's mounted live into all services.
- Reports auto-save to `reports/` and are viewable in Streamlit.
- Use `zenml stack list` inside container to check stack status.
- MLflow and Optuna data are persisted across runs using volumes.

---

## 📦 Additional Notes

- Poetry ensures consistent dependency management across services.
- Each service runs in isolation with mounted configs/artifacts.
- ZenML orchestrates runs, but MLflow/Optuna monitor internals.

---

## ✅ Use Cases Covered

- Train/test splits per type (linear/tree/KNN)
- Auto FE + imputation
- Multi-algorithm HPO (Optuna)
- Model selection + explainability (SHAP/LIME)
- Deployment (FastAPI)
- Reporting (Streamlit, Evidently)
- Drift-triggered retraining
- Fully reproducible setup

---

## 🧹 Cleanup

```bash
docker-compose down -v  # remove containers + volumes
```

---

## 🧰 Prerequisites

- Docker Engine v20+
- Docker Compose v2+
- Port access for 5000, 8000, 8501, 9000, 8080

---
