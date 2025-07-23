# 🧠 AutoML for Tabular Data — Fully Local, Modular & Robust

This repository implements a full-scale AutoML pipeline tailored for tabular datasets. It supports classification (binary & multiclass) and regression problems. The system is designed to be **modular**, **reproducible**, and **entirely local**, with **zero paid SaaS dependencies**.

---

## 🎯 What This Project Solves

- Automates the entire machine learning lifecycle — from ingestion to deployment.
- Optimizes models across multiple algorithm families (Linear, Tree-based, Distance-based).
- Tracks experiments, metrics, and artifacts with MLflow.
- Enables configurable pipelines for different types of models via ZenML.
- Supports parallel preprocessing paths for each model family.
- Provides flexible support for different file formats and ingestion sources.

---

## ✅ Current Supported Use Cases

- ✅ Binary Classification (e.g., spam detection, churn)
- ✅ Multi-Class Classification (e.g., Iris, MNIST-style)
- ✅ Regression (e.g., house prices, sales prediction)

Each use case supports:

- Preprocessing and feature engineering tailored to model type
- Hyperparameter tuning with Optuna
- Model comparison and selection
- Drift detection and retraining hooks
- Endpoints deployed via FastAPI
- Local UI via Streamlit

---

## 🚧 Not Yet Supported

- ❌ Time Series Forecasting
- ❌ Natural Language Processing
- ❌ Image or Vector-based Inputs
- ❌ Deep Learning Models
- ❌ Multi-Label Classification

These are listed in [`TODO.md`](./TODO.md) for future inclusion.

---

## ⚙️ How It Works

1. **Data Ingestion** – JSON, XLSX, CSV, ZIP, or API sources.
2. **Profiling & Validation** – Pandera/Great Expectations based checks.
3. **Train/Test Split** – Performed before imputation to prevent leakage.
4. **Parallel Preprocessing** – Separate paths for Linear, Tree, and Distance-based models.
5. **Model Baseline** – DummyClassifier is used to set base metrics.
6. **HPO + Comparison** – Optuna tunes all 10+ models per family; best one selected.
7. **Evaluation & Explainability** – Reports, SHAP, LIME.
8. **Deployment** – FastAPI + Docker + Streamlit UI.
9. **Monitoring** – Drift detection (Evidently/WhyLogs) with retrain triggers.

---

## 🧰 Key Tools & Technologies

| Function               | Tool/Library                |
| ---------------------- | --------------------------- |
| Pipeline Orchestration | ZenML                       |
| Experiment Tracking    | MLflow (local)              |
| Hyperparameter Tuning  | Optuna + DeepCave           |
| Data Validation        | Pandera, Great Expectations |
| Encoding & Scaling     | Custom Transformers         |
| Model Training         | Scikit-Learn, XGBoost       |
| Deployment             | FastAPI, Docker             |
| Monitoring             | Evidently, WhyLogs          |

---

## 🔗 Additional Docs

- [`ARCHITECTURE.md`](./docs/ARCHITECTURE.md) – Full pipeline structure & tooling map
- [`ML_steps.md`](./docs/ML_steps.md) – Detailed ML lifecycle steps
- [`GOALS.md`](./docs/GOALS.md) – Vision, constraints, and roadmap
- [`Prediction.md`](./docs/Prediction.md) – Prediction FastAPI details
- [`Report_Viz.md`](./docs/Report_Viz.md) – Reporting & visualization for entire pipeline
- [`DOCKER_README.md`](./docs/DOCKER_README.md) – Docker setup
- [`TODO.md`](./docs/TODO.md) – Pending feature backlog
- [`V1_Index.md`](./docs/V1/Stage_0_Index_V1.md) - V1 Readme Index
