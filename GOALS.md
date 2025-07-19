# 🥅 MLD_LIFECYCLE: Project Goals & Design Principles

---

## 🎯 Core Mission

Build a **modular, local-first AutoML framework** for tabular datasets with zero SaaS dependencies and full reproducibility.

---

## 🧱 Design Philosophy

| Principle             | Description |
|-----------------------|-------------|
| **Modularity**        | Each pipeline stage is a standalone class integrated with ZenML |
| **Reproducibility**   | Every transformation is tracked with fixed seeds and MLflow |
| **Parallelism**       | Preprocessing and modeling pipelines are branched by model family |
| **Transparency**      | Explainability and metadata logging included |
| **Flexibility**       | Supports any tabular dataset and auto-adapts to its type |

---

## ✅ Supported Model Types

- Linear models (Ridge, Lasso, Logistic Regression)
- Tree models (Random Forest, XGBoost, LightGBM)
- Distance-based models (KNN, RadiusNeighbors)
- DummyClassifier for baselining

---

## 🚧 Future Goals (Planned)

- NLP support via spaCy / HuggingFace pipelines
- Time Series support (ARIMA, Prophet, LSTM hybrids)
- Vector embeddings (word2vec, image embeddings)
- Multi-label classification
- Online learning & streaming pipelines
- Out-of-core training with Dask/Modin
- Model fairness audits and bias detection
