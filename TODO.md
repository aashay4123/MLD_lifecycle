# 🧾 TODO — Backlog & Missing Features

---

## 🚧 Functional Gaps

- [ ] Add **Time Series** support (Prophet, AutoTS, sktime)
- [ ] Add **NLP Pipeline** (tokenization, embeddings, classification)
- [ ] Add support for **Multi-Label Classification**
- [ ] Add support for **Online Learning** (e.g., river)
- [ ] Add support for **Distributed HPO** with Dask/Optuna
- [ ] Add support for **Image/Vision inputs**
- [ ] Add **Fairness Audits** (Fairlearn, Aequitas)
- [ ] Add support for **Multi-modal fusion** (tabular + text)

---

## 🛠️ Technical Enhancements

- [ ] Full **drift explainability** after prediction
- [ ] Schema drift detection in transform-time
- [ ] Auto-persistence of all fitted transformers
- [ ] Add caching for repeated transformer runs
- [ ] Complete async support for disk operations
- [ ] ZenML pipeline cache / step outputs

---

## 📊 Reporting Enhancements

- [ ] HTML Summary Report after every pipeline
- [ ] Add SHAP force plot support
- [ ] Integrate DeepCave optuna metrics in reports
- [ ] Memory usage and size footprint for each stage
- [ ] Summary drift report for production mode

---

## 🧪 Testing & Reproducibility

- [ ] Auto-checksum for every config and pipeline input
- [ ] Random seed verification for all stochastic steps
- [ ] Validate pipeline reproducibility on CI
- [ ] Add integration tests for every ZenML pipeline

---

## ✅ Completed

- [x] Baseline model via DummyClassifier
- [x] Multi-path preprocessing
- [x] HPO via Optuna + logging
- [x] Full MLflow + FastAPI integration
- [x] Pipeline modularization by type
- [x] Streamlit UI & interactive deployment
