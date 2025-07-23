# 📊 Report Visualization Overview

This document outlines the structure and purpose of all reports generated during the ML/DL lifecycle pipeline. All visual outputs are integrated and accessible through a central **Streamlit dashboard** served at `http://localhost:8501`.

---

## 🔗 Dashboard Summary

| Tool                 | Port   | Purpose                           |
| -------------------- | ------ | --------------------------------- |
| **Streamlit**        | `8501` | Unified dashboard for all reports |
| **MLflow UI**        | `7000` | Experiment tracking               |
| **Optuna Dashboard** | `8080` | Hyperparameter optimization       |
| **ZenML Dashboard**  | `9000` | Pipeline lineage & orchestration  |

> ℹ️ FastAPI is excluded from visualization. It's only used for inference endpoints in deployed apps.

---

## 🗂️ Streamlit Page Routing

The Streamlit app is **modularized into multiple pages**, each dedicated to a specific type of report or visualization. Below is a breakdown of the pages and the sources they consume:

---

### 1️⃣ EDA Overview (`EDA.py`)

| Element                      | File(s) / Folder(s)                                 |
| ---------------------------- | --------------------------------------------------- |
| Basic statistics table       | `basic_descriptive_stats.csv`                       |
| Hopkins cluster tendency     | `hopkins_score.txt`                                 |
| Entropy and normality scores | `entropy_scores.csv`, `normality_tests.csv`         |
| Copula visualizations        | `copula_sample.csv`, `distribution_fit_quality.csv` |
| ACF/PACF charts              | `/epda/acf_pacf/` (images)                          |
| Histogram plots              | `/epda/hist/` (images)                              |
| PIT & QQ plots               | `/epda/pit/`, `/epda/qq/`                           |

---

### 2️⃣ Data Health (`DataHealth.py`)

| Element                     | File(s)                                     |
| --------------------------- | ------------------------------------------- |
| Cardinality and correlation | `cardinality.png`, `correlation_matrix.png` |
| Imbalance & skewness        | `imbalance.png`, `skewness.png`             |
| Missingness heatmap         | `missingness.png`                           |
| Leakage map                 | `leakage_report.json`                       |
| Data type summary           | `dtypes.json`, `report.json`                |
| Overall health report       | `data_health_report.json`, `report.md`      |

---

### 3️⃣ Scaler Impact Analysis (`ScalerEffect.py`)

| Element                    | File(s)                            |
| -------------------------- | ---------------------------------- |
| Before/After scaling plots | `/scaler_visuals/**.png`           |
| Grouped by dataset/domain  | Folder-nested (e.g. `wine_od280/`) |

---

### 4️⃣ Model Reports (`ModelReport.py`)

| Element                | File(s) / Source              |
| ---------------------- | ----------------------------- |
| Baseline metrics       | `/baseline_report/`           |
| Trained model metrics  | `/Models/`, MLflow run folder |
| Downloadable artifacts | `.pkl`, `.json`, `.onnx`      |

---

### 5️⃣ Hyperparameter Tuning (`OptunaTuning.py`)

| Element                | Source                                  |
| ---------------------- | --------------------------------------- |
| Trial history & charts | `/optuna/`, Optuna log dumps            |
| Best parameters        | Optuna Dashboard (iframe)               |
| External link          | [localhost:8080](http://localhost:8080) |

---

### 6️⃣ MLflow Experiments (`MLflowUI.py`)

| Element          | Source                                  |
| ---------------- | --------------------------------------- |
| Run history      | MLflow tracking logs                    |
| Artifact preview | Linked via API or UI                    |
| External link    | [localhost:7000](http://localhost:7000) |

---

### 7️⃣ Pipeline Lineage (`ZenMLRunLineage.py`)

| Element             | Source                                  |
| ------------------- | --------------------------------------- |
| ZenML Pipelines     | ZenML orchestrated runs                 |
| Step dependency DAG | Auto-generated                          |
| External link       | [localhost:9000](http://localhost:9000) |

---

## 📁 Reports Folder Overview

Directory structure used by Streamlit app:

```

reports/
├── baseline_report/
├── epda/
│ ├── acf_pacf/
│ ├── hist/
│ ├── pit/
│ ├── qq/
│ ├── \*.csv, \*.json
├── health_report/
│ ├── \*.png
│ ├── \*.json, \*.md
├── scaler_visuals/
│ ├── \<dataset_name>/
│ └── \*\_before_after.png
├── optuna/
├── preprocessor_report/
├── Models/
├── logs/

```

---

## 🚀 How to Launch

```bash
# From root
make dashboard  # or:
streamlit run dashboard/Home.py
```

> Ensure you have `streamlit`, `pandas`, `Pillow`, and other visual dependencies installed.

---

## 🧩 Additional Ideas

- Embed `Evidently` report JSON as visual summary.
- Add warning indicators on each tab if any issue detected (e.g. missingness > 30%).
- Add downloadable ZIP button for all reports.

---
