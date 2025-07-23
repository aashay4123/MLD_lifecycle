## ✅ Step-by-Step Integration of DVC with ZenML + MLflow

### 1. 📦 Initialize DVC in your existing repo

```bash
cd your-ml-repo
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

---

### 2. 🗂️ Choose what to track with DVC (exclude from Git)

Example: You already ignore `Data/`, `reports/`, `mlruns/`, etc.

Now explicitly track these with DVC:

```bash
dvc add Data/raw/
dvc add reports/
dvc add artifacts/final_data/
```

This will generate `.dvc` files (`Data/raw.dvc`, etc.). Add them to Git:

```bash
git add Data/raw.dvc reports.dvc artifacts/final_data.dvc
git commit -m "Track raw data, reports, and final data with DVC"
```

---

### 3. 🧩 Add ZenML steps as `dvc.yaml` stages (optional but powerful)

You can optionally make each **ZenML step** a DVC stage using `dvc stage add`. Example:

```bash
dvc stage add -n ingest_data \
    -d src/Stage_1_Ingestion/data_injection/DataCollector.py \
    -d data/input.csv \
    -o Data/raw/raw.csv \
    python src/Stage_1_Ingestion/data_injection/DataCollector.py
```

You can repeat this for each key stage if you want DVC to orchestrate it too — but this is optional if ZenML does that for you.

---

### 4. 🧾 Use `params.yaml` to decouple configs

You can define hyperparameters, paths, flags for DVC steps in a single config file:

```yaml
# params.yaml
preprocessing:
  imputation_strategy:
    Age: median
    Gender: mode
  outlier_detection:
    method: isolation_forest
    contamination: 0.05
  encoding:
    strategy: auto
modeling:
  model_type: xgboost
  learning_rate: 0.1
  n_estimators: 100
```

Update your preprocessing/modeling scripts to read from this YAML (e.g., using `ruamel.yaml` or `omegaconf`).

---

### 5. 🧱 Example `dvc.yaml` for ZenML step wrapper (optional)

You can write `dvc.yaml` stages if you want full reproducibility:

```yaml
stages:
  preprocess:
    cmd: python pipeline/preprocess.py
    deps:
      - Data/raw/raw.csv
      - pipeline/preprocess.py
      - params.yaml
    outs:
      - Data/processed/clean.csv
  train:
    cmd: python pipeline/train.py
    deps:
      - Data/processed/clean.csv
      - pipeline/train.py
      - params.yaml
    outs:
      - models/model.pkl
      - reports/train_metrics.json
```

---

### 6. 🚀 Use DVC remote for large files (optional but recommended)

```bash
dvc remote add -d storage s3://your-bucket-name/path/
dvc push  # pushes data to remote
```

You can also use local remotes (e.g. `dvc remote add -d local_storage /mnt/dvc_storage`)

---

### 7. 🔄 Final Flow: Git + MLflow + ZenML + DVC

| Component   | Purpose                                  |
| ----------- | ---------------------------------------- |
| Git         | Source code versioning                   |
| DVC         | Data, artifacts, model tracking          |
| ZenML       | Pipeline orchestration                   |
| MLflow      | Model registry, experiment logs          |
| params.yaml | Parameterized config shared by DVC/ZenML |

---

## 🔁 In Your Code

Use `params.yaml` with `omegaconf` or `yaml`:

```python
from omegaconf import OmegaConf

config = OmegaConf.load("params.yaml")
imputer_strategy = config.preprocessing.imputation_strategy
```

---

## ✅ Pro Tips

- You don’t need to define every ZenML step in `dvc.yaml` — you can rely on ZenML to orchestrate, and use DVC just for artifact tracking.
- You **can version ZenML pipeline artifacts** (`.zen/mlflow.db`, `mlruns/`) via DVC too if needed.

Great — here's the complete and **production-grade DVC integration plan** for your ZenML-based MLDLC pipeline (version 2). This includes:

- Initial steps (1–3) unchanged as requested
- Highly detailed `params.yaml` covering all 10+ pipeline stages
- A fully modular `dvc.yaml` with rich stage definitions
- Default local cache, and an optional S3 remote setup (commented out)
- Clear usage instructions

---

## ✅ Step-by-Step DVC Integration for ZenML Pipelines

---

### **Step 1: Initialize DVC in your existing Git-tracked repo**

```bash
cd MLD_lifecycle  # Or your actual root
git init          # If not already done
dvc init
```

---

### **Step 2: Untrack data folders from Git and add to DVC instead**

Let’s say your `Data/`, `mlruns/`, `reports/`, `.zen/`, `artifacts/` and any EDA or drift folders are the candidates:

```bash
# Stop Git from tracking
echo 'Data/' >> .gitignore
echo 'artifacts/' >> .gitignore
echo 'mlruns/' >> .gitignore
echo '.zen/' >> .gitignore
echo 'reports/' >> .gitignore

# Now track with DVC
dvc add Data/
dvc add artifacts/
dvc add mlruns/
dvc add .zen/
dvc add reports/

# Commit changes
git add .gitignore Data.dvc artifacts.dvc mlruns.dvc .zen.dvc reports.dvc
git commit -m "Initialize DVC with data tracking"
```

---

### **Step 3: Create a Modular and Detailed `params.yaml`**

```yaml
# params.yaml

global:
  random_seed: 42
  target_col: "target"
  experiment_name: "MLDLC-AutoPipeline"

stage_1_ingestion:
  file_path: "Data/raw/data.csv"
  pii_masking: true

stage_2_eda:
  profile: true
  probabilistic_analysis: true
  drift_detection: true

stage_3_split:
  test_size: 0.1
  val_size: 0.1
  stratify: true

stage_4_preprocessing:
  missing_imputation:
    strategy_per_column:
      age: median
      salary: knn
      department: mode
  outlier_detection:
    method: iqr
    threshold: 1.5

stage_5_feature_engineering:
  encoding:
    type_per_column:
      gender: onehot
      education: ordinal
  transformation:
    numeric_scaling: standard
    categorical_scaling: none
  feature_construction:
    polynomial: false
    interactions: true
  feature_selection:
    method: auto
    k_best: 20

stage_6_modeling:
  model_type: xgboost
  hyperparameters:
    n_estimators: 100
    max_depth: 6

stage_7_evaluation:
  metrics: [accuracy, f1, auc]
  register_best_model: true

stage_8_tuning:
  search_algo: optuna
  n_trials: 30
  direction: maximize
  tracked_metric: f1

stage_9_reporting:
  generate_html: true
  report_dir: reports/
  plot_drift: true

stage_10_serving:
  mlflow_tracking_uri: "http://localhost:7000"
  model_name: "best_model"
  serve_locally: true
```

---

### **Step 4: Create a complete `dvc.yaml` file**

```yaml
# dvc.yaml

stages:
  ingest:
    cmd: poetry run python src/Stage_1_Ingestion/main.py
    deps:
      - Data/raw/data.csv
      - src/Stage_1_Ingestion/
    outs:
      - Data/interim/

  eda:
    cmd: poetry run python src/Stage_2_ED_Analysis/main.py
    deps:
      - Data/interim/
      - src/Stage_2_ED_Analysis/
    outs:
      - reports/eda/
    params:
      - stage_2_eda

  split:
    cmd: poetry run python src/Stage_3_Split_data/main.py
    deps:
      - Data/interim/
      - src/Stage_3_Split_data/
    outs:
      - Data/split/
    params:
      - stage_3_split

  preprocess:
    cmd: poetry run python src/stage_4_preprocessing/main.py
    deps:
      - Data/split/
      - src/stage_4_preprocessing/
    outs:
      - Data/preprocessed/
    params:
      - stage_4_preprocessing

  feature_engineer:
    cmd: poetry run python src/Stage_5_Feature_Engineering/main.py
    deps:
      - Data/preprocessed/
      - src/Stage_5_Feature_Engineering/
    outs:
      - Data/engineered/
    params:
      - stage_5_feature_engineering

  model_train:
    cmd: poetry run python src/Stage_6_Modeling/main.py
    deps:
      - Data/engineered/
      - src/Stage_6_Modeling/
    outs:
      - artifacts/models/
    params:
      - stage_6_modeling

  evaluate:
    cmd: poetry run python src/Stage_7_Evaluation/main.py
    deps:
      - artifacts/models/
      - src/Stage_7_Evaluation/
    outs:
      - reports/evaluation/
    params:
      - stage_7_evaluation

  tune:
    cmd: poetry run python src/Stage_8_HP_tuning/main.py
    deps:
      - Data/engineered/
      - src/Stage_8_HP_tuning/
    outs:
      - reports/hpt/
    params:
      - stage_8_tuning

  report:
    cmd: poetry run python src/Stage_9_Reporting/main.py
    deps:
      - reports/
      - src/Stage_9_Reporting/
    outs:
      - reports/final/
    params:
      - stage_9_reporting

  serve:
    cmd: poetry run python src/Stage_10_Serving/main.py
    deps:
      - artifacts/models/
      - src/Stage_10_Serving/
    params:
      - stage_10_serving
```

---

### **Step 5: Set up DVC remote**

By default, set up a local remote:

```bash
dvc remote add -d local_cache .dvc/cache
dvc remote modify local_cache type local
```

Optionally, configure an S3 remote:

```bash
# Optional: Add S3 remote (commented out in config)
# dvc remote add s3remote s3://your-bucket-name/path
# dvc remote modify s3remote access_key_id <YOUR_KEY>
# dvc remote modify s3remote secret_access_key <YOUR_SECRET>
# dvc remote default s3remote
```

---

### **Step 6: (Skipped)**

You asked to skip auto `dvc stage add` logic and ZenML coupling here — keeping as-is.

---

### **✅ Usage Instructions**

```bash
# Track changes
dvc repro                     # Reproduces full pipeline
dvc status                   # Check which stages need update
dvc push                     # Push data to remote cache (S3/local)
dvc pull                     # Pull data from cache (when cloning repo)

# Optional: generate lockfile
dvc lock
```

To serve a model:

- You should register and serve using `MLflow` (or FastAPI), but all artifacts used by ZenML can be versioned through this DVC pipeline.

---

Would you like these files zipped and exported too? Or want them as `.yaml` separately printed in chat?
