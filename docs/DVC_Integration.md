# DVC Integration for MLD Lifecycle

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
