````markdown
## 3·0 — Phase 3 · Split, Multi-Baseline & Preprocessor Freeze <a name="3-phase-3--split-baseline"></a>

> **Goal** — take your cleaned & processed dataset, split into train/val/test (with optional SMOTE), train and evaluate multiple dummy baseline models, pick and record the winning baseline (with checksums), run basic sanity checks, and freeze the numeric preprocessor for downstream pipelines.

### Introduction

After you’ve ingested, validated, and explored your data, it’s time to establish a reproducible baseline and lock in your preprocessing.  
Phase 3 ensures that:

- Your data splits are deterministic and, if desired, stratified or oversampled to handle class imbalance.
- Multiple naive “dummy” models (mean/median for regression, most_frequent/stratified/uniform for classification) are trained and scored, so you always have a sanity-check performance floor.
- The best performing baseline is saved with its checksum for auditability, alongside a full metrics report for all candidates.
- Fundamental sanity checks (no leakage or duplicate rows) guard against common pipeline mistakes.
- The numeric preprocessor (StandardScaler) is fitted on the full dataset and frozen for consistent feature transformations in later phases.

---

### 3·0·0 What Happens Under the Hood 🛠

1. **Split Data** (`split_data`)

   - 80/10/10 train/val/test split, with optional stratification on the target
   - Optional SMOTE oversampling on the training fold only (classification)
   - Persist `train.parquet`, `val.parquet`, `test.parquet` under `data/splits/`
   - Write `split_manifest.json` with seed, stratify, oversample flags and row counts

2. **Build Baseline Models** (`build_baseline`)

   - **Regression** candidates: DummyRegressor strategies `mean`, `median`, `quantile`
   - **Classification** candidates: DummyClassifier strategies `most_frequent`, `stratified`, `uniform`
   - Fit each on training set, evaluate on validation set (MAE & R² for regression; accuracy & F1 for classification)
   - Save each model under `models/baselines/` and compute its SHA-256 checksum
   - Generate `baseline_metrics.json` summarizing all candidates and their scores
   - Write `baseline_manifest.json` pointing to the winning model and checksum

3. **Sanity Checks** (`sanity_checks`)

   - Ensure no duplicate rows across train/test splits (index-based)
   - Detect any feature column identical to the target (simple leakage sniff)

4. **Freeze Preprocessor** (`freeze_preprocessor`)
   - Fit a `Pipeline([("scale", StandardScaler())])` on all numeric columns of the processed dataset
   - Save `models/preprocessor.joblib` and record its SHA-256 in `preprocessor_manifest.json`

---

### 🔧 Quick-Start

```bash
pip install pandas numpy scikit-learn imbalanced-learn joblib
```
````

```bash
python split_and_baseline.py \
  --target TARGET_COLUMN \
  [--seed 42] \
  [--stratify] \
  [--oversample]
```

---

### CLI Options

- `--target NAME` **(required)** target column name
- `--seed INT` random seed for reproducibility (default: 42)
- `--stratify` stratify splits by target distribution
- `--oversample` apply SMOTE to training fold (classification only)

---

### Outputs & Directory Structure

```
data/
└─ processed/scaled.parquet         # input processed data
data/
└─ splits/
   ├─ train.parquet
   ├─ val.parquet
   ├─ test.parquet
   └─ split_manifest.json

reports/
└─ baseline/
   └─ baseline_metrics.json

models/
├─ baselines/
│  ├─ mean_regressor.joblib
│  ├─ median_regressor.joblib
│  ├─ quantile0.25_regressor.joblib
│  ├─ most_frequent_clf.joblib
│  ├─ stratified_clf.joblib
│  └─ uniform_clf.joblib
│  └─ baseline_manifest.json
└─ preprocessor.joblib
└─ preprocessor_manifest.json
```

---

> **Next up ➜ Phase 4 · Data Preparation & Feature Engineering**

```

```
