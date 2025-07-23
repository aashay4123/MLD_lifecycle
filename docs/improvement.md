# 🚀 Production-Grade Data Preprocessing Pipeline Specification

> **Scope:** End-to-end design of a robust, reproducible, scalable data preprocessing pipeline.

> **Main goals:**
> ✅ Centralized configuration
> ✅ Comprehensive data validation
> ✅ Robust missing/outlier handling
> ✅ Reproducible transformations
> ✅ Clear reports & artifacts
> ✅ Ready for ML pipelines / ZenML integration

---

## 🛠️ Stage 0: Configuration & Metadata

**Objectives:**

- Centralize thresholds, file paths, and reproducibility.
- Capture full environment metadata for reproducibility (code version, config, random seed).

**Key tasks:**

- Create `config.yaml` with explicit sections for:

  - Paths (raw/processed data, logs, reports)
  - Thresholds (missing, outliers, scaling)
  - Random seeds
  - Reporting options

- Record metadata:

  - Current git SHA (`git rev-parse HEAD`)
  - Timestamp
  - Config file checksum (`sha256sum config.yaml`)
  - Python + dependency versions (`pip freeze > requirements-lock.txt`)

- Store metadata in JSON or YAML with every pipeline run.

**Checks & Metrics:**

- Validate config schema on startup (e.g., with [Cerberus](https://docs.python-cerberus.org/)).
- Abort with clear error if required keys are missing or invalid types.

---

## 📥 Stage 1: Data Ingestion

**Objectives:**

- Load data consistently, enforce expected schema, persist a canonical snapshot.

**Key tasks:**

- Load source data from CSV, Parquet, SQL, etc.
- Enforce schema:

  - Expected columns, types, and nullability.
  - Drop unexpected columns or flag them.
  - Validate dtypes and attempt casting; fail on mismatch.

- Log detailed row/column counts and type coercion actions.
- Save cleaned snapshot as Parquet with checksum (SHA256) for integrity.

**Checks & Metrics:**

- Row count matches source logs.
- Column count matches schema.
- Dataframe checksum recorded for downstream verification.

---

## 🔎 Stage 2: Data Validation & Sanity Checks

**Objectives:**

- Detect structural, semantic, and cross-field issues before processing.

**Key tasks:**

- **Mixed-Type Detection**: columns with multiple Python types.
- **Impossible Values**: forbidden sets (e.g., negative ages, invalid dates).
- **Unexpected High Cardinality**: flag columns exceeding unique ratio threshold.
- **Near-Zero-Variance (NZV)**: drop or log columns with too few unique levels.
- **Duplicates**: detect and optionally remove identical rows.
- **Custom Missing Markers**: detect and recode placeholders like `"?"`, `"N/A"`, etc.
- **Constant Columns**: drop features with zero variance.

**Checks & Metrics:**

- Detailed anomaly report: which columns failed which check, with frequencies.
- Automatically export validation report (JSON/HTML).

---

## 📉 Stage 3: Missingness Analysis & Imputation

**Objectives:**

- Identify missingness mechanisms (MCAR, MAR/MNAR).
- Choose appropriate imputation or drop strategies.

**Key tasks:**

- Little’s MCAR test on full dataset.
- Per-column logistic regression for missingness mechanism.
- Stratified missingness by target (if supervised).
- Drop columns exceeding missingness threshold.
- Impute:

  - Numeric: mean/median/knn/random sample; fallback if dataset too big.
  - Categorical: mode/constant/random-sample.

- **Mixed-type columns**: detect columns stored as strings but mostly numeric, convert if ≥90% digit-like.

**Checks & Metrics:**

- Per-column missing fraction & chosen strategy.
- Compare distribution shifts (e.g., KS p-value) pre- vs post-imputation.
- Log covariate correlation shift after imputation.

---

## 🚨 Stage 4: Outlier Detection & Treatment

**Objectives:**

- Detect univariate and multivariate outliers, apply winsorization or drop.

**Key tasks:**

- **Univariate rules**: IQR, Z-score, ModZ, Tukey fences, percentile capping.
- **Multivariate rules**:

  - Mahalanobis distance (only if `n_rows >= 5×n_features` & approximate normality).
  - LOF if dataset small or low-dimensional.
  - IsolationForest otherwise.

- Voting system: mark rows with ≥ threshold votes as outliers.
- Treatment:

  - Winsorize at percentiles or drop rows.

**Checks & Metrics:**

- Outlier votes per row, final flagged rows, treatment summary.
- Skew/kurtosis before & after treatment to confirm improvement.

---

## ⚖️ Stage 5: Scaling & Transformation

**Objectives:**

- Normalize numeric features to improve model stability.
- Make distributions more Gaussian if needed.

**Key tasks:**

- Drop NZV columns again after cleaning.
- Choose scaler based on skew/kurtosis:

  - RobustScaler → if any |skew| or |kurtosis| above thresholds.
  - StandardScaler → if distributions already near-Gaussian.
  - MinMaxScaler → fallback.

- Fit chosen scaler on train set.
- Evaluate post-scale distributions using Shapiro-Wilk p-values.
- If distributions remain highly non-Gaussian, try transformations (Box-Cox, Yeo-Johnson, QuantileTransform).
- Persist fitted scaler & transformers for consistent application on future data.

**Checks & Metrics:**

- Pre/post scaling distribution report.
- Chosen scaler + reasoning (e.g., high skew).
- Transformation parameters saved with metadata.

---

## 🔨 Stage 6: Feature Engineering & Selection

**Objectives:**

- Enhance data with derived features and reduce redundancy.

**Key tasks:**

- Create optional interaction terms for highly correlated features.
- Auto-bin numeric features using quantiles.
- Encode categoricals:

  - Target/frequency encoding for high-cardinality columns.
  - One-hot or ordinal encoding otherwise.

- Prune correlated features (e.g., drop one of pair if |corr| > threshold).
- Run Recursive Feature Elimination (RFE) or feature importance ranking.

**Checks & Metrics:**

- Final feature count.
- Variance Inflation Factor (VIF) to measure multicollinearity.
- Feature importance snapshot for interpretability.

---

## 🔪 Stage 7: Train/Validation/Test Split & Baseline

**Objectives:**

- Split data into reproducible partitions.
- Compute baseline model metrics.

**Key tasks:**

- Stratified train/val/test splits (60/20/20 recommended) on target variable if classification.
- Apply oversampling (e.g., SMOTE) on training set if class imbalance is significant.
- Compute baseline metrics:

  - Classification → majority class accuracy, F1.
  - Regression → mean predictor MAE, R².

- Optionally train naive baselines (k-NN, Naive Bayes).

**Checks & Metrics:**

- Index overlap check between splits.
- Baseline metrics in JSON for tracking.

---

## 🔒 Stage 8: Leakage Detection & Integrity

**Objectives:**

- Ensure no train/test leakage or features with hidden target information.

**Key tasks:**

- Correlation of each feature with target in train vs. test; unexpected jumps → flag.
- Check timestamps or IDs that perfectly separate splits → flag potential leakage.
- Check for identical features with target values (e.g., accidental data leak).

**Checks & Metrics:**

- Per-feature leakage scores.
- Warnings if high correlations or perfect predictors found.

---

## 📦 Cross-Cutting Enhancements

✅ Centralized config: thresholds & paths in one file.
✅ Random seed control everywhere → reproducibility.
✅ Logging: unified, level-configurable, per-module loggers.
✅ Versioning: save git SHA & config hash with reports.
✅ Time-stamped report filenames: avoid overwriting.
✅ Save artifacts: each fitted object (scaler, encoder) serialized to disk.
✅ Drift monitoring: store pre/post distributions for later comparison.
✅ Flexible I/O: support CSV, Parquet, Feather via `config.io_format`.
✅ CLI/ZenML compatibility: package stages as reusable steps.

---

## 📑 Deliverables

- Canonical cleaned dataset in `data/processed/`.
- Comprehensive validation report.
- Per-stage JSON reports with metrics.
- Fitted pipeline components saved as pickle/joblib.
- Metadata summary (timestamp, config, git SHA).
- Makefile with `make train`, `make test`, `make devup`, `make devdown`, `make clean`.

---

📝 5) Reporting
Include column memory usage in report to flag problematic columns.

Add status field in report (e.g., skipped_constant_or_insufficient, processed, fallback_applied).

Write full candidate info as JSON in report’s candidates column.

Capture and log features skipped due to invalid data, with reason included in report.

Add info on enforced transform for multimodal features.

Include final stats (mean, std, skew, kurtosis, entropy) for each feature after transformation.

Add consistent feature name column in report for easy indexing.

⚙️ 6) Transform Logic
Preserve column order during transform even if DataFrame’s columns changed.

Handle missing columns gracefully during transform with warnings.

Ensure non-numeric columns are untouched and left in-place.

Add clear error if trying to transform before calling fit().

Parallelize transformations if dataset is very wide.

Avoid repeated transformer/scaler fitting during transform.

💡 7) Robustness
Always validate inputs for missing values before transform and log result.

Add support for input DataFrames with integer dtypes (cast to float).

Support explicit seed for reproducibility; default random_state=42.

Track columns dropped or added between fit/transform.

Gracefully handle columns with all NaNs in transform.

Validate column data types on both fit/transform to catch schema drift.

Add clear error/warning if expected numeric columns are missing at transform-time.

📊 8) Visualizations & Artifacts
Allow optional disabling of plotting with generate_plots=False flag.

Optimize plotting (reduce saved figures if >100 columns).

Log visualizations and report as MLflow artifacts if MLflow integration is active.

Automatically generate HTML report after fit() or on-demand.

🔗 9) MLflow & Integrations
Add optional MLflow logging of report and plots.

Use monitor() decorator for performance tracking of key methods.

Log MLflow artifacts with proper step/experiment naming.

Include environment details (Python version, library versions) in report or MLflow metadata.

Support saving/loading the full fitted state including adaptive thresholds, chosen candidates, and enforced transforms.

✅ 10) Miscellaneous
Remove redundant code: e.g., repeated calls to \_shapiro_skew() replaced with robust \_normality_scores().

Use consistent logging levels (info vs. warning vs. debug).

Document new parameters in class docstring and usage guide.

Ensure QuantileTransformer always uses random_state for deterministic behavior.

Add version info of FeatureScalerTransformer in report metadata (e.g., "FeatureScalerTransformer v3.0").

<!-- Links for Automl -->

https://github.com/pycaret/pycaret
https://github.com/AutoViML/Auto_ViML
https://github.com/mljar/mljar-supervised
https://github.com/AxeldeRomblay/MLBox/blob/master/mlbox/model/regression/stacking_regressor.py
https://khiops.org/tutorials/Notebooks/Use_in_any_ML_pipeline/

https://github.com/mlflow/mlflow/tree/master/docs
https://github.com/AutoViML/AutoViz
https://github.com/cod3licious/autofeat/tree/main/src/autofeat
https://github.com/automl/DeepCAVE
https://github.com/Shriram-Vibhute/CampusX-DSMP2.0/tree/main
https://github.com/Yash-Kavaiya/CampusX-courses/tree/main
https://github.com/zenml-io/zenml-workshop-mlops
https://github.com/zenml-io/zenml-projects
