## 3 — Phase 3 · **Data Preparation**<a name="3-phase-3--data-preparation"></a>

> **Goal** — turn a raw snapshot from Phase-2 into a _model-ready_, versioned,
> privacy-hardened dataset in `data/processed/`, plus an interim copy in
> `data/interim/`.
> All logic lives in
> **[`src/ml_pipeline/prepare.py`](src/ml_pipeline/prepare.py)** —
> a configurable pipeline class (**`DataPreparer`**).

---

### 3A Schema Validation & Data Types<a name="3a-schema-validation--data-types"></a>

| Tool                        | What it does                                              | Where                                |
| --------------------------- | --------------------------------------------------------- | ------------------------------------ |
| **Pandera**                 | enforce column names, dtypes, value ranges, allowed enums | `schema = pa.DataFrameSchema({...})` |
| **pyjanitor**               | snake-cases column names (`df.clean_names()`)             | first line of `load_and_validate()`  |
| Data-quality tests (opt-in) | `great_expectations` (`--gx`)                             | `dq_validate()`                      |

**Why:** catch bad upstream changes early; guarantee downstream code never
breaks on dtype surprises.

---

### 3B.1 De-duplication & Invariant Pruning <a name="3b1-dedup"></a>

- `--dedup uid` → drops perfect-duplicate _rows_.
- `--prune-const 0.99` → removes columns where one value ≥ 99 %.

---

### 3B Missing-Value Strategy<a name="3b-missing-value-strategy"></a>

_Default_: median (numeric) + mode (categorical).
_Optional_: `--knn` flag enables **`KNNImputer`** (k=5).

| Technique      | Flag              | Notes                                |
| -------------- | ----------------- | ------------------------------------ |
| Median / Mode  | _(default)_       | fast & deterministic                 |
| **KNNImputer** | `--knn`           | non-linear numeric guess             |
| Drop column    | `--drop-miss 0.4` | removes any feature with > 40 % NaNs |
| Drop row       | `--drop-miss 0.4` | removes any row with > 40 % NaNs     |

```bash
python -m ml_pipeline.prepare --knn      # fancy impute
```

_Diagnostics:_ generates a `missingno.matrix` plot for the first 1 000 rows. (saved under `reports/lineage`).

---

### 3C Outlier Detection & Treatment<a name="3c-outlier-detection--treatment"></a>

| Method             | Flag                      | Notes                          |                    |                            |
| ------------------ | ------------------------- | ------------------------------ | ------------------ | -------------------------- |
| IQR fence (1.5×)   | `--outlier iqr` (default) | quick & interpretable          |                    |                            |
| Z-score (          | z                         | < 3)                           | `--outlier zscore` | good for gaussian-ish data |
| Isolation Forest   | `--outlier iso`           | detects multivariate anomalies |                    |                            |
| Local Outlier Fac. | `--outlier lof`           | cluster-shaped data            |

---

### 3D Data Transformation & Scaling<a name="3d-data-transformation--scaling"></a>

| Transform                          | Flag                       | Comment                    |
| ---------------------------------- | -------------------------- | -------------------------- |
| log-transform on `amount`          | on by default (`np.log1p`) | stabilise heavy-tail       |
| **StandardScaler**                 | `--scaler standard`        | zero-mean / unit-var       |
| **RobustScaler** (IQR)             | `--scaler robust`          | heavy-outlier datasets     |
| **PowerTransformer (Yeo-Johnson)** | `--scaler yeo`             | make data closer to normal |

**Examples**:

```bash
python -m data_cleaning.data_preparation --scaler robust
python -m data_cleaning.data_preparation --outlier iso --scaler yeo
```

---

### 3E Class / Target Balancing<a name="3e-class-target-balancing"></a>

| Technique                   | Flag                 | Use-case                  |
| --------------------------- | -------------------- | ------------------------- |
| **SMOTE** over-sampling     | `--balance smote`    | minority boost            |
| **NearMiss** under-sampling | `--balance nearmiss` | huge majority down-sample |

```bash
python -m ml_pipeline.prepare --balance smote
```

---

### 3F Data Versioning & Lineage<a name="3f-data-versioning--lineage"></a>

- Saves **both** `data/interim/clean.parquet` (pre-scale) _and_
  `data/processed/scaled.parquet` (final).
- Writes `reports/lineage/prep_manifest.json`, e.g.

```jsonc
{
  "timestamp": "2025-05-30T12:42:01",
  "rows": 104876,
  "scaler": "robust",
  "outlier": "iso",
  "balance": "smote",
  "raw_sha": "7b12e0f83e01"
}
```

---

### 3G Feature Pruning (High NaN / High Corr) <a name="3g-prune"></a>

- **NaN threshold** `--drop-miss p` → prune if NaNs > p
- **Corr threshold** `--drop-corr 0.95` → greedily drop highly-correlated pair

Manifest of drops saved to `reports/lineage/prune_log.json`.

---

### 🔧 Quick-Start Cheat-Sheet

```bash
# 1. Default happy-path (median/mode, IQR, standard scale)
python -m ml_pipeline.prepare

# 2. Robust pipeline for gnarly data
python -m ml_pipeline.prepare \
       --knn \
       --outlier iso \
       --scaler robust \
       --balance smote
```

---

## 3.H. [Feature Selection & Early Train/Test Split](#4.5-feature-selection--split)

> **Why here?** Any statistic that _uses_ the target (variance filter,  
> mutual-information, Cramer-V, leakage sniff, etc.) must be learned on
> **training rows only**.  
> Therefore we:
>
> 1. **Split once — right now** (80 / 20 stratified by `target`  
>    or `--time-split` if temporal).
> 2. **Fit feature filters on _train_**, replay them on _val_ / _test_.
>    | Sub-step | Purpose | Script | Artefact |
>    | --------------------------- | ------------------------------------- | --------------------- | --------------------------------------------- |
>    | **4·½·0 Split** | Freeze leak-free `train / val / test` | `feature_selector.py` | `data/splits/*.parquet` `split_manifest.json` |
>    | **4·½·1 Low-variance drop** | remove near-constant cols | ″ | logged in manifest |
>    | **4·½·2 Target filter** | MI / chi² < threshold | ″ | `"kept","dropped"` lists |
>    | **4·½·3 Collinearity** | drop one of pairs with ρ > 0.95 | ″ | correlation heatmap |
>    | **4·½·4 Save plan** | Column lists for next phases | `"feature_plan.json"` |

```bash
 # full run – stratified split, MI filter @ 0.001, corr prune @ 0.95
 python -m Data_Analysis.feature_selector \
    --target is_churn \
    --mi-thresh 0.001 \
    --corr-thresh 0.95 \
    --seed 42
```

**Exit checklist** _ ✅ `data/splits/train.parquet` & `test.parquet` exist  
 _ ✅ `feature_plan.json` lists “keep” & “drop” columns  
 _ ✅ No feature on the **drop list** is referenced downstream  
 _ ✅ Issue **“Phase 4·½ Complete → start Phase 5 FE”** created

---
