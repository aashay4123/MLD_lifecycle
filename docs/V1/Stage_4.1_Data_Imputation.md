# `MissingImputer` – Production-Grade Missing Data Handler

This module implements an **automated, stateful, and statistically-aware imputation pipeline** for both numeric and categorical features. It's designed for production ML pipelines where interpretability, reproducibility, and distributional integrity matter.

---

## 🔧 What It Does

✅ Auto-detects missingness mechanism (MCAR/MAR)
✅ Per-column strategy selection (mean, median, KNN, MICE, random sample, etc.)
✅ Uses statistical scoring (KS test, variance ratio, covariance shift)
✅ Smart handling of categorical variables with TVD scoring
✅ Reuses KNN/MICE imputations efficiently (no redundant recompute)
✅ Auto-serializes full imputation state on `fit()` → `.pkl`
✅ Loads from disk on `transform()` if already fit
✅ Full logging + JSON reporting of imputations per column

---

## 🚀 How It Works

### Fit Phase (`imputer.fit(df)`)

1. **Detect missingness mechanism**:

   - Run Little’s MCAR test
   - Use per-column logistic regression to estimate MCAR vs MAR/MNAR

2. **Drop severe cases** (if enabled):

   - Drop columns with > `max_col_nan_fraction`
   - Drop rows with > `max_row_nan_fraction`

3. **Imputation strategy selection** (per column):

   - Try multiple strategies
   - Score each using:

     - KS statistic (distribution preservation)
     - Variance ratio
     - Covariance shift (vs non-missing)

   - Pick best scoring strategy

4. **Avoid recomputing expensive imputers**:

   - KNN and MICE imputations are done **once per block** and reused across columns

5. **Save model to disk**:

   - Stores imputer object, strategies, dropped columns, and all fitted models in `.pkl`

---

### Transform Phase (`imputer.transform(df)`)

1. **Load model from disk** (if not already loaded)
2. **Use saved imputers per column**
3. **Apply same imputation logic as in training**
4. **Do not re-evaluate or refit** anything
5. **Returns clean DataFrame**

---

## ✅ Best Practice Checklist

| Step                        | Best Practice                                                                |
| --------------------------- | ---------------------------------------------------------------------------- |
| ❓ Assess Missingness       | Use Little’s test + per-column logistic regression                           |
| 📉 Drop thresholding        | Drop columns > 50% missing, rows > 30% missing features                      |
| 🔢 Univariate Imputation    | Mean/Median/Mode/Constant + indicator flag                                   |
| 🔄 Multivariate Imputation  | Use KNN (fast, local) or MICE (powerful, slower) only **once**, reuse result |
| 🔁 Random Sample Impute     | For low missingness MCAR features, preserves distribution                    |
| 🧠 Strategy Selection       | Use scoring (KS, variance, cov) to choose imputers                           |
| 🪪 Missing Indicator Columns | Add `.isna()` flags where info-rich missingness is suspected                 |
| 🗂️ Storage                  | Auto-pickle fitted model (strategies + imputers)                             |
| 🧪 Avoid Data Leakage       | Fit on train only, use transform for val/test                                |
| 🧾 Logging + Reporting      | Store imputation report as JSON + optional Markdown                          |

---

## 🧬 Example Usage

```python
from missing_imputer import MissingImputer

imputer = MissingImputer(
    max_col_nan_fraction=0.5,
    max_row_nan_fraction=0.3,
    enable_random_sample=True,
    enable_mice=True,
    enable_knn=True
)

# Fit and auto-pickle model
imputer.fit(train_df)

# Later...
# Automatically loads model from disk and transforms
val_clean = imputer.transform(val_df)
```

---

## 📁 Output

- `imputer.pkl`: Serialized model (strategies, imputers, parameters)
- `report.json`: Per-column imputation report
- Optional: Markdown summary (for docs)
