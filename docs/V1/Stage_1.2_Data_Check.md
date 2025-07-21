````markdown
## 1·2 — Phase 1.2 · Pre-Training EDA Health Checks <a name="1-phase-1.2--pre-training-eda"></a>

> **Goal** — run a battery of Exploratory Data Analysis checks on a pandas DataFrame (dimensionality, missingness, skew, outliers, collinearity, VIF, etc.) and produce a self-contained HTML report. Everything is orchestrated by **[`DataFrameHealthCheck`](src/eda/dataframe_health_check.py)**.

### Introduction

Before you train any model, you need to know whether your dataset is fit for purpose.  
High-dimensional data (many features vs. few samples) can lead to overfitting or unstable estimates; missing or skewed values can bias your model; extreme outliers or multicollinear features can distort coefficients and degrade performance; imbalanced targets can lead to poor generalization on minority classes; and malformed dates or batch effects can hide temporal or grouping artifacts.  
`DataFrameHealthCheck` automates all these sanity checks so you can:

- Spot **dimensionality regimes** (p≫n vs. n≫p) and decide whether to reduce features or gather more data.
- Quantify **missingness** and plan imputation strategies.
- Identify **skewed** distributions that may warrant transformations.
- Detect **univariate outliers** and decide whether to cap or remove them.
- Uncover **high pairwise correlations** and **Variance Inflation Factors (VIF)** to flag multicollinearity before feature selection.
- Measure **target class imbalance** to choose sampling or weighting schemes.
- Validate **date columns** and **batch effects** to catch temporal inconsistencies or grouping biases.

### 1·2·0 What happens under the hood 🛠

1. **Dimensionality** → regime tag (p≫n / n≫p / p≈n)
2. **Missingness** → overall % + top 10 missing columns
3. **Data Types** → dtype counts
4. **Skewness** → top 10 numeric features by skew
5. **Categorical Cardinality** → top 10 high-cardinality features
6. **Outliers** → univariate outlier counts (IQR method)
7. **Collinearity** → top 10 feature pairs with |r|>0.9
8. **VIF** → top 10 features by Variance Inflation Factor
9. **Target Imbalance** (if `target_col` set)
10. **Date Issues** (if `datetime_cols` set) → parse %, min/max
11. **Batch Distribution** (if `batch_col` set)

---

### Concepts & Rationale

- **Dimensionality Regimes** (`p≫n` vs. `n≫p`):  
  When the number of features greatly exceeds the number of samples (p≫n), models can easily overfit. Conversely, when you have far more samples than features (n≫p), you may safely fit more complex models. Tagging the regime helps decide on feature selection or regularization strategies.

- **Skewness & Transformations**:  
  Many algorithms assume roughly symmetric (Gaussian-like) feature distributions. Highly skewed features (e.g. long tails) often benefit from log, Box-Cox, or Yeo-Johnson transforms to stabilize variance and improve model robustness.

- **Univariate Outliers (IQR Method)**:  
  Outliers are values lying beyond 1.5× IQR from the quartiles. They can disproportionately influence means and variances. Detecting them early lets you decide between capping, removal, or dedicated robust models.

- **Multicollinearity** (Correlation & VIF):  
  Highly correlated features (|r|>0.9) and high VIF (>10) indicate redundant information. Multicollinearity inflates coefficient variances in linear models and can lead to unstable predictions. Identifying these helps guide drop/merge decisions or use of dimensionality reduction.

- **Target Imbalance**:  
  In classification, if one class dominates, naïve models may ignore minority classes. Quantifying imbalance informs sampling (oversample/undersample), synthetic data generation (SMOTE), or class-weighting strategies.

- **Date Parsing & Batch Effects**:  
  Malformed or inconsistent datetime columns can break time series features; batch or group columns can introduce hidden stratification. Verifying parse rates and distributions ensures temporal integrity and unbiased evaluation.

---

### 🔧 Quick-Start

```bash
pip install pandas numpy scipy statsmodels
```
````

```python
from src.eda.dataframe_health_check import DataFrameHealthCheck
import pandas as pd

# load your data
df = pd.read_csv("data/sample.csv")

# instantiate checker
checker = DataFrameHealthCheck(
    df,
    target_col="label",            # optional: column to check class balance
    batch_col="batch_id",          # optional: column to summarize batches
    datetime_cols=["created_at"]   # optional: list of date columns to validate
)

# generate HTML report
html = checker.generate_report()

# save to disk
with open("reports/health_report.html", "w") as f:
    f.write(html)

print("✓ Report saved → reports/health_report.html")
```

---

> **Next up ➜ Phase 2 · Data Preparation & Feature Engineering**

```

```
