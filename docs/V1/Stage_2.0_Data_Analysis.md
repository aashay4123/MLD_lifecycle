## 2·0 — Phase 2.0 · Combined EDA for Mathematical & Probabilistic Approach <a name="2-phase-2.0--combined-eda"></a>

> **Goal** — bring together exploratory, advanced and probability-based analyses into a single, comprehensive stage. Stage 2.0 delivers mathematical diagnostics (skew, kurtosis, correlation, VIF), clustering & embedding metrics (Hopkins, PCA), information-theoretic measures (mutual information, entropy), and probabilistic modeling (distribution fitting, copulas, predictive intervals) on your DataFrame. All powered by **[`UnifiedPDA`](src/eda/unified_pda.py)**.

### Introduction

After raw ingestion and cleanup, it’s essential to probe deeper statistical and probabilistic structure before feature engineering.  
Stage 2.0:

- Quantifies distributional shape (skew, kurtosis) and tail behavior
- Flags multicollinearity (high Pearson r, VIF) to avoid unstable models
- Measures cluster tendency (Hopkins) and dimensionality reduction needs (PCA)
- Ranks features by information content (mutual information, entropy)
- Fits parametric distributions (AIC-based) and transforms via PIT/quantile methods
- Builds copula models to capture joint dependencies
- Estimates predictive uncertainty (Bayesian tests, normal predictive intervals)
- Generates rich visuals (heatmaps, scree, embeddings, QQ/PP) and HTML profiles

---

### 2·0·0 What Happens Under the Hood 🛠

1. **Basic EDA** (`ExploratoryDataAnalysis`)

   - Shape, dtype, missing & unique counts, memory footprint
   - Descriptive stats + skew/kurtosis, target–feature correlations (Pearson / point-biserial, Cramér’s V)
   - Correlation heatmap, optional pairplots, and leakage sniff via datetime

2. **Advanced EDA** (`AdvancedEDA`)

   - Categorical association heatmap (Cramér’s V)
   - Mutual information ranking vs. target (numeric & categorical)
   - Cluster tendency (Hopkins statistic)
   - Time-series decomposition & ACF/PACF plots
   - KMeans elbow & silhouette analyses
   - t-SNE embedding scatter

3. **Probabilistic Analysis** (`ProbabilisticAnalysis`)

   - Missing-value imputation (median or KNN)
   - Fit candidate dists (normal, exponential, gamma, beta, log-normal) via AIC
   - Shannon entropy per feature
   - Mutual information scores in parallel
   - Conditional probability tables for categoricals
   - PIT & quantile transforms for uniform/normal marginals
   - Bayesian group comparison (Welch’s t-test)
   - Predictive intervals under Gaussian assumption
   - Gaussian copula fitting for joint modeling
   - RandomForest-based feature importances
   - QQ & PP diagnostic plots

4. **Unified Orchestration**
   - `UnifiedPDA.run(...)` invokes each module in sequence
   - Saves plots, CSVs, HTML, and `manifest.json` under your chosen `out_dir`

---

### Key Mathematical & Probabilistic Concepts

- **Skewness & Kurtosis**: reveal asymmetry and tail weight—guide transforms (log, Box-Cox).
- **IQR & Outlier Detection**: outliers = 1.5× IQR beyond Q1/Q3—affect means/variances.
- **Pearson / Point-Biserial / Cramér’s V**: linear vs. binary vs. categorical associations.
- **Hopkins Statistic**: tests if data are clusterable vs. uniformly random.
- **Variance Inflation Factor (VIF)**: quantifies multicollinearity; VIF > 10 signals concern.
- **Shannon Entropy**: measures unpredictability in a feature’s distribution.
- **Mutual Information**: non-linear dependency strength between feature and target.
- **Probability Integral Transform (PIT)**: maps data to uniform under fitted CDF; basis for copulas.
- **Copula Modeling**: decouples marginals from joint dependency structure.
- **Bayesian Group Comparison**: Welch’s t-test for robust difference inference.
- **Predictive Intervals**: quantify uncertainty around point estimates (normal assumption).
- **PCA & Scree**: identify number of components explaining ≥ 90% variance.

---

### 🔧 Quick-Start

```bash
pip install pandas numpy scipy statsmodels scikit-learn pingouin copulas joblib matplotlib seaborn
```

```python
from src.eda.unified_pda import UnifiedPDA
import pandas as pd

# 1. Load your DataFrame (parse dates as needed)
df = pd.read_csv("data/my_data.csv", parse_dates=["signup_date"])

# 2. Initialize UnifiedPDA
uda = UnifiedPDA(
    df=df,
    target="label",            # optional: target column
    out_dir="reports/my_analysis"
)

# 3. Run full Stage 2.0 pipeline
report = uda.run(
    profile=True,    # generate ydata_profiling HTML if available
    pairplots=True,  # save seaborn pairplot
    impute_method="knn"
)

print("✓ Stage 2.0 EDA & PDA complete. See reports/my_analysis/")
```

---

### Outputs & Directory Structure

```
reports/my_analysis/
├─ advanced/
│  ├─ cramers_v_heatmap.png
│  ├─ mutual_info.csv, mutual_info.png
│  ├─ ts_decompose.png, acf_pacf.png
│  ├─ kmeans_elbow.png, kmeans_silhouette.png
│  ├─ tsne_embedding.png
├─ probabilistic/
│  ├─ distribution_fits.json
│  ├─ pit_transform.csv
│  ├─ quantile_transform.csv
│  ├─ bayesian_comparison.csv
│  ├─ predictive_intervals.csv
│  ├─ qq_<feature>.png, pp_<feature>.png
│  ├─ hist_<feature>.png, diag_<feature>.png
├─ profile.html           # if ProfileReport enabled
├─ pairplots.png         # if requested
└─ manifest.json
```

---

> **Next up ➜ Phase 3 · Data Preparation & Feature Engineering**
