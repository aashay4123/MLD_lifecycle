````markdown
## 2·1 — Phase 2.1 · Probabilistic Analysis v2 <a name="2-phase-2.1--probabilistic-analysis-v2"></a>

> **Goal** — run a flexible, CLI-driven probabilistic analysis on your cleaned DataFrame in “auto” or “full” mode. This stage fits parametric distributions, measures entropy and mutual information, builds conditional probability tables, applies probability-integral and quantile transforms, runs Bayesian group comparisons, and fits a Gaussian copula. All powered by **[`ProbabilisticAnalysis`](probabilistic_analysis_v2.py)**.

### Introduction

Before feature engineering, it’s crucial to quantify the probabilistic structure of your data:

- **Distribution fitting** (AIC + KS-test) reveals the best parametric family for each numeric feature.
- **Shannon entropy** measures unpredictability in numeric or categorical data.
- **Mutual information** (or F-scores) ranks features by predictive power against the target.
- **Conditional probability tables** capture P(target|category) for each categorical feature.
- **Probability Integral Transform (PIT)** and **quantile transforms** normalize marginals for downstream modeling.
- **Bayesian group comparison** yields group means with 95% CIs.
- **Copula modeling** uncovers joint dependency structures independent of marginals.

Use `--mode auto` to skip sparse features, or `--mode full` to analyze every column.

---

### 2·1·0 What Happens Under the Hood 🛠

1. **Argument Parsing**

   - `--data`: path to cleaned Parquet
   - `--outdir`: directory for all outputs
   - `--mode`: `auto` (threshold-based) vs. `full` (all columns)
   - `--target`, `--quantile-output`, `--min-dist-count`, `--entropy-bins`, `--jobs`

2. **Data Loading**

   - Reads `args.data` via `pd.read_parquet()`

3. **Distribution Fitting** (`fit_all_distributions`)

   - Candidate families: norm, lognorm, gamma, beta, weibull_min
   - Select best by AIC; compute KS stat & p-value
   - Save `best_fit_distributions.json`

4. **Shannon Entropy** (`shannon_entropy`)

   - Numeric features → histogram bins; categoricals → frequency bins
   - Save `shannon_entropy.csv`

5. **Mutual Information / F-score** (`mutual_info_scores`)

   - Classification MI if target has ≤10 classes; F-score regression otherwise
   - One-hot encode features; save `mutual_info_scores.csv`

6. **Conditional Probability Tables** (`conditional_probability_tables`)

   - For each object-dtype column, compute P(target|category)
   - Save `cpt_<column>.csv` per feature

7. **Probability Integral Transform** (`pit_transform`)

   - ECDF-based CDF values → uniform marginals
   - Save `pit_transformed.csv`

8. **Quantile Transform** (`quantile_transform`)

   - Map numeric features to uniform or normal distribution
   - Save `quantile_transformed_normal.csv` or `quantile_transformed_uniform.csv`

9. **Bayesian Group Comparison** (`bayesian_group_comparison`)

   - For each categorical feature vs. numeric target: group mean + 95% CI
   - Save `bayesian_group_stats.csv`

10. **Copula Modeling** (`copula_modeling`)

    - Fit a `GaussianMultivariate` copula on numeric columns
    - Save `copula_params.json`

11. **Optional Diagnostics**
    - `qq_pp_plots(feature)` → QQ & PP plots
    - `feature_importance()` → permutation importance via RandomForest
    - `diagnostic_plots(feature)` → histogram + fitted PDF

---

### 🔧 Quick-Start

```bash
pip install pandas numpy scipy statsmodels scikit-learn copulas joblib
```
````

```bash
python probabilistic_analysis_v2.py \
  --data clean.parquet \
  --outdir reports/probabilistic_v2 \
  --mode auto \
  --target TARGET_COLUMN \
  --quantile-output normal \
  --min-dist-count 50 \
  --entropy-bins 20 \
  --jobs -1
```

---

### CLI Options

- `--data PATH` Path to cleaned Parquet file (required)
- `--outdir DIR` Directory to write JSON / CSV / plots (default: `reports/probabilistic_v2`)
- `--mode {auto,full}` `auto`: skip columns below count threshold; `full`: process all (default: `auto`)
- `--target NAME` Column name for target (enables MI, CPTs, Bayesian stats)
- `--quantile-output {normal,uniform}`
  Output distribution for quantile transform (default: `normal`)
- `--min-dist-count N` Min non-null values to fit distributions in `auto` (default: 50)
- `--entropy-bins N` Bins for numeric entropy (default: 20)
- `--jobs N` Parallel jobs for distribution fitting (default: all CPUs)

---

### Outputs & Directory Structure

```
reports/probabilistic_v2/
├─ best_fit_distributions.json
├─ shannon_entropy.csv
├─ mutual_info_scores.csv         # only if --target provided
├─ cpt_<feature>.csv …            # one per categorical feature
├─ pit_transformed.csv
├─ quantile_transformed_<mode>.csv
├─ bayesian_group_stats.csv       # only if --target provided
├─ copula_params.json
├─ qqpp_<feature>.png             # when run manually
├─ diagnostic_<feature>.png       # when run manually
```

---

> **Next up ➜ Phase 2·2 · Data Preparation & Feature Engineering**

```

```
