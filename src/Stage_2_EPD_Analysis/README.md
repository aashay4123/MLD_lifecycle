## # Stage 2: EPD Analysis - README

### ✅ EDAnalyzer.py

- **Contains:** `EDAnalyzer` class (Basic + Advanced EDA).
- **Details:**

  - Runs univariate stats with missingness, skew, kurtosis.
  - Bivariate numeric-numeric, numeric-categorical, and categorical-categorical associations.
  - Correlation heatmaps, PCA scree, VIF, Mardia's multivariate normality, Hopkins statistic.
  - Target column analysis: numeric/categorical correlation with target, distribution plots, leakage checks.
  - Parallelizes time-consuming tasks and limits figure generation with `max_figures`.
  - Automatically saves detailed CSV and PNG outputs.

- **Use when:** you need thorough exploratory analysis on features (numerical + categorical) and target behavior.

---

### ✅ PDAnalysis.py

- **Contains:** `ProbabilisticAnalysis` class.
- **Details:**

  - Fits candidate distributions for each numeric feature (AIC, KS, Anderson-Darling).
  - Computes Shannon entropy for numeric and categorical features.
  - Calculates Cramér’s V, Theil’s U, Kendall’s Tau, Spearman’s correlation matrices.
  - Computes target-dependency scores with adaptive linear/non-linear tests (f-statistics & mutual information).
  - Performs drift detection using Jensen-Shannon divergence, PSI, KL divergence.
  - Fits Gaussian copula models to numeric features for dependency modeling.
  - Generates QQ/PP diagnostic plots, PIT transforms, quantile transforms, Bayesian group comparisons, and feature importance with permutation.
  - Saves structured artifacts in JSON, CSV, and images for full auditability.

- **Use when:** you need deep probabilistic diagnostics, drift detection, or want to model dependencies statistically.

---

### ✅ UnifiedEPDA.py

- **Contains:** `UnifiedEPDA` class (high-level orchestrator).
- **Details:**

  - Runs both EDAnalyzer and PDAnalysis classes.
  - Merges EDA and probabilistic reports into one unified manifest.
  - Generates t-SNE projections, clustering metrics, target correlations, advanced stats.
  - Limits runtime for very wide datasets by sampling top features by variance in `mode=auto`.
  - Centralizes pipeline reporting, useful for a **one-step comprehensive report**.

- **Use when:** you want a single entry point producing complete diagnostics in one go.

---

### ✅ EPDA.py

- **Contains:** lightweight **EPDA** script/class (quick EDA).
- **Details:**

  - Implements a fast, surface-level exploration.
  - Skips detailed drift analysis or advanced diagnostics.
  - Useful as a **quick glance tool** or for running automated EDA checks before deeper analysis.
  - Great for datasets where you just need an initial understanding in seconds/minutes.

- **Use when:** you need a **fast sanity check** or integration in CI pipelines for basic health checks.

---

### ✅ PED_Analysis.py

- **Contains:** PED Analysis orchestrator (function or class).
- **Details:**

  - Designed specifically to wrap EDAnalyzer, AdvancedEDA, PDAnalysis, or UnifiedEPDA in a **ZenML-friendly step**.
  - Supports performance monitoring, artifact tracking, and clean handoff of outputs for MLOps pipelines.
  - Standardizes outputs for ML pipeline logging (e.g., MLflow, ZenML).
  - Could expose a single function like `run_PED_analysis(df, target)` to trigger all steps coherently.

- **Use when:** you want a **production-ready pipeline step**, plugging advanced EDA directly into your ZenML or custom pipeline.

---

## 🔎 Key Differences

| File                | Primary Purpose                                      | Depth            | Pipeline Integration |
| ------------------- | ---------------------------------------------------- | ---------------- | -------------------- |
| **EDAnalyzer.py**   | Full exploratory data analysis                       | Deep EDA         | Manual or integrated |
| **PDAnalysis.py**   | Probabilistic analysis + drift & distribution checks | Deep probability | Manual or integrated |
| **UnifiedEPDA.py**  | Single command for full EDA & probabilistic analysis | Deep + unified   | Manual or integrated |
| **EPDA.py**         | Fast superficial EDA                                 | Light            | Fast checks / CI     |
| **PED_Analysis.py** | ZenML-friendly orchestrator                          | Flexible         | MLOps-ready          |

---

## ⚙️ Usage Overview

- **Run EDA** alone:

  ```bash
  python EDAnalyzer.py --input data.csv --outdir eda_reports
  ```

- **Run Probabilistic Analysis** alone:

  ```bash
  python PDAnalysis.py --data data.csv --outdir prob_reports
  ```

- **Run Unified EPDA** (EDA + probabilistic):

  ```bash
  python UnifiedEPDA.py --data data.csv --outdir unified_reports
  ```

- **Quick check with EPDA.py**:

  ```bash
  python EPDA.py --data data.csv --outdir quick_reports
  ```

- **Production pipeline step with PED_Analysis.py**:

  ```python
  from PED_Analysis import run_PED_analysis
  run_PED_analysis(df, target="my_target")
  ```

---

## 📦 Directory Manifest

- `EDAnalyzer.py` → Basic & Advanced EDA
- `PDAnalysis.py` → Probabilistic Analysis
- `UnifiedEPDA.py` → Full EDA + Probabilistic
- `EPDA.py` → Superficial, quick EDA
- `PED_Analysis.py` → ZenML-friendly orchestration

---

✅ **All files together provide maximum flexibility**, letting you pick quick checks, deep dives, or unified pipelines — **plus a fully pipeline-ready orchestrator for production**.
