# One‑Shot AutoML Algorithm Blueprint

```text
FUNCTION one_shot_auto_ml(df: DataFrame, task: {“classification”|“regression”}) →

    ┌──────────────────────────┐
    │ 0. INITIAL DATA CHECK   │
    └──────────────────────────┘
1.  schema ← extract column names, dtypes
2.  summary ← for each col: count, unique, %missing, basic stats
3.  present summary to user (table + key warnings)

    ┌───────────────────────────────────────────────────────────┐
    │ 1. MISSING‑VALUE HANDLING (USER‑GUIDED)                  │
    └───────────────────────────────────────────────────────────┘
4.  miss_cols ← [col for col in df if df[col].has_missing()]
5.  For each col in miss_cols:
      a. infer_type ← numeric vs categorical vs datetime
      b. suggest_strategies:
           • numeric: {“mean”, “median”, “KNN”, “model”}
           • categorical: {“mode”, “constant='missing'”, “model”}
      c. prompt_user(col, infer_type, suggested_strategies) → strategy[col]
6.  Apply imputation: for each col, fit on train slice and transform all data

    ┌───────────────────────────────────────────────────────────┐
    │ 2. ENCODING & TRANSFORMATION (PARTIAL USER‑OVERRIDE)     │
    └───────────────────────────────────────────────────────────┘
7.  cat_cols ← detect categorical columns (dtype == object or < threshold unique)
8.  For each col in cat_cols:
      a. card ← df[col].n_unique()
      b. default_encoder ← if card ≤ 10 then “one‑hot” else “target”
      c. prompt_user(col, card, default_encoder, choices=[“one‑hot”,“ordinal”,“target”,“embedding”]) → encoder[col]
9.  Apply encoders: fit on train, transform all splits
10. detect numeric_cols ← remaining numeric features
11. For each col in numeric_cols:
      a. dist ← test normality/skew
      b. default_scaler ← select Standard / Power / Robust
      c. (optional) prompt_user? [skip by default]
12. Apply selected scalers to numeric_cols

    ┌───────────────────────────────────────────────────────────┐
    │ 3. OUTLIER DETECTION & HANDLING (AUTO or ASK)            │
    └───────────────────────────────────────────────────────────┘
13. threshold_rules ← {
       “univariate”: IQR(1.5) and |z|>3,
       “multivariate”: IsolationForest(contamination=auto)
    }
14. If user_opt_in(“Handle outliers automatically?”):
       apply both rules → cap or drop based on severity heuristic
    Else:
       detect only & log outlier counts; leave to user to decide later

    ┌───────────────────────────────────────────────────────────┐
    │ 4. SPLIT & STRATIFY                                       │
    └───────────────────────────────────────────────────────────┘
15. choose_split ← if task==classification then “stratified” else “random”
16. Ask user for stratify_by (target or group) if applicable
17. (train, val, test) ← train_test_split(df, strategy=choose_split, ratios=(0.7,0.15,0.15), seed=RANDOM_STATE)
18. store splits metadata (indices, seed)

    ┌───────────────────────────────────────────────────────────┐
    │ 5. FEATURE ENGINEERING (AUTO + PLUGINS)                  │
    └───────────────────────────────────────────────────────────┘
19. If user_opt_in(“Advanced FE?”):
       run transformations:
         • regex_split on object cols
         • date/time extraction
         • group‑by aggregations on user‑specified keys
         • custom user functions
    Else:
       skip advanced FE

    ┌───────────────────────────────────────────────────────────┐
    │ 6. FEATURE SELECTION (ENSEMBLE FILTERS)                  │
    └───────────────────────────────────────────────────────────┘
20. initial_feats ← all processed features
21. Apply filters:
       a. drop near‑zero variance (<1e‑5)
       b. drop high missingness (>50%)
       c. drop high collinearity (|corr|>0.9)
22. rank_feats:
       • mutual_info / F‑test
       • tree_importance (RandomForest)
       • L1‑based (Lasso/Logistic)
23. optional_user_cut ← prompt_user(“Max #features?”)
24. final_feats ← select top N by combined rank + user_cut
25. persist feature list + pipeline transform parameters

    ┌───────────────────────────────────────────────────────────┐
    │ 7. MODEL SEARCH & HYPERPARAMETER OPTIMIZATION           │
    └───────────────────────────────────────────────────────────┘
26. candidate_models ← if task==classification then [LogisticRegression, RandomForest, XGBoost] else [LinearRegression, RandomForest, XGBoostRegressor]
27. define_search_space(candidate_models):
       • LR: C ∈ [1e‑3,1e3]
       • RF: n_estimators ∈ [100,1000], max_depth ∈ [3,None]
       • XGB: learning_rate ∈ [0.01,0.3], max_depth ∈ [3,10]
28. Use Optuna (or Hyperopt) to run K‑fold CV (k=5) over search space:
       • objective: AUC (classification) or RMSE (regression)
       • early stopping heuristics
29. collect best_trial for each model; record metrics

    ┌───────────────────────────────────────────────────────────┐
    │ 8. MODEL ENSEMBLING & SELECTION                          │
    └───────────────────────────────────────────────────────────┘
30. top_models ← select top K (default K=3) by validation score
31. If K>1:
       a. simple average / majority voting ensemble
       b. stacking: train meta‑learner on out‑of‑fold predictions
32. evaluate ensemble on val set; compare vs best single model
33. chosen_model ← pick whichever has higher score

    ┌───────────────────────────────────────────────────────────┐
    │ 9. FINAL EVALUATION & ROBUSTNESS                         │
    └───────────────────────────────────────────────────────────┘
34. test_metrics ← evaluate chosen_model on hold‑out test set
35. run robustness checks:
       • time‑split evaluation
       • subgroup (e.g. demographic) performance
       • adversarial/noise stress tests
36. log all metrics + plots to MLflow

    ┌───────────────────────────────────────────────────────────┐
    │10. INTERPRETABILITY & EXPLANATIONS                       │
    └───────────────────────────────────────────────────────────┘
37. global_shap ← compute SHAP values on sample of train data
38. local_explain ← generate LIME/SHAP for user‑provided instances
39. PDPs ← top‑3 features partial dependence plots

    ┌───────────────────────────────────────────────────────────┐
    │11. DEPLOYMENT ARTIFACTS                                  │
    └───────────────────────────────────────────────────────────┘
40. bundle_pipeline ← serialize full pipeline (preprocessing + chosen_model) to ONNX or joblib
41. save environment spec (`poetry export -f requirements.txt`)
42. generate minimal Dockerfile:
       • base: python:3.11-slim
       • install dependencies & copy serialized pipeline
       • expose `/predict` via FastAPI stub
43. push artifacts to model registry or local `./deploy/`

    ┌───────────────────────────────────────────────────────────┐
    │12. SERVICE STARTUP & HEALTH                              │
    └───────────────────────────────────────────────────────────┘
44. spin up Docker container via `docker-compose up -d`
45. run health check on `/health` endpoint
46. notify user on success/failure (Discord webhook)

    ┌───────────────────────────────────────────────────────────┐
    │13. MONITORING & AUTOMATED RETRAIN                        │
    └───────────────────────────────────────────────────────────┘
47. schedule auto‑monitor job (daily/weekly)
48. collect new incoming data → compute drift (PSI/JSD)
49. if drift > threshold or performance ↓ threshold:
       trigger one_shot_auto_ml(new_df, task)
    END FUNCTION
```

---

## one_shot_auto_ml(df, task)

### 0. INITIAL DATA CAPTURE & VERSIONING

1. **Load & Snapshot**

   - Detect source (CSV/DB/API) → load to `df`
   - Write immutable Parquet snapshot → `raw/…`
   - Compute checksum (MD5) + save
   - Extract schema (column→dtype) → save JSON

2. **Versioning**

   - DVC add + commit snapshots & metadata
   - Record lineage: source URI ↔ checksum ↔ schema version

### 1. DATA VALIDATION & PROFILING (EDA)

3. **Schema Checks**

   - Ensure each col’s dtype matches expected; flag mismatches

4. **Missingness Profiling**

   - % missing per col/row, heatmap of patterns
   - Suggest “candidate for imputation” list

5. **Distribution Profiling**

   - Univariate: histograms, skew/kurtosis, normality tests
   - Bivariate: correlations (Pearson/Spearman), crosstabs

6. **Business‑Rule Checks**

   - ID formats, date ranges, FK integrity

7. **Bias & Balance**

   - Class imbalance stats, demographic parity heuristics

8. **Outlier Pre‑Scan**

   - Rough univariate flagging (IQR, z‑score)

9. **Report Generation**

   - Compile interactive HTML/JSON EDA report; log to MLflow

### 2. MISSING‑VALUE HANDLING (USER‑GUIDED)

10. **Detect** columns with missing values
11. **For each** such column:

    - Infer type (numeric/categorical/datetime)
    - Suggest strategies:

      - Numeric → {mean, median, KNN, model‑based}
      - Categorical → {mode, constant=`"missing"`, model‑based}

    - **Prompt user**: choose or override strategy

12. **Fit** chosen imputers on training portion
13. **Transform** entire `df` (train/val/test) with fitted imputers
14. **Log** per‑column strategy & fill values

### 3. ENCODING & SCALING (PARTIAL USER‑OVERRIDE)

15. **Categorical Encoding**

    - Identify categorical cols (dtype/object or low‑unique)
    - For each:

      - Compute cardinality
      - Default: one‑hot if ≤10 uniques else target encoding
      - **Prompt user** to accept or choose among {one‑hot, ordinal, target, embedding}

    - **Fit** encoders on train → **transform** all splits
    - Save mapping tables for audit

16. **Numeric Scaling**

    - For each numeric col:

      - Test distribution (normality/skew)
      - Default: StandardScaler for near‑Gaussian; PowerTransformer if skewed; RobustScaler if outlier‑heavy
      - (Optional user override)

    - **Fit** scalers on train → **transform** val/test
    - Log scaler parameters

### 4. OUTLIER DETECTION & HANDLING

17. **Univariate Detection** (IQR 1.5×, z‑score>3)
18. **Multivariate Detection** (IsolationForest, LOF)
19. **Ask user**: auto‑handle outliers?

    - Yes → cap or drop rows based on severity heuristic
    - No → leave as is but log counts & thresholds

20. **Apply** decided handling to all splits

### 5. SPLIT & STRATIFY

21. **Choose** strategy: stratified if classification else random
22. **Prompt user** for group/stratify key if needed
23. **Split** into train/val/test (70/15/15) with a fixed seed
24. **Verify** target distribution consistency
25. **Persist** indices & seed

### 6. ADVANCED FEATURE ENGINEERING (OPTIONAL)

26. **Prompt user**: enable advanced FE?

    - If yes, run:

      - Regex splits (alpha/digits/punct)
      - JSON/XML flattening
      - URL/IP/color/path parsing
      - Multi‑label string → binary flags
      - Date/time extractions
      - Group‑by aggs, rolling stats
      - Custom user functions via plugin API

    - If no, skip

### 7. FEATURE SELECTION (ENSEMBLE FILTERS)

27. **Filter‑based**

    - Drop near‑zero variance (<ε)
    - Drop high missingness (>50%)
    - Drop highly collinear pairs (|corr|>0.9)

28. **Statistical Ranking**

    - Mutual information / F‑test scores

29. **Model‑based Ranking**

    - Tree‑based importances (RF/GBDT)
    - L1 regularization (Lasso/Logistic)

30. **Wrapper Methods**

    - RFE, sequential forward/backward selection

31. **Voting Ensemble**

    - Combine “keep” votes; select features with ≥K votes

32. **Prompt user** for any manual max‑feature constraint
33. **Finalize** feature list & persist transform pipeline

### 8. DIMENSIONALITY REDUCTION (OPTIONAL)

34. **PCA**: choose n_components by explained variance threshold
35. **KernelPCA / Autoencoder** if non‑linear compaction desired
36. **Evaluate** retained variance; decide via threshold

### 9. MODEL DEVELOPMENT & HPO

37. **Candidate Models**

    - Classification → \[Logistic, RF, XGBoost, (optional) SVM/NN]
    - Regression → \[Linear, RF, XGBoost, (optional) GBM/NN]

38. **Define Hyperparameter Spaces** per model
39. **Optuna‑Driven HPO**

    - K‑fold CV (k=5)
    - Objective: AUC (class.) or RMSE (reg.)
    - Early‑stopping heuristics

40. **Log** best trials & metrics

### 10. MODEL ENSEMBLING & SELECTION

41. **Select top K** models by validation score (default K=3)
42. **Ensemble Methods**

    - Simple averaging/voting
    - Stacking meta‑learner on OOF predictions

43. **Evaluate** ensembles on val set vs best single model
44. **Choose** whichever yields higher metric

### 11. FINAL EVALUATION & ROBUSTNESS

45. **Assess** chosen model on hold‑out test set → final test metrics
46. **Robustness Checks**

    - Time‑series splits
    - Subgroup (e.g. by category) performance
    - Adversarial/noise stress tests

47. **Log** all results to MLflow

### 12. INTERPRETABILITY & EXPLANATIONS

48. **Global**: SHAP summary plots, bar charts
49. **Local**: LIME/SHAP explanations for sample instances
50. **PDP/ICE**: partial dependence & ICE plots for top features
51. **Compile** interpretability report

### 13. DEPLOYMENT & PACKAGING

52. **Bundle** full pipeline (preprocessing + model) → ONNX/joblib
53. **Export** environment spec → `requirements.txt` / `conda.yaml`
54. **Generate** minimal Dockerfile & `docker-compose.yml` stub for inference:

    - Base image → install deps → copy bundle → serve via FastAPI

55. **Push** artifacts to local registry or `./deploy/` folder

### 14. SERVICE STARTUP & HEALTHCHECK

56. **Launch** Docker Compose (containers for API + optional DB)
57. **Health Check**: poll `/health`; report success/failure
58. **Notify** via Discord webhook on startup result

### 15. MONITORING & AUTOMATED RETRAIN

59. **Schedule** periodic or drift‑based checks
60. **Compute** data drift (PSI/JSD) and performance decay
61. **If** drift or decay exceeds thresholds → call `one_shot_auto_ml(new_df, task)`
62. **Archive** previous models & logs in MLflow registry

### 16. GOVERNANCE & AUDITABILITY

63. **Log** pipeline metadata: commit SHA, DVC version, parameters
64. **Lineage** tracking: feature provenance, model versions
65. **Access Control** hooks (defer to later Auth integration)
66. **Encryption**: ensure sensitive logs/metas are encrypted at rest

---

### Interaction & Defaults

- **User prompts** only at ambiguous points (imputation, encoding, outlier handling, advanced FE, feature limits).
- **Defaults** follow best practices (mean/median impute, one‑hot/target encode, Standard/Power/Robust scaling, IQR & IsolationForest), requiring no user input.
- **All metadata**, intermediate artifacts, and models logged to MLflow.
- **Entire pipeline** runs in one shot under ZenML’s DockerSequentialRunner, with host volumes for data & models.

This **intelligent AutoML algorithm** ensures maximum automation—with just-in-time user guidance—to deliver a production‑ready model and deployment artifacts from a single DataFrame input.
