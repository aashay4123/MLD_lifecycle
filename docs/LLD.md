## 1. Core Modules & Classes

### 1.1 `Orchestrator`

- `run(df: pd.DataFrame, target: Optional[str])`
  – Kicks off the pipeline, sequences steps, handles errors.
- `load_config() → Config`
  – Reads global settings & thresholds.
- `notify_error(step_name: str, error: Exception)`
  – Sends alerts (Discord, logs).

### 1.2 `PipelineData`

- Holds:

  - `df: DataFrame`
  - `target: Optional[str]`
  - `task_type: TaskType`
  - `metadata: dict`
  - `artifacts: dict`

### 1.3 `PipelineStep` (abstract base)

- `name: str`
- `execute(data: PipelineData) → PipelineData`

### 1.4 Concrete Steps (one class per stage)

- **IngestionStep**
- **ValidationStep**
- **ImputationStep**
- **EncodingStep**
- **ScalingStep**
- **OutlierStep**
- **SplittingStep**
- **FeatureEngineeringStep**
- **FeatureSelectionStep**
- **HpoStep**
- **EnsemblingStep**
- **EvaluationStep**
- **ExplainabilityStep**
- **DeploymentStep**
- **MonitoringStep**

Each just implements `execute()` using utilities.

---

## 2. Utility Modules

### 2.1 `utils/problem.py`

- `detect_problem_type(df, target) → TaskType`
  • Inspect `target` dtype & cardinality to pick classification/regression/clustering.

### 2.2 `utils/data.py`

- `load_data(source: str) → DataFrame`
- `snapshot_schema(df, path: str)`
- `compute_checksum(path: str) → str`
- `split_data(df, target, ratios, seed) → (train, val, test)`

### 2.3 `utils/impute.py`

- `suggest_imputation_strategies(df) → Dict[col, List[str]]`
- `fit_imputers(train_df, strategies) → Dict[col, Imputer]`
- `apply_imputers(df, imputers) → DataFrame`

### 2.4 `utils/encode.py`

- `suggest_encoding(df, thresh) → Dict[col, str]`
- `fit_encoders(train_df, strategies) → Dict[col, Encoder]`
- `apply_encoders(df, encoders) → DataFrame`

### 2.5 `utils/scale.py`

- `suggest_scalers(df) → Dict[col, str]`
- `fit_scalers(train_df, strategies) → Dict[col, Scaler]`
- `apply_scalers(df, scalers) → DataFrame`

### 2.6 `utils/outlier.py`

- `detect_univariate(df) → Dict[col, (low, high)]`
- `detect_multivariate(df) → Series[bool]`
- `handle_outliers(df, uni_thresh, multi_mask, strategy) → DataFrame`

### 2.7 `utils/fe.py`

- `basic_fe(df) → DataFrame`
- `advanced_fe(df, plugins: List[Callable]) → DataFrame`

### 2.8 `utils/select.py`

- `filter_features(df) → DataFrame`
- `rank_features(X, y) → DataFrame(feat,score)`
- `vote_features(ranks, k) → List[str]`

### 2.9 `utils/hpo.py`

- `define_search_space(models) → Dict[model, space]`
- `run_hpo(X, y, spaces, task, trials, folds) → Dict[model, BestTrial]`

### 2.10 `utils/ensemble.py`

- `build_voting_ensemble(models, X, y, task) → EnsembleModel`
- `build_stacking_ensemble(models, X, y, task) → EnsembleModel`

### 2.11 `utils/eval.py`

- `compute_metrics(y_true, y_pred, task) → Dict[str, float]`
- `run_robustness(model, X_splits, y_splits) → Dict[str,Any]`

### 2.12 **MLflow Utilities** (`utils/mlflow.py`)

- `init_mlflow(uri, experiment_name)`
- `log_params(params: dict)`
- `log_metrics(metrics: dict)`
- `register_model(run_id, model_uri, name)`
- `load_model(name, stage) → PyFuncModel`

### 2.13 **Serving Utilities** (`utils/serve.py`)

- `create_fastapi_app(model_name, stage) → FastAPI`
- `predict_endpoint(request: PredictRequest) → PredictResponse`

---
