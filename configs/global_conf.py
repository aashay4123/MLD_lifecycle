# These global parameters should be the same across all workflow stages.
import optuna
CSV_PATH = "Data/housing.csv"
DATASET_TARGET_COLUMN_NAME = "label"

RAW_PARQUET_PATH = "Data/raw"
SPLIT_PARQUET_PATH = "Data/split"


INJESTION_LOGS_PATH = "reports/logs/ingest.log"
HEALTH_CHECK_REPORT_PATH = "reports/health_report"
EPDA_REPORT_PATH = "reports/epda"
BASELINE_REPORT_PATH = "reports/baseline_report"
PREPROCESSOR_REPORT_PATH = "reports/preprocessor_report"


DR_REPORT_PATH = "reports/Dimentionality_Reduction_report"

MODEL_ARTIFACTS_PATH = "reports/Models"

PIPELINE_LOGS_PATH = "reports/logs/pipeline_monitor.log"
MLFLOW_REPORT_PATH = "reports/mlflow/"
OPTUNA_REPORT_PATH = "reports/optuna/"

# Model training and evaluation parameters
RANDOM_STATE = 23
TRAIN_TEST_SPLIT = 0.2
MIN_TRAIN_ACCURACY = 0.9
MIN_TEST_ACCURACY = 0.9
MAX_SERVE_TRAIN_ACCURACY_DIFF = 0.1
MAX_SERVE_TEST_ACCURACY_DIFF = 0.05
WARNINGS_AS_ERRORS = False
MODEL_NAME = "gitflow_model"


# global_conf.py additions

TRAIN_PARQUET_PATH = "artifacts/final/production/train.parquet"
VAL_PARQUET_PATH = "artifacts/final/production/val.parquet"
TEST_PARQUET_PATH = "artifacts/final/production/test.parquet"
FINAL_MODEL_PATH = "artifacts/final/final_model.joblib"
MLFLOW_ENSEMBLE_ARTIFACT_PATH = "final_ensemble_model"
ENSEMBLE_TOP_N = 3
DEFAULT_SCORING = "roc_auc"

# optuna_hpo/config.py

DEFAULT_N_TRIALS = 50
DEFAULT_CV_FOLDS = 5
DEFAULT_METRIC_CLASSIFICATION = "roc_auc"
DEFAULT_METRIC_REGRESSION = "r2"


DEFAULT_SAMPLER = optuna.samplers.TPESampler(seed=42)
AVAILABLE_PRUNERS = {
    "median": optuna.pruners.MedianPruner(),
    "successive_halving": optuna.pruners.SuccessiveHalvingPruner(),
    "hyperband": optuna.pruners.HyperbandPruner(),
}


def choose_pruner(X):
    n_samples, n_features = X.shape
    if n_samples <= 5000:
        return AVAILABLE_PRUNERS["median"]
    elif n_samples <= 100000:
        return AVAILABLE_PRUNERS["successive_halving"]
    else:
        return AVAILABLE_PRUNERS["hyperband"]


# Expanded and more realistic search spaces:
SEARCH_SPACES_CLASSIFICATION = {
    "RandomForestClassifier": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "max_depth": lambda t: t.suggest_int("max_depth", 2, 40),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "min_weight_fraction_leaf": lambda t: t.suggest_float("min_weight_fraction_leaf", 0.0, 0.5),
        "max_features": lambda t: t.suggest_categorical("max_features", ["sqrt", "log2", None]),
        "min_impurity_decrease": lambda t: t.suggest_float("min_impurity_decrease", 0.0, 1.0),
    },
    "GradientBoostingClassifier": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "max_depth": lambda t: t.suggest_int("max_depth", 2, 20),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": lambda t: t.suggest_categorical("max_features", ["sqrt", "log2", None]),
    },
    "ExtraTreesClassifier": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "max_depth": lambda t: t.suggest_int("max_depth", 2, 40),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": lambda t: t.suggest_categorical("max_features", ["sqrt", "log2", None]),
    },
    "XGBClassifier": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 20),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": lambda t: t.suggest_float("gamma", 0, 10),
        "reg_alpha": lambda t: t.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 0, 10),
    },
    "LGBMClassifier": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 20, 300),
        "max_depth": lambda t: t.suggest_int("max_depth", -1, 40),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": lambda t: t.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": lambda t: t.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 0, 10),
    },
    "CatBoostClassifier": {
        "iterations": lambda t: t.suggest_int("iterations", 50, 400, step=10),
        "depth": lambda t: t.suggest_int("depth", 3, 12),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "l2_leaf_reg": lambda t: t.suggest_float("l2_leaf_reg", 1, 10),
        "bagging_temperature": lambda t: t.suggest_float("bagging_temperature", 0, 1),
    },
    "LogisticRegression": {
        "C": lambda t: t.suggest_float("C", 0.001, 100.0, log=True),
        "penalty": lambda t: t.suggest_categorical("penalty", ["l1", "l2"]),
        "solver": lambda t: t.suggest_categorical("solver", ["liblinear", "saga"]),
    },
    "SVC": {
        "C": lambda t: t.suggest_float("C", 0.01, 100.0, log=True),
        "kernel": lambda t: t.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "gamma": lambda t: t.suggest_categorical("gamma", ["scale", "auto"]),
    },
}

SEARCH_SPACES_REGRESSION = {
    "RandomForestRegressor": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "max_depth": lambda t: t.suggest_int("max_depth", 2, 40),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": lambda t: t.suggest_categorical("max_features", ["sqrt", "log2", None]),
    },
    "GradientBoostingRegressor": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "max_depth": lambda t: t.suggest_int("max_depth", 2, 20),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": lambda t: t.suggest_categorical("max_features", ["sqrt", "log2", None]),
    },
    "ExtraTreesRegressor": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "max_depth": lambda t: t.suggest_int("max_depth", 2, 40),
        "min_samples_split": lambda t: t.suggest_int("min_samples_split", 2, 30),
        "min_samples_leaf": lambda t: t.suggest_int("min_samples_leaf", 1, 20),
        "max_features": lambda t: t.suggest_categorical("max_features", ["sqrt", "log2", None]),
    },
    "Ridge": {
        "alpha": lambda t: t.suggest_float("alpha", 0.0001, 1000.0, log=True),
    },
    "Lasso": {
        "alpha": lambda t: t.suggest_float("alpha", 0.0001, 1000.0, log=True),
    },
    "XGBRegressor": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 20),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.4, 1.0),
        "gamma": lambda t: t.suggest_float("gamma", 0, 10),
        "reg_alpha": lambda t: t.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 0, 10),
    },
    "LGBMRegressor": {
        "n_estimators": lambda t: t.suggest_int("n_estimators", 50, 400, step=10),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 20, 300),
        "max_depth": lambda t: t.suggest_int("max_depth", -1, 40),
        "learning_rate": lambda t: t.suggest_float("learning_rate", 0.001, 0.5, log=True),
        "subsample": lambda t: t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": lambda t: t.suggest_float("colsample_bytree", 0.4, 1.0),
        "min_child_samples": lambda t: t.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": lambda t: t.suggest_float("reg_alpha", 0, 10),
        "reg_lambda": lambda t: t.suggest_float("reg_lambda", 0, 10),
    },
    "LinearRegression": {},  # No hyperparameters to tune but included for completeness
}

SEARCH_SPACES_UNSUPERVISED = {
    "KMeans": {
        "n_clusters": lambda t: t.suggest_int("n_clusters", 2, 50),
        "init": lambda t: t.suggest_categorical("init", ["k-means++", "random"]),
        "max_iter": lambda t: t.suggest_int("max_iter", 100, 1000, step=50),
    },
    "DBSCAN": {
        "eps": lambda t: t.suggest_float("eps", 0.1, 10.0),
        "min_samples": lambda t: t.suggest_int("min_samples", 1, 50),
    },
    "GaussianMixture": {
        "n_components": lambda t: t.suggest_int("n_components", 1, 30),
        "covariance_type": lambda t: t.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"]),
    },
}
SEARCH_SPACES = {**SEARCH_SPACES_CLASSIFICATION,
                 ** SEARCH_SPACES_REGRESSION, **SEARCH_SPACES_UNSUPERVISED}
