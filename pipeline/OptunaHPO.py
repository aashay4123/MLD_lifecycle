from zenml.steps import step
from zenml.pipelines import pipeline
from zenml.integrations.mlflow.steps import enable_mlflow
import pandas as pd
import mlflow
from configs import global_conf
from src.Stage_8_HP_tuning.tuner import OptunaHyperTuner
from src.Stage_8_HP_tuning.ensembler import OptunaEnsembler
from src.Stage_8_HP_tuning.reporter import OptunaReporter
from src.utils import monitor

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans


@enable_mlflow
@step
@monitor(name="load_data")
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(global_conf.TRAIN_PARQUET_PATH)
    val = pd.read_parquet(global_conf.VAL_PARQUET_PATH)
    return train, val


@enable_mlflow
@step
@monitor(name="baseline_model")
def baseline_step(train: pd.DataFrame, val: pd.DataFrame) -> None:
    X_train, y_train = train.drop("target", axis=1), train["target"]

    if y_train is None:
        task = "unsupervised"
        baseline_models = {
            "kmeans": KMeans(),
        }
    elif y_train.dtype == "O" or y_train.nunique() <= 20:
        task = "classification"
        baseline_models = {
            "logreg": LogisticRegression(max_iter=1000),
            "dtree": DecisionTreeClassifier(),
            "knn": KNeighborsClassifier(),
        }
    else:
        task = "regression"
        baseline_models = {
            "linreg": LinearRegression(),
            "dtree": DecisionTreeRegressor(),
            "knn": KNeighborsRegressor(),
        }

    print(f"🚦 Baseline task: {task} | Models: {list(baseline_models.keys())}")

    baseline_tuner = OptunaHyperTuner(
        X_train, y_train,
        n_trials=3,
        mlflow_run=mlflow.active_run()
    )
    baseline_tuner.models = baseline_models
    baseline_tuner.tune()


@enable_mlflow
@step
@monitor(name="optuna_hpo_step")
def optuna_hpo_step(train: pd.DataFrame, val: pd.DataFrame) -> dict:
    X_train, y_train = train.drop("target", axis=1), train["target"]
    tuner = OptunaHyperTuner(X_train, y_train, mlflow_run=mlflow.active_run())
    tuner.tune()
    return tuner.get_best_models()


@enable_mlflow
@step
@monitor(name="optuna_ensemble_step")
def ensemble_step(train: pd.DataFrame, val: pd.DataFrame, best_models: dict) -> object:
    X_train, y_train = train.drop("target", axis=1), train["target"]
    X_val, y_val = val.drop("target", axis=1), val["target"]
    X_full = pd.concat([X_train, X_val])
    y_full = pd.concat([y_train, y_val])

    task = next(iter(best_models.values()))[
        1].best_trial.user_attrs.get("task", "classification")
    scoring = global_conf.DEFAULT_SCORING

    ensembler = OptunaEnsembler(
        best_models, X_full.to_numpy(), y_full.to_numpy(),
        task=task, scoring=scoring, cv_folds=global_conf.DEFAULT_CV_FOLDS,
        top_n=global_conf.ENSEMBLE_TOP_N, mlflow_run=mlflow.active_run()
    )
    ensembler.build_ensemble()
    final_ensemble = ensembler.get_ensemble()
    mlflow.sklearn.log_model(
        final_ensemble, artifact_path=global_conf.MLFLOW_ENSEMBLE_ARTIFACT_PATH)
    return ensembler.get_ensemble_info()


@enable_mlflow
@step
@monitor(name="generate_reports")
def reporting_step(best_models: dict, ensemble_info: dict) -> None:
    reporter = OptunaReporter(best_models, ensemble=ensemble_info.get(
        "ensemble_details"), mlflow_run=mlflow.active_run())
    reporter.generate_reports()


@enable_mlflow
@pipeline(enable_cache=True)
def optuna_hpo_pipeline(
    load_data, baseline_step, optuna_hpo_step, ensemble_step, reporting_step
):
    train, val = load_data()
    baseline_step(train, val)
    best_models = optuna_hpo_step(train, val)
    ensemble_info = ensemble_step(train, val, best_models)
    reporting_step(best_models, ensemble_info)


def run_pipeline():
    optuna_hpo_pipeline(
        load_data=load_data(),
        baseline_step=baseline_step(),
        optuna_hpo_step=optuna_hpo_step(),
        ensemble_step=ensemble_step(),
        reporting_step=reporting_step(),
    ).run()
