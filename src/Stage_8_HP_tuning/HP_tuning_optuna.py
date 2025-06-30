from zenml.steps import step, Output
import pandas as pd
import numpy as np
import optuna
import joblib
import logging
import warnings
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from typing import Any, Dict, List

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@step(enable_cache=False)
def hp_tuning_optuna_probabilistic_approach(
    train: pd.DataFrame, val: pd.DataFrame
) -> Output(final_model=Any, best_params=dict, val_metrics=Dict[str, float]):

    def detect_categorical(df: pd.DataFrame) -> List[str]:
        return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_val = val.drop(columns=["target"])
    y_val = val["target"]

    categorical_columns = detect_categorical(X_train)
    logger.info(f"Detected categorical columns: {categorical_columns}")

    def objective(trial):
        model_type = trial.suggest_categorical(
            "model",
            ["xgb", "lgb", "cat", "rf", "et", "gb", "ada", "lr", "svc", "knn", "dt"],
        )

        if model_type == "xgb":
            model = XGBClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 3, 15),
                learning_rate=trial.suggest_float("lr", 0.01, 0.3),
                use_label_encoder=False,
                eval_metric="logloss",
            )
        elif model_type == "lgb":
            model = LGBMClassifier(
                num_leaves=trial.suggest_int("num_leaves", 20, 100),
                learning_rate=trial.suggest_float("lr", 0.01, 0.3),
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
            )
        elif model_type == "cat":
            model = CatBoostClassifier(
                iterations=trial.suggest_int("iterations", 100, 300),
                depth=trial.suggest_int("depth", 3, 10),
                learning_rate=trial.suggest_float("lr", 0.01, 0.3),
                verbose=0,
            )
        elif model_type == "rf":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                random_state=42,
            )
        elif model_type == "et":
            model = ExtraTreesClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                random_state=42,
            )
        elif model_type == "gb":
            model = GradientBoostingClassifier(
                learning_rate=trial.suggest_float("lr", 0.01, 0.3),
                n_estimators=trial.suggest_int("n_estimators", 50, 200),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                random_state=42,
            )
        elif model_type == "ada":
            model = AdaBoostClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 150),
                learning_rate=trial.suggest_float("lr", 0.01, 1.0),
                random_state=42,
            )
        elif model_type == "lr":
            model = LogisticRegression(
                C=trial.suggest_float("C", 0.01, 10), max_iter=1000
            )
        elif model_type == "svc":
            model = SVC(
                C=trial.suggest_float("C", 0.1, 10),
                kernel=trial.suggest_categorical("kernel", ["linear", "rbf"]),
                probability=True,
            )
        elif model_type == "dt":
            model = DecisionTreeClassifier(
                max_depth=trial.suggest_int("max_depth", 3, 15)
            )
        elif model_type == "knn":
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", 3, 15)
            )

        if model_type in ["cat"]:
            model.fit(X_train, y_train, cat_features=categorical_columns)
        else:
            model.fit(X_train, y_train)

        return model.score(X_val, y_val)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=60)

    top_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:5]
    model_lookup = {
        "xgb": XGBClassifier,
        "lgb": LGBMClassifier,
        "cat": CatBoostClassifier,
        "rf": RandomForestClassifier,
        "et": ExtraTreesClassifier,
        "gb": GradientBoostingClassifier,
        "ada": AdaBoostClassifier,
        "lr": LogisticRegression,
        "svc": SVC,
        "dt": DecisionTreeClassifier,
        "knn": KNeighborsClassifier,
    }

    trained_models = []
    for t in top_trials:
        p = t.params.copy()
        model_type = p.pop("model")
        model = model_lookup[model_type](**p)
        if model_type == "cat":
            model.fit(X_train, y_train, cat_features=categorical_columns)
        else:
            model.fit(X_train, y_train)
        trained_models.append((model_type, model))

    # === Ensemble Study ===
    def ensemble_objective(trial):
        method = trial.suggest_categorical("ensemble", ["soft", "hard", "stacking"])
        estimators = trained_models[:3]

        if method == "soft":
            model = VotingClassifier(estimators=estimators, voting="soft")
        elif method == "hard":
            model = VotingClassifier(estimators=estimators, voting="hard")
        else:
            meta = trial.suggest_categorical("meta", ["lr", "ridge"])
            meta_model = LogisticRegression() if meta == "lr" else RidgeClassifier()
            model = StackingClassifier(
                estimators=estimators, final_estimator=meta_model, passthrough=True
            )

        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return accuracy_score(y_val, preds)

    ensemble_study = optuna.create_study(direction="maximize")
    ensemble_study.optimize(ensemble_objective, n_trials=20)

    best_ens_method = ensemble_study.best_params["ensemble"]
    if best_ens_method == "soft":
        final_model = VotingClassifier(estimators=trained_models[:3], voting="soft")
    elif best_ens_method == "hard":
        final_model = VotingClassifier(estimators=trained_models[:3], voting="hard")
    else:
        meta = ensemble_study.best_params["meta"]
        meta_model = LogisticRegression() if meta == "lr" else RidgeClassifier()
        final_model = StackingClassifier(
            estimators=trained_models[:3], final_estimator=meta_model, passthrough=True
        )

    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    y_pred = final_model.predict(X_val)
    y_prob = (
        final_model.predict_proba(X_val)
        if hasattr(final_model, "predict_proba")
        else None
    )
    metrics = {
        "accuracy": accuracy_score(y_val, y_pred),
        "f1": f1_score(y_val, y_pred, average="weighted"),
        "log_loss": log_loss(y_val, y_prob) if y_prob is not None else np.nan,
    }

    # Save
    joblib.dump(final_model, "artifacts/final_model.joblib")

    try:
        from mlflow import log_metric, log_params, log_artifacts

        log_params(study.best_params)
        for k, v in metrics.items():
            log_metric(k, v)
        log_artifacts("artifacts")
    except Exception:
        logger.warning("MLflow logging failed")

    return final_model, study.best_params, metrics
