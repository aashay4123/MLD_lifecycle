import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def evaluate_ensemble(model, X_val, y_val, metric="accuracy") -> float:
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
    if metric == "accuracy":
        return accuracy_score(y_val, preds)
    elif metric == "f1":
        return f1_score(y_val, preds, average="weighted")
    elif metric == "log_loss" and proba is not None:
        return -log_loss(y_val, proba)
    return 0.0


def ensemble_selector_optuna(
    train: pd.DataFrame,
    val: pd.DataFrame,
    model_candidates: List[Tuple[str, ClassifierMixin]],
    top_k: int = 5,
    metric: str = "accuracy",
) -> Tuple[Any, float]:
    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_val = val.drop(columns=["target"])
    y_val = val["target"]

    # Sort and select top-k models
    top_models = model_candidates[:top_k]
    estimators = [(name, model) for name, model in top_models]

    def objective(trial):
        method = trial.suggest_categorical(
            "method",
            ["soft", "hard", "weighted", "stack_logreg", "stack_ridge", "residual"],
        )

        if method == "soft":
            ensemble = VotingClassifier(estimators=estimators, voting="soft")
        elif method == "hard":
            ensemble = VotingClassifier(estimators=estimators, voting="hard")
        elif method == "weighted":
            weights = [
                trial.suggest_float(f"w_{name}", 0.5, 2.0) for name, _ in estimators
            ]
            ensemble = VotingClassifier(
                estimators=estimators, voting="soft", weights=weights
            )
        elif method == "stack_logreg":
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(max_iter=1000),
                passthrough=True,
            )
        elif method == "stack_ridge":
            ensemble = StackingClassifier(
                estimators=estimators,
                final_estimator=RidgeClassifier(),
                passthrough=True,
            )
        elif method == "residual":
            # Residual Blending (greedy)
            base = estimators[0][1]
            base.fit(X_train, y_train)
            residual = y_train - base.predict(X_train)
            second_model = LogisticRegression().fit(X_train, residual)

            class ResidualBlend:
                def predict(self, X):
                    return (
                        np.clip(base.predict(X) + second_model.predict(X), 0, 1)
                        .round()
                        .astype(int)
                    )

                def predict_proba(self, X):
                    preds = self.predict(X)
                    return np.column_stack((1 - preds, preds))

            ensemble = ResidualBlend()

        ensemble.fit(X_train, y_train)
        score = evaluate_ensemble(ensemble, X_val, y_val, metric)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_method = study.best_params["method"]
    best_score = study.best_value

    # Re-train best ensemble
    if best_method == "soft":
        final = VotingClassifier(estimators=estimators, voting="soft")
    elif best_method == "hard":
        final = VotingClassifier(estimators=estimators, voting="hard")
    elif best_method == "weighted":
        weights = [study.best_params[f"w_{name}"] for name, _ in estimators]
        final = VotingClassifier(estimators=estimators, voting="soft", weights=weights)
    elif best_method == "stack_logreg":
        final = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            passthrough=True,
        )
    elif best_method == "stack_ridge":
        final = StackingClassifier(
            estimators=estimators, final_estimator=RidgeClassifier(), passthrough=True
        )
    elif best_method == "residual":
        base = estimators[0][1]
        base.fit(X_train, y_train)
        residual = y_train - base.predict(X_train)
        second_model = LogisticRegression().fit(X_train, residual)

        class ResidualBlend:
            def predict(self, X):
                return (
                    np.clip(base.predict(X) + second_model.predict(X), 0, 1)
                    .round()
                    .astype(int)
                )

            def predict_proba(self, X):
                preds = self.predict(X)
                return np.column_stack((1 - preds, preds))

        final = ResidualBlend()

    final.fit(X_train, y_train)
    return final, best_score
