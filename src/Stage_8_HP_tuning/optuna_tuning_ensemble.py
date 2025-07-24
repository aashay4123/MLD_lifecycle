import optuna
import numpy as np
from sklearn.ensemble import (
    VotingClassifier,
    VotingRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import get_scorer
from sklearn.utils.multiclass import type_of_target
from configs import global_conf
from src.utils.utils import get_cv, get_metric
import mlflow


class OptunaEnsembler:
    def __init__(
        self,
        best_models,
        X,
        y,
        task,
        scoring,
        cv_folds=5,
        top_n=3,
        optimize_weights=True,
        n_trials=30,
        mlflow_run=None,
        X_val=None,
        y_val=None,
    ):
        self.best_models = best_models
        self.X, self.y = X, y
        self.X_val, self.y_val = X_val, y_val
        self.task = task
        self.scoring = scoring
        self.cv_folds = cv_folds
        self.top_n = top_n
        self.optimize_weights = optimize_weights
        self.n_trials = n_trials
        self.mlflow_run = mlflow_run
        self.ensemble = None
        self.ensemble_study = None
        self.ensemble_score = None
        self.fallback_used = False

    def build_ensemble(self):
        selected = self._select_top_models()
        estimators = [(name, model) for name, model, _ in selected]

        self._refit_models(estimators)

        if not self.optimize_weights:
            self.ensemble = self._default_ensemble(estimators)
            self._finalize_ensemble(selected)
            return

        def objective(trial):
            method = trial.suggest_categorical("method", self._available_methods())
            try:
                ens = self._build_candidate_ensemble(estimators, trial, method)
                if ens is None:
                    return -np.inf
                return self._cross_val_score(ens)
            except Exception as e:
                print(f"⚠️ Ensemble candidate {method} failed: {e}")
                return -np.inf

        reverse = (
            "roc_auc" in str(self.scoring).lower()
            or "accuracy" in str(self.scoring).lower()
        )
        self.ensemble_study = optuna.create_study(
            direction="maximize" if reverse else "minimize"
        )
        self.ensemble_study.optimize(
            objective, n_trials=self.n_trials, show_progress_bar=True
        )

        best_method = self.ensemble_study.best_params["method"]
        self.ensemble = self._build_candidate_ensemble(
            estimators, self.ensemble_study, best_method, final=True
        )
        self.ensemble.fit(self.X, self.y)
        self._finalize_ensemble(selected, best_method)

    def _select_top_models(self):
        model_scores = [
            (name, model, getattr(study, "best_value", None) or 0.0)
            for name, (model, study) in self.best_models.items()
        ]
        reverse = (
            "roc_auc" in str(self.scoring).lower()
            or "accuracy" in str(self.scoring).lower()
        )
        model_scores.sort(key=lambda x: x[2], reverse=reverse)
        selected = model_scores[: self.top_n]
        print(f"🔗 Selected top-{self.top_n} models: {[e[0] for e in selected]}")
        return selected

    def _refit_models(self, estimators):
        for name, model in estimators:
            print(f"🔎 Re-training {name} on full dataset...")
            model.fit(self.X, self.y)
        for name, model in estimators:
            print(f"✅ {name} classes_: {getattr(model, 'classes_', 'N/A')}")

    def _available_methods(self):
        if self.task == "classification":
            n_classes = len(np.unique(self.y))
            methods = ["soft", "hard", "weighted", "stack_logreg", "stack_ridge"]
            if n_classes > 2:
                # residuals meaningful for ordered multiclass
                methods.append("residual")
            return methods
        elif self.task == "regression":
            return ["soft", "weighted", "stack_ridge", "residual"]
        return []

    def _default_ensemble(self, estimators):
        return (
            VotingClassifier(estimators=estimators, voting="soft")
            if self.task == "classification"
            else VotingRegressor(estimators=estimators)
        )

    def _cross_val_score(self, ensemble):
        cv = get_cv(self.cv_folds, self.task)
        scorer = get_scorer(self.scoring)
        scores = []

        for train_idx, val_idx in cv.split(self.X, self.y):
            ensemble.fit(self.X[train_idx], self.y[train_idx])
            y_val = self.y[val_idx]
            pred = self._safe_predict(ensemble, self.X[val_idx], y_val)
            score = scorer._score_func(y_val, pred)
            scores.append(score)

        return np.mean(scores)

    def _safe_predict(self, model, X, y_val):
        if self.task == "classification" and "roc_auc" in str(self.scoring).lower():
            try:
                proba = model.predict_proba(X)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    return proba[:, 1]
                return proba
            except AttributeError:
                print("⚠️ Ensemble lacks predict_proba; using predict fallback.")
        return model.predict(X)

    def _build_candidate_ensemble(self, estimators, trial, method, final=False):
        if self.task == "classification":
            return self._classification_ensembles(estimators, trial, method, final)
        elif self.task == "regression":
            return self._regression_ensembles(estimators, trial, method, final)
        raise ValueError(f"[ENSEMBLE] Unsupported task: {self.task}")

    def _classification_ensembles(self, estimators, trial, method, final):
        if method in ("soft", "hard"):
            return VotingClassifier(estimators=estimators, voting=method)
        elif method == "weighted":
            weights = self._get_weights(estimators, trial, final)
            return VotingClassifier(
                estimators=estimators, voting="soft", weights=weights
            )
        elif method.startswith("stack"):
            final_est = (
                LogisticRegression(max_iter=1000)
                if method == "stack_logreg"
                else RidgeClassifier()
            )
            return StackingClassifier(
                estimators=estimators, final_estimator=final_est, passthrough=True
            )
        elif method == "residual":
            return self._build_residual_classifier(estimators)
        return None

    def _regression_ensembles(self, estimators, trial, method, final):
        if method == "soft":
            return VotingRegressor(estimators=estimators)
        elif method == "weighted":
            weights = self._get_weights(estimators, trial, final)
            return VotingRegressor(estimators=estimators, weights=weights)
        elif method == "stack_ridge":
            final_est = getattr(global_conf, "STACK_FINAL_ESTIMATOR", Ridge())
            return StackingRegressor(
                estimators=estimators, final_estimator=final_est, passthrough=True
            )
        elif method == "residual":
            return self._build_residual_regressor(estimators)
        return None

    def _build_residual_classifier(self, estimators):
        base = estimators[0][1]
        base.fit(self.X, self.y)
        residual = self.y - base.predict(self.X)
        if len(np.unique(residual)) < 2:
            print(
                "⚠️ Residuals have only one class — skipping residual ensemble candidate."
            )
            return None
        residual_model = LogisticRegression(max_iter=1000).fit(self.X, residual)

        class ResidualBlend(BaseEstimator, ClassifierMixin):
            def __init__(self, base_model, residual_model):
                self.base_model, self.residual_model = base_model, residual_model

            def fit(self, X, y):
                self.base_model.fit(X, y)
                residual = y - self.base_model.predict(X)
                self.residual_model.fit(X, residual)
                self.classes_ = np.unique(y)
                return self

            def predict(self, X):
                preds = (
                    np.clip(
                        self.base_model.predict(X) + self.residual_model.predict(X),
                        0,
                        len(self.classes_) - 1,
                    )
                    .round()
                    .astype(int)
                )
                return preds

            def predict_proba(self, X):
                if hasattr(self.base_model, "predict_proba"):
                    return self.base_model.predict_proba(X)
                return np.eye(len(self.classes_))[self.predict(X)]

        return ResidualBlend(base, residual_model)

    def _build_residual_regressor(self, estimators):
        base = estimators[0][1]
        base.fit(self.X, self.y)
        residual = self.y - base.predict(self.X)
        if np.allclose(residual, 0):
            print(
                "⚠️ Residuals are effectively zero — skipping residual ensemble candidate."
            )
            return None
        residual_model = Ridge().fit(self.X, residual)

        class ResidualBlend(BaseEstimator, RegressorMixin):
            def __init__(self, base_model, residual_model):
                self.base_model, self.residual_model = base_model, residual_model

            def fit(self, X, y):
                self.base_model.fit(X, y)
                residual = y - self.base_model.predict(X)
                self.residual_model.fit(X, residual)
                return self

            def predict(self, X):
                return self.base_model.predict(X) + self.residual_model.predict(X)

        return ResidualBlend(base, residual_model)

    def _get_weights(self, estimators, trial, final):
        return [
            (
                trial.suggest_float(f"weight_{name}", 0.5, 2.0)
                if not final
                else trial.best_params[f"weight_{name}"]
            )
            for name, _ in estimators
        ]

    def _finalize_ensemble(self, selected_models, method="equal_weights"):
        if self.X_val is not None and self.y_val is not None:
            self.ensemble_score = self._evaluate_on_holdout()
        else:
            print("⚠️ No holdout set provided; evaluating ensemble on train data.")
            self.ensemble_score = self._evaluate(self.ensemble)
        self._check_fallback(selected_models)
        self._log_mlflow(
            {
                "ensemble_type": method,
                "ensemble_score": self.ensemble_score,
            }
        )
        print(f"✅ Final ensemble built with method: {method}")

    def _evaluate(self, ensemble):
        scorer = get_scorer(self.scoring)
        preds = self._safe_predict(ensemble, self.X, self.y)
        return scorer._score_func(self.y, preds)

    def _evaluate_on_holdout(self):
        scorer = get_scorer(self.scoring)
        preds = self._safe_predict(self.ensemble, self.X_val, self.y_val)
        return scorer._score_func(self.y_val, preds)

    def _check_fallback(self, selected_models):
        best_single_score = selected_models[0][2]
        if self.ensemble_score is not None and self.ensemble_score < best_single_score:
            print("⚠️ Ensemble underperforms best single model — falling back.")
            self.ensemble = selected_models[0][1]
            self.ensemble_score = best_single_score
            self.fallback_used = True

    def _log_mlflow(self, data):
        if self.mlflow_run:
            for k, v in data.items():
                mlflow.log_param(k, v)
            mlflow.sklearn.log_model(
                self.ensemble, artifact_path="final_ensemble_model"
            )

    def get_ensemble(self):
        return self.ensemble

    def get_ensemble_info(self):
        return {
            "ensemble_score": self.ensemble_score,
            "fallback_used": self.fallback_used,
            "ensemble_details": str(self.ensemble),
        }
