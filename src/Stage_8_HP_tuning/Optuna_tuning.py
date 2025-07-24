import optuna
import numpy as np
import warnings
import pandas as pd
import joblib
import mlflow
from sklearn.base import clone
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    Lasso,
    LinearRegression,
    RidgeClassifier,
)
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier
from configs import global_conf
from src.utils.utils import (
    detect_task_and_search_space,
    get_cv,
    get_metric,
    cross_val_objective,
    compute_metric,
)
import optuna.visualization as vis
from sklearn.metrics import get_scorer
from sklearn.exceptions import ConvergenceWarning

# Suppress specific sklearn convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")


class OptunaHyperTuner:
    """
    Unified Optuna tuner for multi-model selection with auto model & search space selection,
    advanced logging, and MLflow integration.
    """

    def __init__(
        self,
        X,
        y=None,
        X_test=None,
        y_test=None,
        n_trials=global_conf.DEFAULT_N_TRIALS,
        cv_folds=global_conf.DEFAULT_CV_FOLDS,
        scoring=None,
        sampler=global_conf.DEFAULT_SAMPLER,
        pruner=None,
        storage=None,
        study_name="optuna_study",
        mlflow_run=None,
        generate_reports=True,
    ):
        """
        X: feature dataframe
        y: target array/series or None (unsupervised)
        """
        self.X, self.y = X, y
        self.X_test, self.y_test = X_test, y_test
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.sampler = sampler
        self.storage = storage
        self.study_name = study_name
        self.mlflow_run = mlflow_run
        self.generate_reports = generate_reports

        self.pruner = pruner or global_conf.choose_pruner(X)

        self.task, self.scoring, self.search_spaces = detect_task_and_search_space(
            {}, y, scoring
        )
        if self.task == "classification" and scoring not in ["roc_auc", "accuracy"]:
            raise ValueError(f"Incompatible scoring {scoring} for classification task")
        if self.task == "regression" and scoring not in [
            "r2",
            "neg_mean_squared_error",
            "neg_root_mean_squared_error",
        ]:
            raise ValueError(f"Incompatible scoring {scoring} for regression task")
        self.models = self._auto_select_models()
        self.best_models = {}

    def _auto_select_models(self):
        if self.task == "classification":
            return {
                "RandomForestClassifier": RandomForestClassifier(
                    random_state=global_conf.RANDOM_STATE
                ),
                "GradientBoostingClassifier": GradientBoostingClassifier(
                    random_state=global_conf.RANDOM_STATE
                ),
                "ExtraTreesClassifier": ExtraTreesClassifier(
                    random_state=global_conf.RANDOM_STATE
                ),
                "XGBClassifier": XGBClassifier(
                    random_state=global_conf.RANDOM_STATE, eval_metric="logloss"
                ),
                "LGBMClassifier": LGBMClassifier(
                    random_state=global_conf.RANDOM_STATE, verbosity=-1
                ),
                "CatBoostClassifier": CatBoostClassifier(
                    random_state=global_conf.RANDOM_STATE, verbose=0
                ),
                "SVC": SVC(probability=True),
                "LogisticRegression": LogisticRegression(max_iter=1000),
            }
        elif self.task == "regression":
            return {
                "RandomForestRegressor": RandomForestRegressor(
                    random_state=global_conf.RANDOM_STATE
                ),
                "GradientBoostingRegressor": GradientBoostingRegressor(
                    random_state=global_conf.RANDOM_STATE
                ),
                "ExtraTreesRegressor": ExtraTreesRegressor(
                    random_state=global_conf.RANDOM_STATE
                ),
                "XGBRegressor": XGBRegressor(random_state=global_conf.RANDOM_STATE),
                "LGBMRegressor": LGBMRegressor(
                    random_state=global_conf.RANDOM_STATE, verbosity=-1
                ),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "LinearRegression": LinearRegression(),
            }
        elif self.task == "unsupervised":
            return {
                "KMeans": KMeans(),
                "DBSCAN": DBSCAN(),
                "GaussianMixture": GaussianMixture(),
            }
        else:
            raise ValueError(f"[AUTO MODELS] Unknown task: {self.task}")

    def evaluate(self, X_val, y_val):
        """
        Evaluate tuned best models on a separate held-out test set (no leakage).
        """
        metric_str = get_metric(self.task, self.scoring)
        print("\n🔎 Evaluating best models on held-out test set...")
        for model_name, (model, _) in self.best_models.items():
            test_score = compute_metric(y_val, model, X_val, metric_str)
            print(f"✅ {model_name} test {self.scoring}: {test_score:.4f}")
            if self.mlflow_run:
                mlflow.log_metric(f"{model_name}_test_{self.scoring}", test_score)

    def tune(self):
        """
        Run Optuna HPO for each model with child studies,
        MLflow logging, trial pruning stats, and optional Optuna plots.
        """
        for model_name, model in self.models.items():
            print(f"\n🔎 Starting HPO for model: {model_name}")
            model_class = model.__class__.__name__
            search_space = self.search_spaces.get(model_class)
            if not search_space:
                print(f"⚠️ Skipping {model_name}: no search space defined.")
                continue

            def objective(trial):
                params = {}
                for k, v in search_space.items():
                    try:
                        params[k] = v(trial)
                    except Exception as e:
                        print(f"[WARN] Skipping param {k} due to error: {e}")
                model_instance = clone(model).set_params(**params)
                cv = get_cv(self.cv_folds, self.task)
                score = cross_val_objective(
                    model_instance, self.X, self.y, cv, self.scoring, self.task
                )
                return score

            direction = self._infer_direction()
            study = optuna.create_study(
                direction=direction,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=self.storage,
                study_name=f"{self.study_name}_{model_name}",
                load_if_exists=True,
            )
            study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

            best_model = clone(model).set_params(**study.best_params)
            best_model.fit(self.X, self.y)
            print(f"✅ {model_name} best CV {self.scoring}: {study.best_value:.4f}")
            self.best_models[model_name] = (best_model, study)

            if self.mlflow_run:
                self._log_mlflow(model_name, study, best_model)

            if self.generate_reports:
                self._generate_optuna_plots(model_name, study)

            num_pruned = sum(
                1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
            )
            print(f"📉 {num_pruned} trials were pruned for {model_name}.")
            print(f"✅ Best trial for {model_name}: {study.best_trial.value:.5f}")

    def get_best_models(self):
        """
        Returns dict of model_name: (best_model, study)
        """
        return self.best_models

    def _infer_direction(self):
        higher_is_better = (
            "roc_auc" in str(self.scoring).lower()
            or "accuracy" in str(self.scoring).lower()
        )
        return "maximize" if higher_is_better else "minimize"

    def _log_mlflow(self, model_name, study, model):
        mlflow.log_metric(f"{model_name}_best_score", study.best_value)
        for k, v in study.best_params.items():
            mlflow.log_param(f"{model_name}_{k}", v)
        mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model")
        joblib.dump(model, f"artifacts/final/{model_name}_best_model.joblib")

    def _generate_optuna_plots(self, model_name, study):
        plots = [
            ("optimization_history", vis.plot_optimization_history),
            ("param_importance", vis.plot_param_importances),
            ("slice", vis.plot_slice),
            ("parallel_coords", vis.plot_parallel_coordinate),
        ]
        for suffix, plot_func in plots:
            try:
                fig = plot_func(study)
                fig.write_html(f"reports/optuna/{model_name}_{suffix}.html")
                if self.mlflow_run:
                    mlflow.log_artifact(f"reports/optuna/{model_name}_{suffix}.html")
            except Exception as e:
                print(f"[WARN] Could not generate {suffix} plot for {model_name}: {e}")
