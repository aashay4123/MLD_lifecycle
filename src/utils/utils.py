# optuna_hpo/utils.py

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.utils.multiclass import type_of_target

from configs import global_conf


def get_cv(n_splits, task, random_state=42):
    """
    Returns appropriate cross-validator based on explicit task input.
    """
    if task == "classification":
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    elif task == "regression":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif task == "unsupervised":
        # For unsupervised, skip proper CV → single dummy split (entire data as train/val)
        return [(np.arange(len(range(10))), np.arange(len(range(10))))]
    else:
        raise ValueError(f"[get_cv] Unknown task type: {task}")


def get_metric(task, custom_metric=None):
    """
    Returns metric string based on task; uses sensible defaults.
    """
    if custom_metric:
        return custom_metric
    if task == "classification":
        return "roc_auc_ovo"
    elif task == "regression":
        return "r2"
    elif task == "unsupervised":
        return None
    else:
        raise ValueError(f"[get_metric] Unknown task type: {task}")


def compute_metric(y_true, model, X, metric_name, task=None):
    """
    Computes metric safely for classification (binary/multiclass) and regression.

    - Uses predict_proba/decision_function for ROC AUC on classification
    - Uses predict() directly for regression metrics
    - Handles multi-class ROC AUC gracefully with multi_class="ovr"
    """
    scorer = get_scorer(metric_name)
    y_type = type_of_target(y_true)
    n_classes = len(np.unique(y_true))

    if task == "regression" or y_type in ("continuous", "continuous-multioutput"):
        y_pred = model.predict(X)
        return scorer._score_func(y_true, y_pred)

    if "roc_auc" in metric_name.lower():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X)
        else:
            raise ValueError(
                f"Model {model.__class__.__name__} lacks predict_proba and decision_function, cannot compute ROC AUC."
            )

        # 🔥 FIX: ensure y_score shape is 1D for binary
        if y_score.ndim == 2 and y_score.shape[1] == 2:
            y_score = y_score[:, 1]

        if n_classes > 2:
            return scorer._score_func(y_true, y_score, multi_class="ovr")
        else:
            return scorer._score_func(y_true, y_score)
    else:
        y_pred = model.predict(X)
        return scorer._score_func(y_true, y_pred)


def cross_val_objective(model, X, y, cv, scoring, task):
    """
    Returns mean cross-validation score across folds, explicitly requiring task.
    """
    if task in ["classification", "regression"]:
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return np.mean(scores)
    elif task == "unsupervised":
        # No objective for unsupervised → return dummy zero (or you can implement clustering scores)
        return 0
    else:
        raise ValueError(f"[cross_val_objective] Unknown task type: {task}")


def detect_task_and_search_space(models, y, scoring=None):
    """
    Automatically detect task type (classification/regression/unsupervised) from y,
    assign sensible default metrics, and return the full search space dictionary.
    """
    if y is None:
        task = "unsupervised"
        scoring_metric = None
        search_spaces = global_conf.SEARCH_SPACES_UNSUPERVISED

    else:
        y_series = pd.Series(y)
        unique_values = y_series.nunique()
        unique_ratio = unique_values / len(y_series)

        if y_series.dtype == "O" or unique_values <= 20:
            task = "classification"
        elif np.issubdtype(y_series.dtype, np.integer) and unique_ratio < 0.05:
            task = "classification"
        elif np.issubdtype(y_series.dtype, np.floating) and unique_ratio < 0.05:
            task = "classification"
        else:
            task = "regression"

        scoring_metric = get_metric(task, scoring)
        search_spaces = (
            global_conf.SEARCH_SPACES_CLASSIFICATION
            if task == "classification"
            else global_conf.SEARCH_SPACES_REGRESSION
        )

    print(
        f"🔎 Detected task: {task} (unique={unique_values}, unique_ratio={unique_ratio:.4f})"
    )
    print(f"🔎 Models detected: {[m.__class__.__name__ for m in models.values()]}")
    print(f"🔎 Available search space keys: {list(search_spaces.keys())}")

    return task, scoring_metric, search_spaces
