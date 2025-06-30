#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from typing import Dict, Any
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)


class AutoBaseline:
    def __init__(self, target: str, verbose: bool = True):
        self.target = target
        self.verbose = verbose
        self.results: Dict[str, Dict[str, Any]] = {}
        self.REPORT_DIR = Path("reports/baseline")
        self.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    def run(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Dict[str, Dict[str, Any]]:
        y_train = train_df[self.target]
        y_test = test_df[self.target]

        # Determine if regression (continuous) vs classification
        is_regression = pd.api.types.is_float_dtype(y_train.dtype)

        if is_regression:
            self._run_regression_baselines(train_df, test_df, y_train, y_test)
        else:
            self._run_classification_baselines(train_df, test_df, y_train, y_test)

        # TODO: Add Report to HTML file named baseline model stats
        # self.REPORT_DIR

        return self.results

    def _run_regression_baselines(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        X_train = train_df.drop(columns=[self.target])
        X_test = test_df.drop(columns=[self.target])

        # 1) Mean regressor
        dr_mean = DummyRegressor(strategy="mean")
        dr_mean.fit(X_train, y_train)
        y_pred_mean = dr_mean.predict(X_test)

        metrics_mean = {
            "type": "mean_regressor",
            "mae": float(mean_absolute_error(y_test, y_pred_mean)),
            "mse": float(mean_squared_error(y_test, y_pred_mean)),
            "r2": float(r2_score(y_test, y_pred_mean)),
        }
        self.results["mean_regressor"] = metrics_mean
        if self.verbose:
            print(
                f"[mean_regressor] MAE={metrics_mean['mae']:.4f}, "
                f"MSE={metrics_mean['mse']:.4f}, R2={metrics_mean['r2']:.4f}"
            )

        # 2) Median regressor
        dr_med = DummyRegressor(strategy="median")
        dr_med.fit(X_train, y_train)
        y_pred_med = dr_med.predict(X_test)

        metrics_med = {
            "type": "median_regressor",
            "mae": float(mean_absolute_error(y_test, y_pred_med)),
            "mse": float(mean_squared_error(y_test, y_pred_med)),
            "r2": float(r2_score(y_test, y_pred_med)),
        }
        self.results["median_regressor"] = metrics_med
        if self.verbose:
            print(
                f"[median_regressor] MAE={metrics_med['mae']:.4f}, "
                f"MSE={metrics_med['mse']:.4f}, R2={metrics_med['r2']:.4f}"
            )

    def plot_summary(self):
        df = pd.DataFrame.from_dict(self.results, orient="index")
        df.plot(kind="bar", figsize=(10, 6), title="Baseline Model Comparison")
        plt.tight_layout()
        path = self.REPORT_DIR / "baseline_summary.png"
        plt.savefig(path)
        plt.close()
        return str(path)

    def _run_classification_baselines(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        X_train = train_df.drop(columns=[self.target])
        X_test = test_df.drop(columns=[self.target])

        # 1) Most frequent
        dc_mf = DummyClassifier(strategy="most_frequent", random_state=0)
        dc_mf.fit(X_train, y_train)
        y_pred_mf = dc_mf.predict(X_test)

        metrics_mf = {
            "type": "most_frequent",
            "accuracy": float(accuracy_score(y_test, y_pred_mf)),
            "f1": float(f1_score(y_test, y_pred_mf, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_mf, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_mf, zero_division=0)),
        }

        self.results["most_frequent"] = metrics_mf
        if self.verbose:
            print(
                f"[most_frequent] Acc={metrics_mf['accuracy']:.4f}, "
                f"F1={metrics_mf['f1']:.4f}, Prec={metrics_mf['precision']:.4f}, Rec={metrics_mf['recall']:.4f}"
            )

        # 2) Stratified
        dc_strat = DummyClassifier(strategy="stratified", random_state=0)
        dc_strat.fit(X_train, y_train)
        y_pred_strat = dc_strat.predict(X_test)

        metrics_strat = {
            "type": "stratified",
            "accuracy": float(accuracy_score(y_test, y_pred_strat)),
            "f1": float(f1_score(y_test, y_pred_strat, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_strat, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_strat, zero_division=0)),
        }
        self.results["stratified"] = metrics_strat
        if self.verbose:
            print(
                f"[stratified] Acc={metrics_strat['accuracy']:.4f}, "
                f"F1={metrics_strat['f1']:.4f}, Prec={metrics_strat['precision']:.4f}, Rec={metrics_strat['recall']:.4f}"
            )

        # 3) Uniform (random)
        dc_unif = DummyClassifier(strategy="uniform", random_state=0)
        dc_unif.fit(X_train, y_train)
        y_pred_unif = dc_unif.predict(X_test)

        metrics_unif = {
            "type": "uniform",
            "accuracy": float(accuracy_score(y_test, y_pred_unif)),
            "f1": float(f1_score(y_test, y_pred_unif, zero_division=0)),
            "precision": float(precision_score(y_test, y_pred_unif, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred_unif, zero_division=0)),
        }
        self.results["uniform"] = metrics_unif
        if self.verbose:
            print(
                f"[uniform] Acc={metrics_unif['accuracy']:.4f}, "
                f"F1={metrics_unif['f1']:.4f}, Prec={metrics_unif['precision']:.4f}, Rec={metrics_unif['recall']:.4f}"
            )
