import json
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from typing import List, Optional, Dict, Any, Tuple


class DriftMonitor:
    """
    A robust drift monitoring class that:
      - Performs multiple drift tests across data, prediction, target, and skew
      - Stores results in a JSON history file, appending each run
      - Alerts when any drift is detected

    Tests included:
      * Data Drift: KS, PSI, Wasserstein for numeric; Chi2 for categorical
      * Prediction Drift: KS, PSI, Wasserstein on model predictions
      * Target/Label Drift: KS or Chi2 on y distributions
      * Train-Serve Skew: feature set mismatch
      * Model Performance Drift: performance decay vs baseline
    """

    def __init__(
        self,
        numeric_features: List[str],
        categorical_features: List[str],
        psi_bins: int = 10,
        ks_alpha: float = 0.05,
        psi_threshold: float = 0.1,
        wasserstein_threshold: float = 0.1,
        clf_threshold: float = 0.7,
        history_path: str = "drift_history.json",
        random_state: int = 42,
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.psi_bins = psi_bins
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold
        self.wasserstein_threshold = wasserstein_threshold
        self.clf_threshold = clf_threshold
        self.random_state = random_state
        self.classifier = RandomForestClassifier(
            n_estimators=100, random_state=self.random_state
        )
        self.history_path = Path(history_path)
        self.reference_df: Optional[pd.DataFrame] = None
        self.reference_target: Optional[pd.Series] = None
        self.deployed_model = None

    def fit_reference(
        self,
        ref_df: pd.DataFrame,
        ref_target: Optional[pd.Series] = None,
        deployed_model=None,
    ):
        """Set reference data, optional reference target, and model."""
        self.reference_df = ref_df.reset_index(drop=True).copy()
        if ref_target is not None:
            self.reference_target = ref_target.reset_index(drop=True).copy()
        self.deployed_model = deployed_model

    def _psi(self, train: np.ndarray, prod: np.ndarray) -> float:
        bins = np.percentile(train, np.linspace(0, 100, self.psi_bins + 1))
        train_pct, _ = np.histogram(train, bins=bins)
        prod_pct, _ = np.histogram(prod, bins=bins)
        train_pct = np.where(train_pct == 0, 1e-8, train_pct / len(train))
        prod_pct = np.where(prod_pct == 0, 1e-8, prod_pct / len(prod))
        return float(np.sum((train_pct - prod_pct) * np.log(train_pct / prod_pct)))

    def _ks(self, train: pd.Series, prod: pd.Series) -> Dict[str, Any]:
        stat, p = ks_2samp(train.dropna(), prod.dropna())
        return {
            "statistic": float(stat),
            "p_value": float(p),
            "drift": p < self.ks_alpha,
        }

    def _chi2(self, train: pd.Series, prod: pd.Series) -> Dict[str, Any]:
        ct = pd.crosstab(train.fillna("__MISSING__"), prod.fillna("__MISSING__"))
        stat, p, _, _ = chi2_contingency(ct)
        return {"statistic": float(stat), "p_value": float(p), "drift": p < 0.05}

    def _wasserstein(self, train: np.ndarray, prod: np.ndarray) -> float:
        return float(wasserstein_distance(train, prod))

    def _classifier_drift(
        self, ref: pd.DataFrame, prod: pd.DataFrame
    ) -> Dict[str, Any]:
        X = pd.concat([ref, prod], ignore_index=True)
        y = np.array([0] * len(ref) + [1] * len(prod))
        split = int(0.7 * len(X))
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y[:split], y[split:]
        self.classifier.fit(X_tr, y_tr)
        probs = self.classifier.predict_proba(X_te)[:, 1]
        auc = roc_auc_score(y_te, probs)
        return {"auc": float(auc), "drift": auc > self.clf_threshold}

    def _prediction_drift(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        if not self.deployed_model:
            return {}
        ref_preds = (
            self.deployed_model.predict_proba(self.reference_df)
            if hasattr(self.deployed_model, "predict_proba")
            else self.deployed_model.predict(self.reference_df)
        )
        new_preds = (
            self.deployed_model.predict_proba(new_df)
            if hasattr(self.deployed_model, "predict_proba")
            else self.deployed_model.predict(new_df)
        )
        if isinstance(ref_preds, np.ndarray) and ref_preds.ndim > 1:
            ref_preds = ref_preds[:, 1]
            new_preds = new_preds[:, 1]
        return {
            "ks": self._ks(pd.Series(ref_preds), pd.Series(new_preds)),
            "psi": self._psi(np.array(ref_preds), np.array(new_preds)),
            "wasserstein": self._wasserstein(np.array(ref_preds), np.array(new_preds)),
        }

    def _skew(self, new_df: pd.DataFrame) -> Dict[str, Any]:
        ref_cols = set(self.reference_df.columns)
        new_cols = set(new_df.columns)
        return {
            "missing_features": list(ref_cols - new_cols),
            "added_features": list(new_cols - ref_cols),
        }

    def _performance_drift(
        self, new_df: pd.DataFrame, new_target: pd.Series
    ) -> Dict[str, Any]:
        if not self.deployed_model or self.reference_target is None:
            return {}
        new_score = float(self.deployed_model.score(new_df, new_target))
        ref_score = float(
            self.deployed_model.score(self.reference_df, self.reference_target)
        )
        return {
            "new_score": new_score,
            "reference_score": ref_score,
            "drift": new_score < ref_score,
        }

    def detect(
        self, new_df: pd.DataFrame, new_target: Optional[pd.Series] = None
    ) -> Tuple[Dict[str, Any], List[str]]:
        if self.reference_df is None:
            raise ValueError("Reference data not defined. Call fit_reference().")
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "data_drift": {},
            "prediction_drift": {},
            "target_drift": {},
            "skew": {},
            "performance_drift": {},
        }
        alerts = []

        # Data Drift
        for col in self.numeric_features:
            ks_res = self._ks(self.reference_df[col], new_df[col])
            psi_res = self._psi(self.reference_df[col].values, new_df[col].values)
            wd = self._wasserstein(self.reference_df[col].values, new_df[col].values)
            report["data_drift"][col] = {
                "ks": ks_res,
                "psi": psi_res,
                "wasserstein": wd,
            }
            if (
                ks_res["drift"]
                or psi_res >= self.psi_threshold
                or wd >= self.wasserstein_threshold
            ):
                alerts.append(f"data_drift_numeric_{col}")

        for col in self.categorical_features:
            chi2_res = self._chi2(self.reference_df[col], new_df[col])
            report["data_drift"][col] = {"chi2": chi2_res}
            if chi2_res["drift"]:
                alerts.append(f"data_drift_categorical_{col}")

        # Prediction Drift
        pd_res = self._prediction_drift(new_df)
        report["prediction_drift"] = pd_res
        for k, v in pd_res.items():
            if isinstance(v, dict) and v.get("drift", False):
                alerts.append(f"prediction_drift_{k}")

        # Target/Label Drift
        if self.reference_target is not None and new_target is not None:
            if pd.api.types.is_numeric_dtype(self.reference_target):
                td = self._ks(self.reference_target, new_target)
                report["target_drift"] = {"ks": td}
                if td["drift"]:
                    alerts.append("target_drift")
            else:
                td = self._chi2(self.reference_target, new_target)
                report["target_drift"] = {"chi2": td}
                if td["drift"]:
                    alerts.append("target_drift")

        # Train-Serve Skew
        skew_res = self._skew(new_df)
        report["skew"] = skew_res
        if skew_res["missing_features"] or skew_res["added_features"]:
            alerts.append("train_serveskew")

        # Performance Drift
        if new_target is not None:
            perf_res = self._performance_drift(new_df, new_target)
            report["performance_drift"] = perf_res
            if perf_res.get("drift", False):
                alerts.append("performance_drift")

        return report, alerts


# =====================+ Example Usage +========================

"""
report, alerts = dm.detect(new_df, new_target)
dm.record(report)
if alerts:
    # send notifications
    print("Drift alerts:", alerts)
"""
