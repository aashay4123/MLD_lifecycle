#!/usr/bin/env python3
import pickle
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from pathlib import Path
from src.utils.perfkit import perfclass, PerfMixin
from configs import global_conf
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns


@perfclass()
class OutlierDetector(BaseEstimator, TransformerMixin, PerfMixin):
    """
    Detect & treat outliers in one step (fit/fit_transform), then flag on new data via transform().

    – Univariate fences: IQR, Z-score, Modified Z, Tukey (2× IQR), 1st/99th percentile
    – Multivariate fences (Mahalanobis → MinCovDet fallback, LOF, IsolationForest)
    – “Real outlier” if flagged by ≥ outlier_threshold rules
    – fit/fit_transform: automatically drop or winsorize rows (unless cap_outliers=None)
    – transform(): flag (no removal) on new data, producing an “is_outlier” column

    Parameters
    ----------
    outlier_threshold : int
        Minimum vote count for a row to be considered a “real outlier.”
    robust_covariance : bool
        If True, will fallback to MinCovDet if EmpiricalCovariance flags >5% of rows.
    cap_outliers : Optional[bool]
        If True, winsorize detected rows (clip to 1st/99th percentiles).
        If False, drop detected rows outright.
        If None, do not remove or cap (only detect & report).
    model_family : Optional[str]
        If "linear" or "bayesian", we force winsorization (unless cap_outliers=False is explicitly set).
        Otherwise ignored. Default None.
    random_state : int
        Seed for IsolationForest, MinCovDet, and LOF.
    verbose : bool
        If True, prints basic log messages during detection & treatment.
    """

    # ─────────────── Instance Variables ───────────────
    REPORT_PATH = Path(f"{global_conf.PREPROCESSOR_REPORT_PATH}/outliers")
    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = Path(
        f"{global_conf.MODEL_ARTIFACTS_PATH}/outlier_model_state.pkl")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ─────────────── Class-Level Constants ───────────────
    # Univariate thresholds
    UNIV_IQR_FACTOR = 1.5  # multiplier for IQR fences
    UNIV_ZSCORE_CUTOFF = 3.0  # |z| > 3.0
    UNIV_MODZ_CUTOFF = 3.5  # |modified_z| > 3.5
    TUKEY_MULTIPLIER = 2.0  # Tukey uses 2× IQR
    PCTL_LOW = 5  # 5th percentile
    PCTL_HIGH = 95  # 95th percentile

    # Multivariate settings
    GAUSS_SKEW_THRESH = 1.0  # abs(skew) < 1.0
    GAUSS_KURT_THRESH = 5.0  # abs(kurtosis) < 5.0
    MAHA_MIN_RATIO = 5  # need n_samples ≥ 5 * n_features for Mahalanobis
    GAUSS_FRAC_THRESH = 0.6  # need ≥ 60% of features approx Gaussian
    LOF_MAX_SAMPLES = 2000  # only run LOF if n_samples < 2000
    LOF_MAX_FEATURES = 50  # only run LOF if n_features < 50
    ISO_CONTAMINATION = 0.01  # 1% contamination for IsolationForest

    MULTI_CI = 0.975  # confidence for Mahalanobis cutoff

    def __init__(
        self,
        outlier_threshold: int = 3,
        robust_covariance: bool = True,
        cap_outliers: bool = False,
        model_family: str = None,
        random_state: int = 42,
        verbose: bool = False,
        n_jobs: int = -1,  # Number of parallel jobs for scoring rules
        max_charts: int = 50
    ):
        self.outlier_threshold = outlier_threshold
        self.robust_covariance = robust_covariance
        self.cap_outliers = cap_outliers
        self.model_family = model_family
        self.random_state = random_state
        self.verbose = verbose

        # These fields will be set during fit():
        self.df: pd.DataFrame = None
        self.numeric_cols: List[str] = []
        self.scaler = None
        self.cov_estimator = None
        self.mahal_threshold = None
        self.lof_model = None
        self.iso_model = None
        self._n_jobs = n_jobs
        self.max_charts = max_charts

        # After fit, we store:
        self.train_clean_: pd.DataFrame = None  # post-treatment training set
        self.votes_table_: pd.DataFrame = None  # per-row rule votes
        # how many values clipped per numeric column
        self.clipped_counts_: Dict[str, int] = {}

        self.best_rules_per_column_: Dict[str, List[str]] = {}
        self.fences_: Dict[str, Dict[str, Tuple[float, float]]] = {}
        # Reporting
        self.report: Dict[str, Any] = {
            "univariate_outliers": {},  # {column -> {rule_name: count_flagged, ...}, ...}
            "multivariate_outliers": {},
            "real_outliers": {},  # {"indices": [...], "count": N}
            "treatment": {},  # details about drop vs winsorize
        }

    def convert_paths_to_str(self, obj):
        if isinstance(obj, dict):
            return {k: self.convert_paths_to_str(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_paths_to_str(v) for v in obj]
        elif isinstance(obj, Path):
            return str(obj)
        else:
            return obj

    def _plot_histogram(self, data, title="", hue=None, filename="") -> str:
        plt.figure(figsize=(6, 4))

        if hue is not None:
            # Combine data & hue into a long-form DataFrame
            df = pd.DataFrame({"value": data, "is_outlier": hue})
            sns.histplot(data=df, x="value", hue="is_outlier",
                         kde=True, palette="muted")
        else:
            sns.histplot(data, kde=True, color="steelblue")

        plt.title(title)
        chart_path = Path(f"{self.REPORT_PATH}/histogram")
        chart_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{chart_path}/{filename}")
        plt.close()
        return chart_path

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _save_state(self, filepath: str = MODEL_PATH):
        state = {
            "numeric_cols": self.numeric_cols,
            "scaler": self.scaler,
            "cov_estimator": self.cov_estimator,
            "lof_model": self.lof_model,
            "iso_model": self.iso_model,
            "mahal_threshold": self.mahal_threshold,
            "outlier_threshold": self.outlier_threshold,
            "best_rules_per_column_": self.best_rules_per_column_,
            "fences_": self.fences_,
            "model_family": self.model_family,
            "cap_outliers": self.cap_outliers,
            "robust_covariance": self.robust_covariance,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        self._log(f"✔ Model state saved to {filepath}")

    def _load_state(self, filepath: str = MODEL_PATH):
        if not Path(filepath).exists():
            raise RuntimeError("No model fitted. Run `.fit()` first.")
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)
        self._log(f"✔ Model state loaded from {filepath}")

    # ───────────── Univariate Outlier Rules ─────────────

    def _iqr_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size == 0:
            return []
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - self.UNIV_IQR_FACTOR * iqr, q3 + self.UNIV_IQR_FACTOR * iqr
        return series[(series < lb) | (series > ub)].index.tolist()

    def _zscore_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size < 2:
            return []
        mu, sigma = series.mean(), series.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return []
        z = (series - mu) / sigma
        return series[np.abs(z) > self.UNIV_ZSCORE_CUTOFF].index.tolist()

    def _modz_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size < 2:
            return []
        med = np.median(arr)
        mad = np.median(np.abs(arr - med))
        if mad == 0:
            return []
        modz = 0.6745 * (series - med) / mad
        return series[np.abs(modz) > self.UNIV_MODZ_CUTOFF].index.tolist()

    def _tukey_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size == 0:
            return []
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lb, ub = q1 - self.TUKEY_MULTIPLIER * iqr, q3 + self.TUKEY_MULTIPLIER * iqr
        return series[(series < lb) | (series > ub)].index.tolist()

    def _percentile_outliers(self, series: pd.Series) -> List[int]:
        arr = series.dropna().values
        if arr.size == 0:
            return []
        p1, p99 = np.percentile(arr, [self.PCTL_LOW, self.PCTL_HIGH])
        return series[(series < p1) | (series > p99)].index.tolist()

    # ───────────── Multivariate Outlier Routines ─────────────

    def _fit_mahalanobis(self, df_numeric: pd.DataFrame):
        """
        1) Z-score numeric block → self.scaler
        2) Fit EmpiricalCovariance on z-scored data → self.cov_estimator
        3) Compute χ² cutoff at MULTI_CI (default 97.5%) → self.mahal_threshold
        """
        scaler = StandardScaler()
        Xz = scaler.fit_transform(df_numeric.values)
        self.scaler = scaler

        try:
            emp = EmpiricalCovariance().fit(Xz)
            self.cov_estimator = emp
        except Exception:
            self.cov_estimator = None

        p = Xz.shape[1]
        self.mahal_threshold = chi2.ppf(self.MULTI_CI, df=p)

    def _compute_mahalanobis_indices(self, df_numeric: pd.DataFrame) -> List[int]:
        """
        1) Transform df_numeric via self.scaler
        2) Compute MDs via self.cov_estimator.mahalanobis(...)
        3) If >5% of rows flagged and robust_covariance=True → refit with MinCovDet
        4) Return list of row indices exceeding threshold
        """
        if self.cov_estimator is None:
            return []

        Xz = self.scaler.transform(df_numeric.values)
        md = self.cov_estimator.mahalanobis(Xz)
        flagged = list(df_numeric.index[md > self.mahal_threshold])

        frac_flagged = len(flagged) / float(df_numeric.shape[0])
        if self.robust_covariance and frac_flagged > 0.05:
            # Fallback to MinCovDet
            try:
                mcd = MinCovDet(random_state=self.random_state).fit(Xz)
                self.cov_estimator = mcd
                md2 = mcd.mahalanobis(Xz)
                flagged = list(df_numeric.index[md2 > self.mahal_threshold])
            except Exception:
                # If MinCovDet fails, keep the original flags
                pass

        return flagged

    def _fit_lof(self, df_numeric: pd.DataFrame):
        """
        Fit a LocalOutlierFactor model in novelty mode so we can call .predict(...) on new data.
        Contamination fixed at ISO_CONTAMINATION.
        """
        lof = LocalOutlierFactor(
            n_neighbors=20, contamination=self.ISO_CONTAMINATION, novelty=True
        )
        lof.fit(df_numeric.values)
        self.lof_model = lof

    def _compute_lof_indices(self, df_numeric: pd.DataFrame) -> List[int]:
        """
        Return row indices where LOF.predict(...) == -1 (outlier).
        """
        if self.lof_model is None:
            return []
        preds = self.lof_model.predict(df_numeric.values)
        return list(df_numeric.index[preds == -1])

    def _fit_isolation_forest(self, df_numeric: pd.DataFrame):
        """
        Fit an IsolationForest on df_numeric with contamination=ISO_CONTAMINATION.
        """
        iso = IsolationForest(
            contamination=self.ISO_CONTAMINATION, random_state=self.random_state
        )
        iso.fit(df_numeric.values)
        self.iso_model = iso

    def _compute_isolation_indices(self, df_numeric: pd.DataFrame) -> List[int]:
        """
        Return row indices where IsolationForest.predict(...) == -1 (outlier).
        """
        if self.iso_model is None:
            return []
        preds = self.iso_model.predict(df_numeric.values)
        return list(df_numeric.index[preds == -1])

    def detect_multivariate_outliers(self) -> List[int]:
        """
        Chooses between Mahalanobis, LOF, or IsolationForest, in order:

          1) If n_features >= n_samples → skip Mahalanobis/LOF and go straight to IsolationForest.
          2) Else compute “Gaussian-like” fraction of columns (via |skew|<GAUSS_SKEW_THRESH and |kurtosis|<GAUSS_KURT_THRESH).
             If (n_samples >= MAHA_MIN_RATIO * n_features) and (frac_gaussian ≥ GAUSS_FRAC_THRESH):
               → run Mahalanobis. If Mahalanobis flags ≤5% of rows, accept those. Otherwise fall through.
          3) Else if (n_samples < LOF_MAX_SAMPLES) and (n_features < LOF_MAX_FEATURES):
               → run LOF. If LOF flags ≤5% of rows, accept those. Otherwise fall through.
          4) Otherwise run IsolationForest.

        Returns a list of flagged row indices.
        """
        df_num = self.df[self.numeric_cols].copy().dropna(axis=0, how="any")
        n_samples, n_features = df_num.shape

        if n_samples < 3 or n_features == 0:
            # Not enough data or no numeric features → skip
            self.report["multivariate_outliers"] = {
                "method": None,
                "indices": [],
                "notes": "too few samples or no numeric cols",
            }
            return []

        # 1) If p >= n, skip Mahalanobis/LOF
        if n_features >= n_samples:
            self._log(
                f"Warning: n_features ({n_features}) ≥ n_samples ({n_samples}), skipping Mahalanobis/LOF"
            )
            self._fit_isolation_forest(df_num)
            iso_idxs = self._compute_isolation_indices(df_num)
            self.report["multivariate_outliers"] = {
                "method": "IsolationForest",
                "indices": iso_idxs,
                "notes": "p >= n, direct to IsolationForest",
            }
            return iso_idxs

        # 2) Check “Gaussian-like” fraction
        skews = df_num.apply(lambda col: abs(col.dropna().skew()), axis=0)
        kurts = df_num.apply(lambda col: abs(col.dropna().kurtosis()), axis=0)
        gaussian_like = (
            (skews < self.GAUSS_SKEW_THRESH) & (kurts < self.GAUSS_KURT_THRESH)
        ).sum()
        frac_gaussian = gaussian_like / float(n_features)

        # Attempt Mahalanobis if conditions met
        mahal_used = False
        if (n_samples >= self.MAHA_MIN_RATIO * n_features) and (
            frac_gaussian >= self.GAUSS_FRAC_THRESH
        ):
            self._fit_mahalanobis(df_num)
            maha_idxs = self._compute_mahalanobis_indices(df_num)
            frac_flagged = len(maha_idxs) / float(n_samples)
            if frac_flagged <= 0.05:
                # Accept Mahalanobis
                mahal_used = True
                self.report["multivariate_outliers"] = {
                    "method": "Mahalanobis",
                    "indices": maha_idxs,
                    "frac_flagged": frac_flagged,
                    "notes": "EmpiricalCovariance accepted",
                }
                return maha_idxs
            else:
                # Mark that we tried Mahalanobis but flagged too many; record and fall through
                self.report["multivariate_outliers"] = {
                    "method": "Mahalanobis",
                    "indices": maha_idxs,
                    "frac_flagged": frac_flagged,
                    "notes": "flags > 5%, will fall back",
                }

        # 3) Attempt LOF if still applicable
        if (
            not mahal_used
            and (n_samples < self.LOF_MAX_SAMPLES)
            and (n_features < self.LOF_MAX_FEATURES)
        ):
            self._fit_lof(df_num)
            lof_idxs = self._compute_lof_indices(df_num)
            frac_flagged = len(lof_idxs) / float(n_samples)
            if frac_flagged <= 0.05:
                self.report["multivariate_outliers"] = {
                    "method": "LOF",
                    "indices": lof_idxs,
                    "frac_flagged": frac_flagged,
                    "notes": "LocalOutlierFactor accepted",
                }
                return lof_idxs
            else:
                # Log fallback
                self.report["multivariate_outliers"] = {
                    "method": "LOF",
                    "indices": lof_idxs,
                    "frac_flagged": frac_flagged,
                    "notes": "flags > 5%, will fall back",
                }

        # 4) Default → IsolationForest
        self._fit_isolation_forest(df_num)
        iso_idxs = self._compute_isolation_indices(df_num)
        self.report["multivariate_outliers"] = {
            "method": "IsolationForest",
            "indices": iso_idxs,
            "notes": "fallback to IsolationForest",
        }
        return iso_idxs

    def find_numeric_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Helper function to find numeric columns in a DataFrame.
        Returns a list of column names that are numeric.
        """
        return df.select_dtypes(include=[np.number]).columns.tolist()

    def report_outlier_detector(self, name: str, component: Any) -> Dict:
        """Generate a rich outlier detection report with parallel charts + summaries."""
        report = component.report if hasattr(component, "report") else {}
        df = getattr(component, "df", None)
        train_clean = getattr(component, "train_clean_", None)
        outlier_indices = set(report.get(
            "real_outliers", {}).get("indices", []))
        scores = getattr(component, "votes_table_",
                         {}).get("total_votes", None)
        cols = getattr(component, "numeric_cols", [])

        charts = []
        summary = {}

        if df is not None and scores is not None:
            # Add global summary: rows before/after treatment + missing values
            summary["rows_before"] = int(df.shape[0])
            summary["rows_after"] = int(
                train_clean.shape[0]) if train_clean is not None else None
            summary["rows_dropped"] = (
                summary["rows_before"] - summary["rows_after"]
                if summary["rows_after"] is not None
                else None
            )
            summary["rows_before"] = int(df.shape[0])
            summary["rows_after"] = int(
                train_clean.shape[0]) if train_clean is not None else None

            rows_dropped = (
                summary["rows_before"] - summary["rows_after"]
                if summary["rows_after"] is not None
                else None
            )

            # Only log rows_dropped if it's non-zero and not None
            if rows_dropped:
                summary["rows_dropped"] = rows_dropped

            summary["missing_before"] = int(df.isna().sum().sum())
            summary["missing_after"] = int(
                train_clean.isna().sum().sum()) if train_clean is not None else None

            # Compute standard deviation of outlier rows → prioritize most variable features
            stds = {
                col: df[col].loc[list(outlier_indices)].std()
                for col in cols
                if col in df.columns
            }
            top_cols = sorted(stds.items(), key=lambda x: x[1], reverse=True)[
                : self.max_charts]

            def process_col(item):
                col, _ = item
                n_outliers = int(
                    df[col].loc[list(outlier_indices)].dropna().shape[0])

                fig, axs = plt.subplots(1, 2, figsize=(12, 4))

                # Left: feature histogram with hue
                if df.index.isin(outlier_indices).any():
                    plot_df = pd.DataFrame({
                        "value": df[col],
                        "is_outlier": df.index.isin(outlier_indices)
                    })
                    sns.histplot(data=plot_df, x="value", hue="is_outlier",
                                 kde=True, ax=axs[0], palette="muted")
                else:
                    sns.histplot(df[col], kde=True,
                                 color="steelblue", ax=axs[0])
                axs[0].set_title(f"{name}: {col} (highlighted outliers)")

                # Right: votes distribution histogram for same rows
                if scores is not None:
                    sns.histplot(scores, bins=range(0, scores.max() + 2),
                                 discrete=True, ax=axs[1], color="coral")
                    axs[1].set_title(f"{name}: votes distribution")

                plt.tight_layout()

                chart_path = Path(f"{self.REPORT_PATH}/histogram")
                chart_path.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    f"{chart_path}/{name}_{col}_outliers_and_votes.png")
                plt.close(fig)

                return {
                    "col": col,
                    "chart": chart_path,
                    "n_outliers": n_outliers,
                }

            # Parallel chart creation
            results = self.parallel_map(process_col, top_cols)
            raw_counts = self.report.get("treatment", {}).get("counts", {})
            filtered_counts = {col: count for col,
                               count in raw_counts.items() if count > 0}
            if "treatment" in self.report:
                self.report["treatment"]["counts"] = filtered_counts
            for res in results:
                summary[res["col"]] = {"outliers_detected": res["n_outliers"]}
                charts.append(res["chart"])

        report = {"charts": charts, "summary": {**summary, **report}}
        # preprocessor_report_dir = global_conf.MODEL_ARTIFACTS_PATH
        # os.makedirs(preprocessor_report_dir, exist_ok=True)

        outlier_detector_path = os.path.join(
            self.REPORT_PATH, "outlier_detector_report.json"
        )

        serializable_report = self.convert_paths_to_str(report)

        with open(outlier_detector_path, "w") as f:
            json.dump(serializable_report, f, indent=2)

        return serializable_report, outlier_detector_path

    # ────────────── Main Fit & Fit_Transform ──────────────

    def fit(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_cols = self.find_numeric_columns(df)
        self.best_rules_per_column_ = {}
        self.fences_ = {}

        votes_df = pd.DataFrame(
            0,
            index=self.df.index,
            columns=[
                "iqr",
                "zscore",
                "modz",
                "tukey",
                "percentile",
                "mahalanobis",
                "lof",
                "isolation",
            ],
            dtype=int,
        )

        def score_rules(col: str):
            scores = {}
            fences = {}
            series = self.df[col]

            def try_rule(rule_name, func, *args):
                try:
                    idxs = func(series, *args)
                    for i in idxs:
                        votes_df.at[i, rule_name] = 1
                    scores[rule_name] = len(idxs)
                    return idxs
                except Exception:
                    scores[rule_name] = -1
                    return []

            # IQR
            idxs = try_rule("iqr", self._iqr_outliers)
            if idxs:
                q1, q3 = np.percentile(series.dropna().values, [25, 75])
                iqr = q3 - q1
                fences["iqr"] = (
                    q1 - self.UNIV_IQR_FACTOR * iqr,
                    q3 + self.UNIV_IQR_FACTOR * iqr,
                )

            # Z-score
            idxs = try_rule("zscore", self._zscore_outliers)
            if idxs:
                mu, sigma = series.mean(), series.std(ddof=0)
                fences["zscore"] = (mu, sigma)

            # Modified Z
            idxs = try_rule("modz", self._modz_outliers)
            if idxs:
                med = np.median(series.dropna())
                mad = np.median(np.abs(series.dropna() - med))
                fences["modz"] = (med, mad)

            # Tukey
            idxs = try_rule("tukey", self._tukey_outliers)
            if idxs:
                q1, q3 = np.percentile(series.dropna().values, [25, 75])
                iqr = q3 - q1
                fences["tukey"] = (
                    q1 - self.TUKEY_MULTIPLIER * iqr,
                    q3 + self.TUKEY_MULTIPLIER * iqr,
                )

            # Percentile
            idxs = try_rule("percentile", self._percentile_outliers)
            if idxs:
                p1, p99 = np.percentile(
                    series.dropna().values, [self.PCTL_LOW, self.PCTL_HIGH]
                )
                fences["percentile"] = (p1, p99)

            # Pick best 2 rules (you can limit to 1 if you want stricter)
            sorted_rules = sorted(
                [(k, v) for k, v in scores.items() if v >= 0], key=lambda x: -x[1]
            )
            best = [r[0] for r in sorted_rules[:2]]
            return col, best, fences

        results = self.parallel_map(
            score_rules, self.numeric_cols, prefer="threads")

        for col, best_rules, fences in results:
            self.best_rules_per_column_[col] = best_rules
            self.fences_[col] = fences

        # Multivariate voting
        multi_idxs = self.detect_multivariate_outliers()
        method_used = self.report["multivariate_outliers"].get("method", None)
        for i in multi_idxs:
            if method_used == "Mahalanobis":
                votes_df.at[i, "mahalanobis"] = 1
            elif method_used == "LOF":
                votes_df.at[i, "lof"] = 1
            else:
                votes_df.at[i, "isolation"] = 1

        # Real outliers
        votes_df["total_votes"] = votes_df.sum(axis=1)
        real_mask = votes_df["total_votes"] >= self.outlier_threshold
        real = votes_df.index[real_mask].tolist()
        self.report["real_outliers"] = {"indices": real, "count": len(real)}
        self.votes_table_ = votes_df.copy()

        # Treatment
        df_clean = self.df.copy()
        clipped_counts = {}

        force_winsorize = (self.model_family in ["linear", "bayesian"]) and (
            self.cap_outliers is not False
        )

        if self.cap_outliers is None and not force_winsorize:
            self.train_clean_ = df_clean.copy()
            self.clipped_counts_ = clipped_counts
            self.report["treatment"] = {"mode": "detect_only"}
        elif force_winsorize or self.cap_outliers is True:
            clipped_counts = {col: 0 for col in self.numeric_cols}

            # Precompute p1 and p99 for each column once
            percentile_bounds = {}
            for col in self.numeric_cols:
                arr = self.df[col].dropna().values
                if arr.size == 0:
                    continue
                p1, p99 = np.percentile(arr, [self.PCTL_LOW, self.PCTL_HIGH])
                percentile_bounds[col] = (p1, p99)

            # Apply winsorization only to detected real outlier indices
            for idx in self.report.get("real_outliers", {}).get("indices", []):
                if idx >= len(df_clean):
                    continue  # skip invalid index
                for col in self.numeric_cols:
                    p1, p99 = percentile_bounds.get(col, (None, None))
                    if p1 is None:
                        continue
                    val = df_clean.at[idx, col]
                    # Apply capping only if the value is outside bounds
                    if val < p1:
                        df_clean.at[idx, col] = p1
                        clipped_counts[col] += 1
                    elif val > p99:
                        df_clean.at[idx, col] = p99
                        clipped_counts[col] += 1

            self.train_clean_ = df_clean.copy()
            self.clipped_counts_ = clipped_counts
            self.report["treatment"] = {
                "mode": "winsorize",
                "counts": clipped_counts,
            }

        else:
            df_clean.drop(index=real, inplace=True)
            df_clean.reset_index(drop=True, inplace=True)
            self.train_clean_ = df_clean.copy()
            self.clipped_counts_ = clipped_counts
            self.report["treatment"] = {"mode": "drop", "dropped_rows": real}

        self._save_state(self.MODEL_PATH)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convenience: run fit(...) then return train_clean_.
        """
        return self.fit(df).train_clean_

    # ────────────── Transform (Flag Only on New Data) ──────────────

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._load_state(self.MODEL_PATH)
        result = df.copy()
        numeric_cols = [col for col in self.numeric_cols if col in df.columns]
        df_num = result[numeric_cols].copy()
        votes = pd.Series(0, index=result.index, dtype=int)

        for col in numeric_cols:
            if col not in self.best_rules_per_column_:
                continue
            rules = self.best_rules_per_column_[col]
            fences = self.fences_.get(col, {})
            series = result[col]

            for rule in rules:
                try:
                    if rule == "iqr" or rule == "tukey":
                        lb, ub = fences[rule]
                        votes.loc[(series < lb) | (series > ub)] += 1
                    elif rule == "zscore":
                        mu, sigma = fences[rule]
                        if sigma != 0:
                            z = (series - mu) / sigma
                            votes.loc[z.abs() > self.UNIV_ZSCORE_CUTOFF] += 1
                    elif rule == "modz":
                        med, mad = fences[rule]
                        if mad != 0:
                            modz = 0.6745 * (series - med) / mad
                            votes.loc[modz.abs() > self.UNIV_MODZ_CUTOFF] += 1
                    elif rule == "percentile":
                        p1, p99 = fences[rule]
                        votes.loc[(series < p1) | (series > p99)] += 1
                except Exception:
                    continue

        # Multivariate
        if self.scaler and self.cov_estimator:
            mask = ~df_num.isna().any(axis=1)
            if mask.any():
                try:
                    Xz = self.scaler.transform(df_num.loc[mask].values)
                    md = self.cov_estimator.mahalanobis(Xz)
                    votes.loc[df_num.loc[mask].index[md
                                                     > self.mahal_threshold]] += 1
                except Exception:
                    pass

        if self.lof_model:
            mask = ~df_num.isna().any(axis=1)
            if mask.any():
                try:
                    preds = self.lof_model.predict(df_num.loc[mask].values)
                    votes.loc[df_num.loc[mask].index[preds == -1]] += 1
                except Exception:
                    pass

        if self.iso_model:
            mask = ~df_num.isna().any(axis=1)
            if mask.any():
                try:
                    preds = self.iso_model.predict(df_num.loc[mask].values)
                    votes.loc[df_num.loc[mask].index[preds == -1]] += 1
                except Exception:
                    pass

        self.outlier_flags_ = votes >= self.outlier_threshold
        return result.copy()
