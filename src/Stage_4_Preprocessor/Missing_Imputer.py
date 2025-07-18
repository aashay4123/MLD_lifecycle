#!/usr/bin/env python3
from __future__ import annotations
import matplotlib.pyplot as plt
import time
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import pickle
import numpy as np
import pandas as pd
import scipy.stats as stats
from src.utils.perfkit import perfclass, PerfMixin
from sklearn.base import clone

from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.covariance import EmpiricalCovariance
from joblib import Parallel, delayed
import os
from configs import global_conf
from .missingness_analyzer import MissingnessAnalyzer

REPORT_PATH = Path(f"{global_conf.PREPROCESSOR_REPORT_PATH}/missingness")
REPORT_PATH.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL_PATH = Path(
    f"{global_conf.MODEL_ARTIFACTS_PATH}/missing_model_state.pkl")
DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

MAX_MISSING_FRAC_DROP: float = 0.90
KNN_NEIGHBORS: int = 5
CAT_TVD_CUTOFF: float = 0.20
KNM_MICE_MAX_ROWS: int = 5000
KNM_MICE_MAX_COLS: int = 500
VARIANCE_RATIO_CUTOFF: float = 0.50
COV_CHANGE_CUTOFF: float = 0.20
RARE_FREQ_CUTOFF: float = 0.01
RANDOM_STATE: int = 42


@perfclass()
class MissingImputer(PerfMixin):
    def __init__(
        self,
        max_missing_frac_drop: float = MAX_MISSING_FRAC_DROP,
        knn_neighbors: int = KNN_NEIGHBORS,
        cat_tvd_cutoff: float = CAT_TVD_CUTOFF,
        knn_mice_max_rows: int = KNM_MICE_MAX_ROWS,
        knn_mice_max_columns: int = KNM_MICE_MAX_COLS,
        var_ratio_cutoff: float = VARIANCE_RATIO_CUTOFF,
        cov_change_cutoff: float = COV_CHANGE_CUTOFF,
        rare_freq_cutoff: float = RARE_FREQ_CUTOFF,
        random_state: int = RANDOM_STATE,
        verbose: bool = False,
        n_jobs: Union[int, float, None] = -1,
        use_gpu: Optional[bool] = None,
        model_path: Union[str, Path] = DEFAULT_MODEL_PATH,
        max_figures: int = 50,
    ):
        # Configuration
        self.max_missing_frac_drop = max_missing_frac_drop
        self.knn_neighbors = knn_neighbors
        self.cat_tvd_cutoff = cat_tvd_cutoff  # unused, but kept for signature
        self.knn_mice_max_rows = knn_mice_max_rows
        self.knn_mice_max_columns = knn_mice_max_columns
        self.var_ratio_cutoff = var_ratio_cutoff
        self.cov_change_cutoff = cov_change_cutoff
        self.rare_freq_cutoff = rare_freq_cutoff
        self.random_state = random_state
        self.verbose = verbose
        self.model_path = Path(model_path)
        self.n_jobs = n_jobs
        self.max_figures = max_figures

        # To be populated in fit()
        self.cols_to_drop: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.train_numeric: Optional[pd.DataFrame] = None
        self.numeric_imputers: Dict[str, Tuple[str, Optional[object]]] = {}
        self.categorical_imputers: Dict[str, Tuple[str, Optional[str]]] = {}

        # (B) hand-off to PerfMixin  (chains Parallel & GPU)
        super().__init__(n_jobs=n_jobs, use_gpu=use_gpu)

        # Aggregated report
        self.report: Dict[str, Dict] = {
            "missing_pattern": {},
            "dropped_cols": {"numeric": [], "categorical": []},
            "missing_numeric": {},
            "missing_categorical": {},
            "other_columns": {},
        }

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def save(self, filepath: Union[str, Path]):
        state = {
            "numeric_cols": self.numeric_cols,
            "categorical_cols": self.categorical_cols,
            "cols_to_drop": self.cols_to_drop,
            "numeric_imputers": self.numeric_imputers,
            "categorical_imputers": self.categorical_imputers,
            "shared_block_imputers": getattr(self, "shared_block_imputers", {}),
            "report": self.report,
            "random_state": self.random_state,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)
        self._log(
            f"✔ Saved MissingImputer model to {Path(filepath).resolve()}")

    def load(self, filepath: Union[str, Path]):
        with open(filepath, "rb") as f:
            state = pickle.load(f)

        self.numeric_cols = state["numeric_cols"]
        self.categorical_cols = state["categorical_cols"]
        self.cols_to_drop = state["cols_to_drop"]
        self.numeric_imputers = state["numeric_imputers"]
        self.categorical_imputers = state["categorical_imputers"]
        self.shared_block_imputers = state.get("shared_block_imputers", {})
        self.report = state["report"]
        self.random_state = state.get("random_state", 42)

    def _report_other_columns(self, df: pd.DataFrame) -> None:
        known = set(self.numeric_cols) | set(self.categorical_cols)
        for col in df.columns:
            if col not in known:
                self.report["other_columns"][col] = str(df[col].dtype)

    def _cast_mixed_numeric(self, df: pd.DataFrame):
        self.report.setdefault("mixed_casted", [])
        for col in df.columns:
            if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
                series = df[col].dropna().astype(str)
                # Count how many values are purely digit‐like (allow 1 decimal point, optional leading “-“)

                def is_digit_like(x: str) -> bool:
                    x2 = x.strip()
                    if x2.startswith("-"):
                        x2 = x2[1:]
                    # Should have at most one ".", and the rest digits
                    return bool(
                        (x2.count(".") <= 1)
                        and all(ch.isdigit() for ch in x2.replace(".", ""))
                    )
                total = len(series)
                if total > 0:
                    digit_like_count = series.map(is_digit_like).sum()
                    if digit_like_count / total >= 0.90:
                        # Cast entire column to numeric
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        self._log(
                            f"  • Cast '{col}' from object to numeric (mixed‐type)."
                        )
                        self.report["mixed_casted"].append(col)

    def _compute_cov_before(self) -> Optional[np.ndarray]:
        """
        Compute covariance matrix on self.train_numeric (complete‐case).
        Returns None if too few complete rows.
        """
        if self.train_numeric is None:
            return None
        complete = self.train_numeric.dropna()
        if complete.shape[0] < max(5, len(self.numeric_cols)):
            return None
        cov = EmpiricalCovariance().fit(complete.values).covariance_
        return cov

    def _random_sample_impute_num(self, orig: pd.Series) -> pd.Series:
        """
        Impute numeric by drawing random samples (with replacement) from non‐missing values.
        """
        nonnull = orig.dropna().values
        out = orig.copy()
        mask = out.isna()
        if len(nonnull) == 0:
            return out.fillna(0.0)
        rng = np.random.RandomState(self.random_state)
        out.loc[mask] = np.random.default_rng(self.random_state).choice(
            nonnull, size=mask.sum(), replace=True
        )

        return out

    def _evaluate_impute_num(
        self,
        col: str,
        orig: pd.Series,
        imputed: pd.Series,
        cov_before: Optional[np.ndarray],
    ) -> Tuple[float, float, float]:
        """
        Given original (orig) and candidate‐imputed (imputed) series:
          - KS p‐value: ks_2samp(orig_nonnull, imp_nonnull)[1]
          - VARIANCE_RATIO: var(imputed) / var(orig)
          - COVARIANCE_CHANGE: sum(|cov_after[col] – cov_before[col]| / (|cov_before[col]| + 1e-9))
        Returns (ks_p, var_ratio, cov_change).
        """
        orig_nonnull = orig.dropna().values
        imp_nonnull = imputed.dropna().values

        # 1) KS‐test
        if len(orig_nonnull) >= 2 and len(imp_nonnull) >= 2:
            try:
                ks_p = float(stats.ks_2samp(orig_nonnull, imp_nonnull)[1])
            except Exception:
                ks_p = 0.0
        else:
            ks_p = 0.0

        # 2) Variance ratio
        var_orig = float(np.nanvar(orig_nonnull)) if len(
            orig_nonnull) > 0 else np.nan
        var_imp = float(np.nanvar(imp_nonnull)) if len(
            imp_nonnull) > 0 else np.nan
        var_ratio = var_imp / \
            var_orig if (var_orig and var_orig > 0) else np.nan

        # 3) Covariance change
        if cov_before is None:
            cov_change = np.nan
        else:
            # Build a temporary copy of numeric block with this column replaced by `imputed`
            temp = self.train_numeric.copy()
            temp[col] = imputed.values
            complete_idx = temp.dropna().index
            if len(complete_idx) < 5:
                cov_change = np.nan
            else:
                cov_after = (
                    EmpiricalCovariance().fit(
                        temp.loc[complete_idx].values).covariance_
                )
                idx_feat = self.numeric_cols.index(col)
                diff = np.abs(cov_after[idx_feat, :] - cov_before[idx_feat, :])
                denom = np.abs(cov_before[idx_feat, :]) + np.finfo(float).eps
                cov_change = float(np.sum(diff / denom))
        return ks_p, var_ratio, cov_change

    def get_imputer(self, col: str) -> Optional[Tuple[str, Optional[object]]]:
        return (
            self.numeric_imputers.get(
                col) or self.categorical_imputers.get(col) or None
        )

    def _evaluate_single_numeric_column(
        self,
        df0: pd.DataFrame,
        col: str,
        knn_block: Optional[pd.DataFrame] = None,
        knn_imp: Optional[KNNImputer] = None,
        mice_block: Optional[pd.DataFrame] = None,
        mice_imp: Optional[IterativeImputer] = None,
        cov_before: Optional[np.ndarray] = None,
    ):
        orig = df0[col]
        n_missing = int(orig.isna().sum())
        if n_missing == 0:
            self._log(f"  • Numeric '{col}': no missing → skip")
            self.report["missing_numeric"][col] = {
                "chosen": "none",
                "note": "no missing",
            }
            self.numeric_imputers[col] = ("none", None)
            return

        self._log(
            f"  • Numeric '{col}': {n_missing} missing, evaluating imputers")
        orig_series = orig.copy()
        metrics: Dict[str, Tuple[float, float, float, float]] = {}
        candidates: Dict[str, pd.Series] = {}
        imputers: Dict[str, Optional[object]] = {}

        # --- Mean Imputer ---
        try:
            start = time.time()
            imp = SimpleImputer(strategy="mean")
            arr = pd.Series(
                imp.fit_transform(orig_series.values.reshape(-1, 1)).flatten(),
                index=orig_series.index,
            )
            ks_p, vr, cc = self._evaluate_impute_num(
                col, orig_series, arr, cov_before)
            if np.isnan(cc) or np.isnan(vr):
                continue  # skip this imputer
            runtime = time.time() - start
            metrics["mean"] = (ks_p, vr, cc, runtime)
            candidates["mean"] = arr
            imputers["mean"] = clone(imp)
            self._log(
                f"    • mean: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s"
            )
        except Exception:
            pass

        # --- Median Imputer ---
        try:
            start = time.time()
            imp = SimpleImputer(strategy="median")
            arr = pd.Series(
                imp.fit_transform(orig_series.values.reshape(-1, 1)).flatten(),
                index=orig_series.index,
            )
            ks_p, vr, cc = self._evaluate_impute_num(
                col, orig_series, arr, cov_before)
            if np.isnan(cc) or np.isnan(vr):
                continue  # skip this imputer

            runtime = time.time() - start
            metrics["median"] = (ks_p, vr, cc, runtime)
            candidates["median"] = arr
            imputers["median"] = clone(imp)
            self._log(
                f"    • median: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s"
            )
        except Exception:
            pass

        # --- Random‐sample Imputer ---
        try:
            start = time.time()
            arr = self._random_sample_impute_num(orig_series)
            ks_p, vr, cc = self._evaluate_impute_num(
                col, orig_series, arr, cov_before)
            if np.isnan(cc) or np.isnan(vr):
                continue  # skip this imputer
            runtime = time.time() - start
            metrics["random_sample"] = (ks_p, vr, cc, runtime)
            candidates["random_sample"] = arr
            imputers["random_sample"] = None
            self._log(
                f"    • random_sample: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s"
            )
        except Exception:
            pass

        # --- KNN Imputer ---
        if knn_block is not None:
            try:
                start = time.time()
                arr = knn_block[col]
                ks_p, vr, cc = self._evaluate_impute_num(
                    col, orig_series, arr, cov_before
                )
                if np.isnan(cc) or np.isnan(vr):
                    continue  # skip this imputer
                runtime = time.time() - start
                metrics["knn"] = (ks_p, vr, cc, runtime)
                candidates["knn"] = arr
                imputers["knn"] = clone(knn_imp)
                self._log(
                    f"    • knn (precomputed): ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s"
                )
            except Exception:
                pass

        # --- MICE Imputer ---
        if mice_block is not None:
            try:
                start = time.time()
                arr = mice_block[col]
                ks_p, vr, cc = self._evaluate_impute_num(
                    col, orig_series, arr, cov_before
                )
                if np.isnan(cc) or np.isnan(vr):
                    continue  # skip this imputer
                runtime = time.time() - start
                metrics["mice"] = (ks_p, vr, cc, runtime)
                candidates["mice"] = arr
                imputers["mice"] = clone(mice_imp)
                self._log(
                    f"    • mice (precomputed): ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}, time={runtime:.2f}s"
                )
            except Exception:
                pass

        # --- Choose Best Candidate ---
        best_method: Optional[str] = None
        # (ks, vr, –cov_ch)
        best_score: Tuple[float, float, float] = (-1.0, -1.0, np.inf)
        for method, (ks_p, vr, cc, rt) in metrics.items():
            # Check QC: var_ratio ≥ var_ratio_cutoff, cov_change ≤ cov_change_cutoff (or nan)
            if not np.isnan(vr) and vr < self.var_ratio_cutoff:
                continue
            if not np.isnan(cc) and cc > self.cov_change_cutoff:
                continue
            score = (ks_p, vr, -cc)
            if score > best_score:
                best_score = score
                best_method = method

        if best_method is None:
            # Fallback → mean
            arr = orig_series.fillna(orig_series.mean())
            ks_p, vr, cc = self._evaluate_impute_num(
                col, orig_series, arr, cov_before)
            best_method = "fallback_mean"
            imp_fb = SimpleImputer(strategy="mean")
            imp_fb.fit(orig_series.values.reshape(-1, 1))
            imputers["fallback_mean"] = clone(imp_fb)
            candidates["fallback_mean"] = arr
            best_score = (ks_p, vr, cc)
            self.numeric_imputers[col] = (best_method, clone(imp_fb))
            self._log(
                f"    • Fallback to mean: ks={ks_p:.3f}, vr={vr:.3f}, cov_ch={cc:.3f}"
            )

        # Record choice and apply to df0
        self.report["missing_numeric"][col] = {
            "chosen": best_method,
            "metrics": best_score,
        }
        self._log(
            f"    → Selected '{best_method}' for '{col}' with metrics={best_score}"
        )

        df0[col] = candidates[best_method].values
        self.numeric_imputers[col] = (best_method, imputers.get(best_method))

    def report_missing_imputer(self, df_before: pd.DataFrame, df_after: pd.DataFrame) -> Dict:
        """
        Comprehensive missingness report:
        - Logs missing counts before and after
        - Generates histograms for top missing cols (before and after)
        - Parallelized chart generation
        - Includes summary from self.report (fit analysis)

        Returns:
            dict with charts list & summary info
        """
        # 1️⃣ Compute missing summaries
        missing_before = df_before.isnull().sum()
        missing_after = df_after.isnull().sum()

        missing_before_total = int(missing_before.sum())
        missing_after_total = int(missing_after.sum())

        print(f"🔎 Total missing before imputation: {missing_before_total}")
        print(f"🔎 Total missing after imputation:  {missing_after_total}")

        # 2️⃣ Determine top N columns with most missing before imputation
        top_missing_cols = missing_before.sort_values(
            ascending=False).head(self.max_figures).index.tolist()

        charts = []

        def plot_hist(col):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            df_before[col].hist(bins=30, ax=axes[0])
            axes[0].set_title(f"[Before] {col}")
            df_after[col].hist(bins=30, ax=axes[1])
            axes[1].set_title(f"[After] {col}")
            plt.suptitle(f"Imputation Effect on {col}")
            plt.tight_layout()

            fname = f"missing_{col}_before_after.png"
            fpath = REPORT_PATH / fname
            fig.savefig(fpath)
            plt.close(fig)
            return str(fpath)

        # 3️⃣ Parallelize charts generation
        charts = Parallel(n_jobs=self.n_jobs)(
            delayed(plot_hist)(col) for col in top_missing_cols
        )

        # 4️⃣ Build full report dictionary
        report = {
            "summary": {
                "missing_before_total": missing_before_total,
                "missing_after_total": missing_after_total,
                "num_columns_before": df_before.shape[1],
                "num_columns_after": df_after.shape[1],
                "columns_with_missing_before": int((missing_before > 0).sum()),
                "columns_with_missing_after": int((missing_after > 0).sum()),
            },
            "charts": charts,
            # includes missing patterns, strategies, etc.
            "fit_report": self.report,
            "dropped_columns": self.cols_to_drop,
            "numeric_imputers": {col: info[0] for col, info in self.numeric_imputers.items()},
            "categorical_imputers": {col: info[0] for col, info in self.categorical_imputers.items()},
        }

        preprocessor_report_dir = global_conf.PREPROCESSOR_REPORT_PATH
        os.makedirs(preprocessor_report_dir, exist_ok=True)

        missing_imputer_path = os.path.join(
            preprocessor_report_dir, "missing_imputer_report.json"
        )

        with open(missing_imputer_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"✅ Missing imputer report completed → {REPORT_PATH}")

        return report, missing_imputer_path

    def fit(
        self,
        train_df: pd.DataFrame,
    ):
        df0 = train_df.copy()
        # 0) Missingness analysis (per‐column + save JSON)
        missing_pat = MissingnessAnalyzer.per_column_missingness(df0)
        self.report["missing_pattern"] = missing_pat

        # Any column not in numeric_cols or categorical_cols is 'other'
        self._report_other_columns(df0)

        # 1) Drop columns (numeric or categorical) with too many missing
        for col, info in missing_pat.items():
            if info["fraction_missing"] > self.max_missing_frac_drop:
                self.cols_to_drop.append(col)
                if pd.api.types.is_numeric_dtype(df0[col]):
                    self.report["dropped_cols"]["numeric"].append(col)
                else:
                    self.report["dropped_cols"]["categorical"].append(col)
                self._log(
                    f"DROPPED '{col}' (missing_frac={info['fraction_missing']:.2f} > {self.max_missing_frac_drop})"
                )

        df0 = df0.drop(columns=self.cols_to_drop)

        # 2) Cast mixed‐type numeric columns (object→numeric if ≥90% digit‐like)
        self._cast_mixed_numeric(df0)

        # 3) Identify numeric vs categorical (100% pure)
        self.numeric_cols = [
            c
            for c in df0.select_dtypes(include=[np.number]).columns
            if c not in self.cols_to_drop
        ]
        self.categorical_cols = [
            c for c in df0.columns if c not in self.numeric_cols]

        # 4) Keep a copy of numeric block for covariance computations
        if not self.numeric_cols:
            self.train_numeric = None
        else:
            self.train_numeric = df0[self.numeric_cols].copy()

        # 5) Compute covariance before imputation (on complete cases)
        cov_before = self._compute_cov_before()

        knn_block = None
        mice_block = None
        knn_imp = None
        mice_imp = None

        n_rows, n_num_cols = df0.shape[0], len(self.numeric_cols)
        can_use_knn_mice = (
            n_rows <= self.knn_mice_max_rows and n_num_cols <= self.knn_mice_max_columns
        )

        if can_use_knn_mice and self.train_numeric is not None:
            try:
                knn_imp = KNNImputer(n_neighbors=self.knn_neighbors)
                knn_block = pd.DataFrame(
                    knn_imp.fit_transform(self.train_numeric),
                    columns=self.numeric_cols,
                    index=self.train_numeric.index,
                )
                self._log("  • Precomputed KNN imputed block.")
            except Exception as e:
                self._log(f"  ⚠ KNN precompute failed: {e}")
                knn_block = None

            try:
                mice_imp = IterativeImputer(
                    estimator=BayesianRidge(),
                    sample_posterior=True,
                    random_state=self.random_state,
                    max_iter=10,
                )
                mice_block = pd.DataFrame(
                    mice_imp.fit_transform(self.train_numeric),
                    columns=self.numeric_cols,
                    index=self.train_numeric.index,
                )
                self._log("  • Precomputed MICE imputed block.")
            except Exception as e:
                self._log(f"  ⚠ MICE precompute failed: {e}")
                mice_block = None
        else:
            self._log(
                f"    • knn/mice skipped (dataset too large: rows={n_rows}, num_cols={n_num_cols})"
            )
        # 6) Impute numeric columns
        Parallel(n_jobs=self.n_jobs)(
            delayed(self._evaluate_single_numeric_column)(
                df0, col, knn_block, knn_imp, mice_block, mice_imp, cov_before
            )
            for col in self.numeric_cols
        )

        # 7) Drop categorical columns with too many missing
        for col in list(self.categorical_cols):
            frac_missing = df0[col].isna().mean()
            if frac_missing > self.max_missing_frac_drop:
                self.cols_to_drop.append(col)
                self.report["dropped_cols"]["categorical"].append(col)
                self._log(
                    f"DROPPED categorical '{col}' (missing_frac={frac_missing:.2f} > {self.max_missing_frac_drop})"
                )
                self.categorical_cols.remove(col)
                df0.drop(columns=[col], inplace=True)
        self.report["rare_levels"] = {}
        # 8) Collapse rare levels for remaining categorical columns
        for col in self.categorical_cols:
            freq = df0[col].value_counts(normalize=True)
            rare_levels = set(freq[freq < self.rare_freq_cutoff].index)
            if rare_levels:
                df0[col] = (
                    df0[col]
                    .where(~df0[col].isin(rare_levels), "__RARE__")
                    .astype("category")
                )
                self.report["rare_levels"][col] = list(rare_levels)
                self._log(
                    f"  • Categorical '{col}': collapsed {len(rare_levels)} rare levels → '__RARE__'"
                )

        # 9) Impute remaining categorical columns
        for col in self.categorical_cols:
            orig = df0[col].astype(object)
            n_missing = int(orig.isna().sum())
            if n_missing == 0:
                self._log(f"  • Categorical '{col}': no missing → skip")
                self.report["missing_categorical"][col] = {
                    "chosen": "none",
                    "note": "no missing",
                }
                self.categorical_imputers[col] = ("none", None)
                continue

            self._log(
                f"  • Categorical '{col}': {n_missing} missing, evaluating imputers"
            )
            # Compute mode
            if not orig.dropna().empty:
                mode_val = orig.dropna().mode().iloc[0]
            else:
                mode_val = "__MISSING__"
            arr_mode = orig.fillna(mode_val)
            common_levels = orig.dropna().unique()
            tvd_mode = float(
                np.sum(
                    np.abs(
                        orig.dropna()
                        .value_counts(normalize=True)
                        .reindex(common_levels, fill_value=0)
                        - arr_mode.value_counts(normalize=True).reindex(
                            common_levels, fill_value=0
                        )
                    )
                )
            )

            # Constant "__MISSING__"
            arr_const = orig.fillna("__MISSING__")
            tvd_const = float(
                np.sum(
                    np.abs(
                        orig.dropna().value_counts(normalize=True)
                        - arr_const.value_counts(normalize=True)
                    ).loc[orig.dropna().unique()]
                )
            )

            # Random-sample
            nonnull_vals = orig.dropna().values
            if len(nonnull_vals) == 0:
                arr_rand = orig.fillna("__MISSING__")
            else:
                arr_rand = orig.copy()
                mask = arr_rand.isna()
                rng = np.random.RandomState(self.random_state)
                arr_rand.loc[mask] = rng.choice(
                    nonnull_vals, size=mask.sum(), replace=True
                )
            tvd_rand = float(
                np.sum(
                    np.abs(
                        orig.dropna().value_counts(normalize=True)
                        - arr_rand.value_counts(normalize=True)
                    ).loc[orig.dropna().unique()]
                )
            )

            scores = {
                "mode": 1 - tvd_mode,
                "constant": 1 - tvd_const,
                "random": 1 - tvd_rand,
            }
            best_cat = max(scores, key=scores.get)
            self.report["missing_categorical"][col] = {
                "chosen": best_cat,
                "scores": scores,
            }
            self._log(
                f"    → Selected '{best_cat}' for '{col}' (scores mode={scores['mode']:.3f}, const={scores['constant']:.3f}, rand={scores['random']:.3f})"
            )

            if best_cat == "mode":
                df0[col] = arr_mode.values
                self.categorical_imputers[col] = ("mode", mode_val)
            elif best_cat == "constant":
                df0[col] = arr_const.values
                self.categorical_imputers[col] = ("constant", "__MISSING__")
            else:  # random
                df0[col] = arr_rand.values
                self.categorical_imputers[col] = ("random", None)

        # Final: store fully imputed training dataframe
        self.shared_block_imputers = {
            "knn": knn_imp,
            "mice": mice_imp,
        }
        self.train_imputed_ = df0.copy()
        self._log("MissingImputer → fit() completed.")
        self.save(self.model_path)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply fitted imputers to a new DataFrame (e.g. validation/test):
        - Drop columns that were dropped in fit()
        - For numeric cols: apply the chosen method & imputer
        - For categorical cols: apply the chosen strategy & value
        """
        if not hasattr(self, "numeric_imputers") or not self.numeric_imputers:
            if self.model_path.exists():
                self.load(self.model_path)
                self._log(f"ℹ Auto-loaded imputer from {self.model_path}")
            else:
                raise RuntimeError(
                    "No fitted state found. Please run `.fit()` before calling `.transform()`."
                )

        df1 = df.copy()
        print(
            f"MissingImputer → transform() on {df1.isnull().sum().sum()} ")
        # 1) Drop the same columns
        df1 = df1.drop(
            columns=[c for c in self.cols_to_drop if c in df1.columns], errors="ignore"
        )

        # 2) Precompute KNN/MICE blocks ONCE
        knn_imputed_block, mice_imputed_block = None, None
        valid_num_cols = [c for c in self.numeric_cols if c in df1.columns]

        methods_used = {method for method, _ in self.numeric_imputers.values()}
        if "knn" in methods_used:
            imp = self.shared_block_imputers.get("knn")
            if imp and valid_num_cols:
                if len(imp.feature_names_in_) != len(valid_num_cols):
                    raise ValueError(
                        f"[KNN] Column mismatch: expected {len(imp.feature_names_in_)}, got {len(valid_num_cols)}")
                block = df1[valid_num_cols]
                knn_imputed_block = pd.DataFrame(
                    imp.transform(block), columns=valid_num_cols, index=block.index
                )

        if "mice" in methods_used:
            imp = self.shared_block_imputers.get("mice")
            if imp and valid_num_cols:
                if len(imp.feature_names_in_) != len(valid_num_cols):
                    raise ValueError(
                        f"[KNN] Column mismatch: expected {len(imp.feature_names_in_)}, got {len(valid_num_cols)}")
                block = df1[valid_num_cols]
                mice_imputed_block = pd.DataFrame(
                    imp.transform(block), columns=valid_num_cols, index=block.index
                )

        def _transform_numeric_column(
            col, method, imputer_obj, df1, knn_block, mice_block
        ):
            if col not in df1.columns:
                return
            series = df1[col]
            if method == "none":
                df1[col] = series.fillna(series.mean())
            elif method in ["mean", "median", "fallback_mean"]:
                df1[col] = imputer_obj.transform(
                    series.values.reshape(-1, 1)).flatten()
            elif method == "knn" and knn_block is not None:
                df1[col] = knn_block[col]
            elif method == "knn" and knn_block is None:
                print(
                    f"[WARN] KNN block missing for '{col}' → fallback to mean")
                df1[col] = series.fillna(series.mean())
            elif method == "mice" and mice_block is not None:
                df1[col] = mice_block[col]
            elif method == "mice" and mice_block is None:
                print(
                    f"[WARN] mice block missing for '{col}' → fallback to mean")
                df1[col] = series.fillna(series.mean())
            elif method == "random_sample":
                nonnull = series.dropna().values
                mask = series.isna()
                if len(nonnull) == 0:
                    df1[col] = series.fillna(0.0)
                else:
                    rng = np.random.default_rng(self.random_state)
                    series.loc[mask] = rng.choice(
                        nonnull, size=mask.sum(), replace=True
                    )
                    df1[col] = series

        Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_numeric_column)(
                col, method, imputer_obj, df1, knn_imputed_block, mice_imputed_block
            )
            for col, (method, imputer_obj) in self.numeric_imputers.items()
        )
        # 3.5) Re-apply rare level collapsing (before categorical imputation)
        rare_levels = self.report.get("rare_levels", {})
        for col, levels in rare_levels.items():
            if col in df1.columns:
                df1[col] = df1[col].where(~df1[col].isin(
                    levels), "__RARE__").astype("category")
                self._log(f"  • Re-applied rare collapse to '{col}'")

        # 4) Categorical column-wise imputation
        def _transform_categorical_column(col, strategy, val, df1):
            if col not in df1.columns:
                return
            series = df1[col].astype(object)
            if strategy == "none":
                return
            elif strategy in ["mode", "constant"]:
                df1[col] = series.fillna(val).astype(object)
            elif strategy == "random":
                nonnull_vals = series.dropna().values
                mask = series.isna()
                if len(nonnull_vals) > 0:
                    rng = np.random.default_rng(self.random_state)
                    series.loc[mask] = rng.choice(
                        nonnull_vals, size=mask.sum(), replace=True
                    )
                else:
                    series.loc[mask] = "__MISSING__"
                df1[col] = series

        Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_categorical_column)(col, strategy, val, df1)
            for col, (strategy, val) in self.categorical_imputers.items()
        )
        print(
            f"MissingImputer → transform() on {df1.isnull().sum().sum()} ")
        return df1

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)
