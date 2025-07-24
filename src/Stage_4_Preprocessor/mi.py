#!/usr/bin/env python3
from __future__ import annotations
import time
import json
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from sklearn.base import clone
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.covariance import EmpiricalCovariance
import scipy.stats as stats

from joblib import Parallel, delayed
from src.utils.perfkit import perfclass, PerfMixin
from configs import global_conf

# Constants
REPORT_PATH = Path(f"{global_conf.PREPROCESSOR_REPORT_PATH}/missingness")
MODEL_PATH = Path(f"{global_conf.MODEL_ARTIFACTS_PATH}/missing_model_state.pkl")
REPORT_PATH.mkdir(parents=True, exist_ok=True)
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Thresholds and Configs
MAX_MISSING_FRAC_DROP = 0.90
VARIANCE_RATIO_CUTOFF = 0.5
COV_CHANGE_CUTOFF = 0.20
RARE_FREQ_CUTOFF = 0.01
RANDOM_STATE = 42
KNN_NEIGHBORS = 5
KNM_MICE_MAX_ROWS = 5000
KNM_MICE_MAX_COLS = 500


@perfclass()
class MissingImputer(PerfMixin):
    def __init__(
        self,
        max_missing_frac_drop: float = MAX_MISSING_FRAC_DROP,
        var_ratio_cutoff: float = VARIANCE_RATIO_CUTOFF,
        cov_change_cutoff: float = COV_CHANGE_CUTOFF,
        rare_freq_cutoff: float = RARE_FREQ_CUTOFF,
        knn_neighbors: int = KNN_NEIGHBORS,
        knn_mice_max_rows: int = KNM_MICE_MAX_ROWS,
        knn_mice_max_cols: int = KNM_MICE_MAX_COLS,
        n_jobs: int = -1,
        use_gpu: Optional[bool] = None,
        verbose: bool = False,
        max_figures: int = 50,
        random_state: int = RANDOM_STATE,
        model_path: Union[str, Path] = MODEL_PATH,
    ):
        self.max_missing_frac_drop = max_missing_frac_drop
        self.var_ratio_cutoff = var_ratio_cutoff
        self.cov_change_cutoff = cov_change_cutoff
        self.rare_freq_cutoff = rare_freq_cutoff
        self.knn_neighbors = knn_neighbors
        self.knn_mice_max_rows = knn_mice_max_rows
        self.knn_mice_max_cols = knn_mice_max_cols
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.max_figures = max_figures
        self.random_state = random_state
        self.model_path = Path(model_path)

        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.cols_to_drop: List[str] = []
        self.numeric_imputers: Dict[str, Tuple[str, Optional[object]]] = {}
        self.categorical_imputers: Dict[str, Tuple[str, Optional[str]]] = {}
        self.train_numeric: Optional[pd.DataFrame] = None
        self.report: Dict[str, Dict] = {
            "missing_pattern": {},
            "dropped_cols": {"numeric": [], "categorical": []},
            "missing_numeric": {},
            "missing_categorical": {},
            "other_columns": {},
            "rare_levels": {},
        }

        super().__init__(n_jobs=n_jobs, use_gpu=use_gpu)

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
        self._log(f"✔ Saved MissingImputer model to {Path(filepath).resolve()}")

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

    def _log(self, msg: str):
        if self.verbose:
            print(f"[MissingImputer] {msg}")

    def _cov_matrix(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        complete = df.dropna()
        if complete.shape[0] < 5:
            return None
        return EmpiricalCovariance().fit(complete.values).covariance_

    def _ks_var_cov_metrics(
        self,
        original: pd.Series,
        imputed: pd.Series,
        cov_before: Optional[np.ndarray],
        df_all: pd.DataFrame,
        col: str,
    ) -> Tuple[float, float, float]:
        try:
            ks = stats.ks_2samp(original.dropna(), imputed.dropna()).pvalue
        except Exception:
            ks = 0.0
        try:
            var_ratio = np.var(imputed) / (np.var(original.dropna()) + 1e-9)
        except Exception:
            var_ratio = 0.0
        try:
            if cov_before is None:
                return ks, var_ratio, 0.0
            temp_df = df_all.copy()
            temp_df[col] = imputed
            cov_after = self._cov_matrix(temp_df)
            if cov_after is None:
                return ks, var_ratio, 0.0
            idx = self.numeric_cols.index(col)
            delta = np.abs(cov_after[idx] - cov_before[idx]) / (
                np.abs(cov_before[idx]) + 1e-9
            )
            return ks, var_ratio, float(np.sum(delta))
        except Exception:
            return ks, var_ratio, 0.0

    def describe_strategy(self, col: str) -> str:
        """Return a human-readable description for how a column was imputed."""
        if col in self.report["missing_numeric"]:
            strat = self.report["missing_numeric"][col]["strategy"]
            return f"Column '{col}' is numeric and was imputed using '{strat}' strategy based on distributional similarity."
        elif col in self.report["missing_categorical"]:
            strat = self.report["missing_categorical"][col]["strategy"]
            return f"Column '{col}' is categorical and was imputed using '{strat}' strategy based on similarity in category distribution."
        elif col in self.cols_to_drop:
            return f"Column '{col}' had too much missing data and was dropped."
        return f"Column '{col}' did not require imputation."

    def _english_summary(self) -> str:
        lines = []

        # Dropped columns
        if self.cols_to_drop:
            lines.append(
                f"Dropped {len(self.cols_to_drop)} columns due to high missingness:"
            )
            for c in self.cols_to_drop:
                lines.append(f"  • {c}")

        # Numeric
        if self.report.get("missing_numeric"):
            lines.append("\nNumeric Imputation Strategies:")
            for col, data in self.report["missing_numeric"].items():
                lines.append(
                    f"  • {col}: {data['strategy']} (ks={data['metrics'].get('ks_p', 0):.2f}, "
                    f"var_ratio={data['metrics'].get('var_ratio', 0):.2f}, "
                    f"cov_delta={data['metrics'].get('cov_delta', 0):.2f})"
                )

        # Categorical
        if self.report.get("missing_categorical"):
            lines.append("\nCategorical Imputation Strategies:")
            for col, data in self.report["missing_categorical"].items():
                lines.append(
                    f"  • {col}: {data['strategy']} (score={max(data['scores'].values()):.2f})"
                )

        # Rare levels
        if self.report.get("rare_levels"):
            lines.append("\nRare Value Collapsing:")
            for col, levels in self.report["rare_levels"].items():
                lines.append(f"  • {col}: collapsed {len(levels)} rare levels")

        return "\n".join(lines)

    def generate_report(self, filename: str = "imputer_report", html: bool = True):
        report_dir = REPORT_PATH
        report_dir.mkdir(parents=True, exist_ok=True)

        txt_path = report_dir / f"{filename}.txt"
        with open(txt_path, "w") as f:
            f.write(self._english_summary())
        self._log(f"🗒️ English summary saved to {txt_path}")

    def _fit_single_numeric(
        self, series: pd.Series, col: str, cov_before: Optional[np.ndarray]
    ):
        original = series
        df = pd.DataFrame({col: series})
        imputers = {
            "mean": SimpleImputer(strategy="mean"),
            "median": SimpleImputer(strategy="median"),
            "knn": KNNImputer(n_neighbors=self.knn_neighbors),
            "mice": IterativeImputer(
                estimator=BayesianRidge(), random_state=self.random_state
            ),
            "random": None,  # handled separately
        }

        results = {}
        for name, imp in imputers.items():
            try:
                if name == "random":
                    fill_vals = original.dropna()
                    imputed = original.copy()
                    mask = imputed.isna()
                    imputed[mask] = np.random.choice(
                        fill_vals, size=mask.sum(), replace=True
                    )
                elif name in {"knn", "mice"}:
                    block = self.train_numeric[self.numeric_cols].copy()
                    if (
                        block.shape[0] > self.knn_mice_max_rows
                        or block.shape[1] > self.knn_mice_max_cols
                    ):
                        continue
                    imp_block = imp.fit_transform(block)
                    imputed = pd.Series(
                        imp_block[:, self.numeric_cols.index(col)], index=block.index
                    )
                else:
                    imp.fit(df[[col]])
                    imputed = imp.transform(df[[col]]).ravel()
                    imputed = pd.Series(imputed, index=series.index)

                ks, var_ratio, cov_delta = self._ks_var_cov_metrics(
                    original, imputed, cov_before, self.train_numeric, col
                )
                results[name] = {
                    "ks_p": ks,
                    "var_ratio": var_ratio,
                    "cov_delta": cov_delta,
                }
            except Exception:
                continue

        for name, metrics in results.items():
            if (
                metrics["ks_p"] > 0.05
                and metrics["var_ratio"] > self.var_ratio_cutoff
                and metrics["cov_delta"] < self.cov_change_cutoff
            ):
                return col, name, None, metrics, imputed

        return col, "mean", None, results.get("mean", {}), imputed

    def _fit_numeric(self, df: pd.DataFrame, cov_before: Optional[np.ndarray]):
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.train_numeric = df[self.numeric_cols].copy()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_numeric)(df[col], col, cov_before)
            for col in self.numeric_cols
            if df[col].isna().any()
        )

        for col, method, imputer_obj, metrics, imputed_col in results:
            self.numeric_imputers[col] = (method, imputer_obj)
            self.report["missing_numeric"][col] = {
                "strategy": method,
                "metrics": metrics,
            }

    def _fit_single_categorical(self, series: pd.Series, col: str):
        orig = series.astype(str)
        mode_val = orig.mode().iloc[0] if not orig.mode().empty else "__MISSING__"

        # Option A: Mode
        mode_filled = orig.fillna(mode_val)

        # Option B: Constant
        const_filled = orig.fillna("__MISSING__")

        # Option C: Random
        fill_vals = orig.dropna()
        rand_filled = orig.copy()
        mask = rand_filled.isna()
        if not fill_vals.empty:
            rand_filled[mask] = np.random.choice(
                fill_vals, size=mask.sum(), replace=True
            )
        else:
            rand_filled[mask] = "__MISSING__"

        # TVD scoring
        def tvd(p, q):
            all_vals = list(set(p.index) | set(q.index))
            return 0.5 * sum(abs(p.get(v, 0) - q.get(v, 0)) for v in all_vals)

        scores = {
            "mode": 1
            - tvd(
                orig.value_counts(normalize=True),
                mode_filled.value_counts(normalize=True),
            ),
            "constant": 1
            - tvd(
                orig.value_counts(normalize=True),
                const_filled.value_counts(normalize=True),
            ),
            "random": 1
            - tvd(
                orig.value_counts(normalize=True),
                rand_filled.value_counts(normalize=True),
            ),
        }

        best_cat = max(scores, key=scores.get)
        if best_cat == "mode":
            val = mode_val
        elif best_cat == "constant":
            val = "__MISSING__"
        else:
            val = "__RANDOM__"

        return (
            col,
            best_cat,
            val,
            scores,
            {"mode": mode_filled, "constant": const_filled, "random": rand_filled}[
                best_cat
            ],
        )

    def _fit_categorical(self, df: pd.DataFrame):
        self.categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_categorical)(df[col], col)
            for col in self.categorical_cols
            if df[col].isna().any()
        )

        for col, strategy, value, scores, imputed_series in results:
            self.categorical_imputers[col] = (strategy, value)
            self.report["missing_categorical"][col] = {
                "strategy": strategy,
                "scores": scores,
            }

            freqs = df[col].value_counts(normalize=True)
            rare_vals = freqs[freqs < self.rare_freq_cutoff].index.tolist()
            if rare_vals:
                self.report["rare_levels"][col] = rare_vals

    def fit(self, df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None):
        df = df.copy()
        df = df.dropna(axis=1, how="all")
        df = df.loc[:, df.columns.notnull()]
        df = df.loc[:, ~df.columns.duplicated()]

        self._log("🔍 Analyzing missingness...")
        analysis = MissingnessAnalyzerV2.analyze(df)
        self.report["missing_pattern"] = analysis

        # Drop over-threshold columns
        drop_cols = df.columns[df.isna().mean() > self.max_missing_frac_drop]
        self.cols_to_drop = drop_cols.tolist()
        df.drop(columns=self.cols_to_drop, inplace=True)

        # Split numeric/categorical
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        self.categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        self.train_numeric = df[self.numeric_cols].copy()

        self._log("⚙️ Fitting numeric imputers...")
        cov_before = self._cov_matrix(df[self.numeric_cols])
        self._fit_numeric(df, cov_before)

        self._log("⚙️ Fitting categorical imputers...")
        self._fit_categorical(df)

        # Save report and model if specified
        self.generate_report()
        if save_path is None:
            save_path = self.model_path

        if save_path:
            self.save(save_path)

        self._log("✅ Fitting complete.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Drop flagged columns
        df.drop(
            columns=[c for c in self.cols_to_drop if c in df.columns],
            errors="ignore",
            inplace=True,
        )

        # Numeric
        for col in self.numeric_cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue

            method, imp = self.numeric_imputers.get(col, ("mean", None))

            if method == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif method == "median":
                df[col] = df[col].fillna(df[col].median())
            elif method == "random":
                vals = df[col].dropna()
                df[col] = df[col].copy()
                df[col][df[col].isna()] = np.random.choice(
                    vals, size=df[col].isna().sum(), replace=True
                )
            elif method in {"knn", "mice"}:
                try:
                    sub = df[self.numeric_cols].copy()
                    _, saved_imp = self.numeric_imputers.get(col, (method, None))
                    if saved_imp is None:
                        imp = (
                            KNNImputer(n_neighbors=self.knn_neighbors)
                            if method == "knn"
                            else IterativeImputer(
                                estimator=BayesianRidge(),
                                random_state=self.random_state,
                            )
                        )
                        arr = imp.fit_transform(sub)
                    else:
                        arr = saved_imp.transform(sub)

                    df[self.numeric_cols] = pd.DataFrame(
                        arr, columns=self.numeric_cols, index=sub.index
                    )
                except Exception as e:
                    self._log(f"⚠️ {method} failed for {col}, using mean. {e}")
                    df[col] = df[col].fillna(df[col].mean())
            self._log(self.describe_strategy(col))

        # Categorical
        for col in self.categorical_cols:
            if col not in df.columns or df[col].isna().sum() == 0:
                continue
            method, val = self.categorical_imputers.get(col, ("mode", "__MISSING__"))
            if method == "random":
                vals = df[col].dropna()
                df[col] = df[col].copy()
                if not vals.empty:
                    df[col][df[col].isna()] = np.random.choice(
                        vals, size=df[col].isna().sum(), replace=True
                    )
                else:
                    df[col] = df[col].fillna("__MISSING__")
            else:
                df[col] = df[col].fillna(val)

            # Rare level collapsing
            if col in self.report.get("rare_levels", {}):
                rare_vals = self.report["rare_levels"][col]
                df[col] = (
                    df[col]
                    .astype(str)
                    .apply(lambda x: "__RARE__" if x in rare_vals else x)
                )

            self._log(self.describe_strategy(col))

        return df

    def fit_transform(
        self, df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        return self.fit(df, save_path=save_path).transform(df)

    def load_and_transform(
        self, filepath: Union[str, Path], df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Loads an imputer from a saved pickle and applies transform to the input DataFrame.
        """
        self.load(filepath)
        return self.transform(df)
