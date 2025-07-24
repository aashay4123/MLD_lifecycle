#!/usr/bin/env python3
"""
FeatureScalerTransformer V6 – complete, advanced feature scaler/transformer with exhaustive evaluation,
multi-distribution auto-detection, before/after visualizations, structured multimodal handling, detailed reporting,
and interactive HTML output. Monitored & performance-tracked by PerfMixin + monitor decorators.

Usage:
    fst = FeatureScalerTransformer()
    X_train_scaled = fst.fit_transform(X_train)
    X_test_scaled = fst.transform(X_test)
    fst.save("scaler.pkl")
"""

import json
import logging
import os
import pickle
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.signal import find_peaks
from scipy.stats import beta, boxcox, expon, gamma, gaussian_kde
from sklearn.cluster import KMeans
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from src.utils.monitor import monitor
from src.utils.perfkit import PerfMixin, perfclass

try:
    from diptest import diptest

    DIPO_AVAILABLE = True
except ImportError:
    DIPO_AVAILABLE = False

warnings.filterwarnings("ignore", message=".*Precision loss.*")

SKEW_THRESHOLD = 1
KURTOSIS_THRESHOLD = (-2, 7)
ENTROPY_UNIFORM_THRESHOLD = 2.0
DIP_P_THRESHOLD = 0.05
VISUALS_DIR = "scaler_visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)
ALPHA = 0.1


@perfclass()
class FeatureScalerTransformer(PerfMixin):
    def __init__(
        self,
        alpha: float = ALPHA,
        skew_thresh: float = 1.0,
        qt_max_rows: int = 100_000,
        verbose: bool = True,
        report_file: str = "scaler_transform_report.csv",
    ):
        self.alpha = alpha
        self.skew_thresh = skew_thresh
        self.qt_max_rows = qt_max_rows
        self.verbose = verbose
        self.available_scalers = [
            StandardScaler(),
            MinMaxScaler(),
            RobustScaler(),
            QuantileTransformer(output_distribution="normal"),
        ]
        self.pre_funcs = {
            "none": lambda x: x.copy(),
            "log1p": lambda x: np.log1p(x),
            "sqrt": lambda x: np.sqrt(x),
            "cbrt": lambda x: np.cbrt(x),
            "reciprocal": lambda x: 1.0 / (x + 1e-9),
        }
        self.extra_transforms = ["none", "boxcox", "yeo", "quantile"]
        self.transformers_: Dict[str, Optional[Any]] = {}
        self.scalers_: Dict[str, Any] = {}
        self.report: Dict[str, Dict[str, Any]] = {}
        self.columns: List[str] = []
        self.global_scaler_: Optional[Any] = None
        self.report_file = report_file

        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="[%(levelname)s] %(message)s",
        )

    def _normality_scores(self, x: np.ndarray) -> Dict[str, float]:
        """Compute multiple normality tests and aggregate pass/fail, accounting for practical vs statistical significance."""
        result = {}
        try:
            p_shapiro = stats.shapiro(x)[1] if len(x) < 5000 else 1.0
        except Exception:
            p_shapiro = 0.0
        try:
            p_dagostino = stats.normaltest(x).pvalue if len(x) >= 20 else 1.0
        except Exception:
            p_dagostino = 0.0
        try:
            ad_stat = stats.anderson(x, dist="norm")
            p_anderson = (
                0.25 if ad_stat.statistic < ad_stat.critical_values[2] else 0.01
            )
        except Exception:
            p_anderson = 0.0

        # Compute individual passes
        passes = sum(
            [
                p_shapiro > self.alpha,
                p_dagostino > self.alpha,
                p_anderson > self.alpha,
            ]
        )
        pass_rate = passes / 3

        # Harmonic mean still included for record
        pvals = [max(p, 1e-8) for p in [p_shapiro, p_dagostino, p_anderson]]
        harmonic_p = len(pvals) / sum(1.0 / p for p in pvals)

        result.update(
            {
                "p_shapiro": p_shapiro,
                "p_dagostino": p_dagostino,
                "p_anderson": p_anderson,
                "harmonic_p": harmonic_p,
                "pass_rate": pass_rate,
                "passes_normality": pass_rate
                >= 0.66,  # require at least 2/3 tests pass
            }
        )
        logging.info(
            f"[NormalityScores] Shapiro={p_shapiro:.4f}, D'Agostino={p_dagostino:.4f}, "
            f"Anderson~p={p_anderson:.4f}, Harmonic={harmonic_p:.4f}, PassRate={pass_rate:.2f}"
        )
        return result

    def _fail_if_invalid(self, df: pd.DataFrame):
        """Fail fast if dataframe is empty, or if numeric columns are invalid."""
        if df.empty:
            raise ValueError("[ERROR] Input DataFrame is empty.")
        X_num = df.select_dtypes(include=[np.number])
        if X_num.empty:
            raise ValueError(
                "[ERROR] No numeric columns found. Please preprocess your data."
            )
        if X_num.shape[1] == 0:
            raise ValueError(
                "[ERROR] No eligible numeric columns after dtype filtering."
            )
        too_many_missing = X_num.isnull().mean() > 0.5
        if too_many_missing.any():
            cols = list(too_many_missing[too_many_missing].index)
            raise ValueError(
                f"[ERROR] Columns with >50% missing: {cols}. Please impute before scaling."
            )
        if (X_num.nunique(dropna=True) <= 1).all():
            raise ValueError(
                "[ERROR] All numeric columns are constant. Cannot scale constant data."
            )

    def _adaptive_thresholds(self, n_rows: int):
        """Adapt thresholds dynamically based on dataset size, unless manually set."""
        if self.alpha is None or self.skew_thresh is None:
            if n_rows < 500:
                self.alpha, self.skew_thresh = 0.05, 0.8
            elif n_rows > 1e5:
                self.alpha, self.skew_thresh = 0.2, 1.5
            else:
                self.alpha, self.skew_thresh = 0.1, 1.0
        logging.info(
            f"[AdaptiveThresholds] alpha={self.alpha}, skew_thresh={self.skew_thresh} for n_rows={n_rows}"
        )

    def estimate_modes_and_variance(self, x: np.ndarray) -> Tuple[int, bool]:
        kde = gaussian_kde(x)
        x_eval = np.linspace(np.min(x), np.max(x), 1000)
        y_kde = kde(x_eval)
        peaks, _ = find_peaks(
            y_kde, height=np.max(y_kde) * 0.1, distance=len(x_eval) // 20
        )
        num_modes = len(peaks)
        structured_multimodal = False
        if num_modes >= 3:
            structured_multimodal = True
        elif num_modes == 2:
            km = KMeans(n_clusters=2, n_init=5, random_state=42).fit(x.reshape(-1, 1))
            cluster_var = [np.var(x[km.labels_ == k]) for k in range(2)]
            overall_var = np.var(x)
            variance_ratio = (
                np.mean(cluster_var) / overall_var if overall_var > 0 else 0
            )
            if variance_ratio < 0.5:
                structured_multimodal = True
        return num_modes, structured_multimodal

    def _detect_distribution(
        self, stats_dict: Dict[str, Any], data: np.ndarray
    ) -> Tuple[str, float]:
        """
        Auto-detect distribution type using skew/kurtosis heuristics, entropy, DIP test,
        and best goodness-of-fit among top 10 common distributions.
        """
        skew, kurt, entropy = (
            stats_dict["skew"],
            stats_dict["kurtosis"],
            stats_dict["entropy"],
        )
        dip_p = None

        # DIP test
        if DIPO_AVAILABLE:
            try:
                dip, dip_p = diptest(data)
                logging.info(f"[DIPTest] dip_p={dip_p:.4f}")
            except Exception:
                dip_p = np.nan
                logging.warning("[DIPTest] DIP test failed; proceeding without it.")
        else:
            logging.info("[DIPTest] DIP test not installed; skipping.")

        # Evaluate multiple candidate distributions
        dist_candidates = [
            ("norm", stats.norm),
            ("gamma", stats.gamma),
            ("lognorm", stats.lognorm),
            ("beta", stats.beta),
            ("expon", stats.expon),
            ("t", stats.t),
            ("weibull", stats.weibull_min),
            ("pareto", stats.pareto),
            ("cauchy", stats.cauchy),
            ("logistic", stats.logistic),
        ]

        best_pval, best_fit = -1, "unknown"
        for name, dist in dist_candidates:
            try:
                params = dist.fit(data)
                _, pval = stats.kstest(data, name, args=params)
                if pval > best_pval:
                    best_pval, best_fit = pval, name
            except Exception:
                continue

        # Combine heuristics + DIP + best distribution
        if dip_p is None or np.isnan(dip_p):
            if (
                abs(skew) > 1.5
                or entropy > ENTROPY_UNIFORM_THRESHOLD + 1
                or abs(kurt) > 5
            ):
                detected = "multimodal"
            else:
                detected = best_fit
        elif dip_p <= DIP_P_THRESHOLD:
            detected = "multimodal"
        elif (
            abs(skew) < SKEW_THRESHOLD
            and KURTOSIS_THRESHOLD[0] < kurt < KURTOSIS_THRESHOLD[1]
            and dip_p > DIP_P_THRESHOLD
        ):
            detected = "normal-like"
        elif abs(skew) >= SKEW_THRESHOLD:
            detected = "skewed"
        elif entropy < ENTROPY_UNIFORM_THRESHOLD:
            detected = "uniform-like"
        else:
            detected = best_fit

        dip_p_str = (
            f"{dip_p:.4f}" if dip_p is not None and not np.isnan(dip_p) else "N/A"
        )
        logging.info(
            f"[DetectDistribution] skew={skew:.2f}, kurtosis={kurt:.2f}, entropy={entropy:.2f}, dip_p={dip_p_str} → DETECTED={detected.upper()} (best_pval={best_pval:.4f} for {best_fit})"
        )
        return detected, dip_p

    def _fit_and_ks(self, x: np.ndarray, dist) -> float:
        """
        Fit a scipy distribution on x and return 1 minus KS p-value as a distance score.
        Lower scores indicate better fit.
        """

        try:
            params = dist.fit(x)
            _, pvalue = stats.kstest(x, dist.cdf, args=params)
            return 1 - pvalue
        except Exception:
            return 1.0

    def _compute_stats(self, data: np.ndarray) -> Dict[str, Any]:
        return {
            "mean": np.mean(data),
            "std": np.std(data),
            "skew": stats.skew(data),
            "kurtosis": stats.kurtosis(data),
            "entropy": stats.entropy(np.histogram(data, bins=30)[0] + 1),
        }

    def _plot_before_after(
        self, col: str, original: np.ndarray, transformed: np.ndarray
    ):
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        axs[0, 0].hist(original, bins=30)
        axs[0, 0].set_title(f"{col} - Original Hist")
        stats.probplot(original, dist="norm", plot=axs[0, 1])
        axs[0, 1].set_title(f"{col} - Original QQ")
        axs[1, 0].hist(transformed, bins=30)
        axs[1, 0].set_title(f"{col} - Transformed Hist")
        stats.probplot(transformed, dist="norm", plot=axs[1, 1])
        axs[1, 1].set_title(f"{col} - Transformed QQ")
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, f"{col}_before_after.png"))
        plt.close("all")

    def _is_valid_for(self, method: str, x: np.ndarray) -> bool:
        """
        Validate whether the given method is appropriate for x.
        Returns False if applying the method would cause an error or nonsense result.
        """
        x_clean = x[~np.isnan(x)]
        if len(x_clean) < 3:
            logging.debug(f"[Validation] Too few non-NaN values for method '{method}'.")
            return False
        if np.isinf(x_clean).any():
            logging.debug(f"[Validation] Inf detected; method '{method}' not suitable.")
            return False
        if np.all(x_clean == x_clean[0]):
            logging.debug(
                f"[Validation] Constant data; method '{method}' not suitable."
            )
            return False

        if method == "boxcox":
            if not np.all(x_clean > 0):
                logging.debug("[Validation] Box-Cox requires all values > 0.")
                return False
        elif method in ("sqrt", "log1p"):
            if not np.all(x_clean >= 0):
                logging.debug(
                    f"[Validation] Method '{method}' requires non-negative values."
                )
                return False
        elif method == "reciprocal":
            if np.any(np.isclose(x_clean, 0)):
                logging.debug("[Validation] Reciprocal invalid with zeros present.")
                return False
        elif method == "quantile":
            # Quantile handles positive/negative; always allow
            return True
        # For all other methods (none, yeo, etc.), allow by default if basic checks passed
        return True

    def _evaluate_candidate(
        self,
        x: np.ndarray,
        method: str,
        scaler,
        scaler_name: str,
        col: str,
        original_entropy: float,
    ) -> Optional[Dict]:
        try:
            # Apply the candidate transformation
            x_t = self._apply_transform(x, method)
            x_scaled = scaler.fit_transform(x_t.reshape(-1, 1)).flatten()
            x_clean = x_scaled[~np.isnan(x_scaled)]

            # Calculate normality metrics
            normality = self._normality_scores(x_clean)
            skew_val = stats.skew(x_clean) if len(x_clean) > 2 else 0.0
            candidate_entropy = stats.entropy(np.histogram(x_clean, bins=30)[0] + 1)

            # Calculate entropy penalty ratio
            entropy_increase_ratio = (
                (candidate_entropy / original_entropy) / np.log(30)
                if original_entropy > 0
                else 1.0
            )
            entropy_threshold = (
                2.0 if getattr(self, "structured_multimodal", False) else 1.5
            )
            if entropy_increase_ratio > entropy_threshold:
                logging.debug(
                    f"[EntropyPenalty] Candidate {method}+{scaler_name} increased entropy by {entropy_increase_ratio:.2f}x; skipping."
                )
                return None

            return {
                "method": method,
                "scaler_name": scaler_name,
                "scaler": scaler,
                "transformer": self._get_transformer(method, x),
                **normality,
                "skew": skew_val,
                "kurtosis": stats.kurtosis(x_clean) if len(x_clean) > 2 else 0.0,
                "entropy": candidate_entropy,
            }

        except Exception as e:
            logging.debug(
                f"[CandidateError] {method}+{scaler_name} on '{col}' failed: {e}"
            )
            return None

    def _choose_fallback_scaler(
        self,
        x: np.ndarray,
        stats_dict: Dict[str, Any],
        num_modes: int,
        structured_multimodal: bool,
    ) -> Tuple[Any, str]:
        """
        Selects fallback scaler and provides explanation.
        Uses already computed statistics to avoid recomputation.
        """
        skew, kurtosis = stats_dict["skew"], stats_dict["kurtosis"]
        is_bounded = np.all((x >= 0) & (x <= 1))
        if is_bounded:
            fallback_scaler, reason = MinMaxScaler(), "bounded_0_1"
        elif structured_multimodal or num_modes >= 2:
            fallback_scaler, reason = (
                RobustScaler(),
                f"structured_multimodal (modes={num_modes})",
            )
        elif abs(skew) > 1.0 or abs(kurtosis) > 5:
            fallback_scaler, reason = (
                RobustScaler(),
                f"heavy_skew/kurtosis (skew={skew:.2f}, kurt={kurtosis:.2f})",
            )
        else:
            fallback_scaler, reason = (
                StandardScaler(),
                f"unimodal (skew={skew:.2f}, kurt={kurtosis:.2f})",
            )
        return fallback_scaler, reason

    def _save_report(self):
        df_report = pd.DataFrame.from_dict(self.report, orient="index")
        df_report.index.name = "feature"
        df_report.to_csv(self.report_file)

    def _process_column(self, series: pd.Series, col: str):
        x = series.dropna().values
        best_scaler, best_transformer = None, None
        fallback_reason = None
        self.structured_multimodal = False  # reset flag each column

        if len(x) < 3 or np.all(x == x[0]):
            logging.warning(
                f"[ColumnCheck] Column '{col}' has insufficient or constant data; skipping."
            )
            self.report[col] = {
                "status": "skipped_constant_or_insufficient",
                "num_nonnull": len(x),
                "constant_value": x[0] if len(x) > 0 else None,
                "reason": "Column has too few points or is constant; no transform applied.",
            }
            return col, None

        original_stats = self._compute_stats(x)
        original_entropy = original_stats["entropy"]
        distribution_type, dip_p = self._detect_distribution(original_stats, x)

        if len(x) < 50:
            num_modes, structured_multimodal = 1, False
        else:
            num_modes, structured_multimodal = self.estimate_modes_and_variance(x)
        self.structured_multimodal = structured_multimodal

        logging.info(
            f"[ModeCheck] Column '{col}': {num_modes} modes; {'Structured' if structured_multimodal else 'Unimodal/Overlapping'}."
        )

        candidates = []
        for method in self.extra_transforms + list(self.pre_funcs.keys()):
            if not self._is_valid_for(method, x):
                continue
            # Relax skip: even structured multimodal allows candidate, but we’ll penalize below
            candidate_scalers = (
                [("robust", RobustScaler()), ("minmax", MinMaxScaler())]
                if structured_multimodal
                else [
                    ("standard", StandardScaler()),
                    ("robust", RobustScaler()),
                    ("minmax", MinMaxScaler()),
                ]
            )
            for scaler_name, scaler in candidate_scalers:
                candidate = self._evaluate_candidate(
                    x, method, scaler, scaler_name, col, original_entropy
                )
                if candidate:
                    # Compute score
                    pval_score = (
                        2
                        if candidate["harmonic_p"] > 0.05
                        else 1 if candidate["harmonic_p"] > 0.01 else 0
                    )
                    skew_score = (
                        2
                        if abs(candidate["skew"]) < self.skew_thresh
                        else 1 if abs(candidate["skew"]) < self.skew_thresh * 1.5 else 0
                    )
                    entropy_ratio = (
                        (candidate["entropy"] / original_entropy) / np.log(30)
                        if original_entropy > 0
                        else 1.0
                    )
                    entropy_score = (
                        2 if entropy_ratio < 1.0 else 1 if entropy_ratio < 1.5 else 0
                    )
                    multimodal_penalty = 2 if structured_multimodal else 0
                    candidate["score"] = (
                        pval_score + skew_score + entropy_score - multimodal_penalty
                    )
                    candidate["entropy_ratio"] = entropy_ratio
                    candidates.append(candidate)

        # Filter extreme entropy blowups: skip only if entropy ratio > 3.0
        candidates = [c for c in candidates if c["entropy_ratio"] < 3.0]

        if candidates:
            candidates_sorted = sorted(
                candidates, key=lambda c: (-c["score"], abs(c["skew"]), c["entropy"])
            )
            best_metrics = candidates_sorted[0]
            best_scaler, best_transformer = (
                best_metrics["scaler"],
                best_metrics["transformer"],
            )
        else:
            # Fallback remains unchanged if no candidate has acceptable transform
            logging.warning(
                f"[Fallback] No valid candidates for '{col}'; applying fallback scaler + optional log1p."
            )
            if np.all(x >= 0):
                x_transformed = np.log1p(x)
                logging.info(
                    f"[Fallback] log1p applied before fallback scaler on '{col}'."
                )
                best_transformer = self.pre_funcs["log1p"]
            else:
                x_transformed = x
                best_transformer = None

            fallback_scaler, fallback_reason = self._choose_fallback_scaler(
                x, original_stats, num_modes, structured_multimodal
            )
            fallback_scaler.fit(x_transformed.reshape(-1, 1))
            best_scaler = fallback_scaler
            normality = self._normality_scores(x)
            best_metrics = {
                "method": (
                    "fallback_log1p+scaler"
                    if best_transformer == self.pre_funcs["log1p"]
                    else "fallback_scaler"
                ),
                "scaler_name": type(fallback_scaler).__name__,
                **normality,
                "skew": float(stats.skew(x)) if len(x) > 2 else 0.0,
                "kurtosis": float(stats.kurtosis(x)) if len(x) > 2 else 0.0,
                "entropy": stats.entropy(np.histogram(x, bins=30)[0] + 1),
                "score": -1,
            }
            logging.info(
                f"[FallbackScaler] Column fallback: {type(fallback_scaler).__name__} | Reason: {fallback_reason}"
            )
        if candidates:
            all_extreme_pvals = all(c["harmonic_p"] < 1e-4 for c in candidates)
        else:
            all_extreme_pvals = False  # ensure it's always defined

        if candidates and not all_extreme_pvals:
            best_scaler, best_transformer = (
                best_metrics["scaler"],
                best_metrics["transformer"],
            )

        already_normalizes = False
        if isinstance(best_transformer, (PowerTransformer, QuantileTransformer)) or (
            isinstance(best_transformer, tuple) and best_metrics["method"] == "boxcox"
        ):
            transformed_t = best_transformer.transform(x.reshape(-1, 1)).flatten()
            already_normalizes = True
        elif callable(best_transformer):
            transformed_t = best_transformer(x)
            already_normalizes = False
        elif best_transformer is None:
            transformed_t = x
            already_normalizes = False
        else:
            raise ValueError(f"Unknown transformer type: {best_transformer}")

        if best_scaler is None:
            logging.error(
                f"[Critical] No valid scaler found for column '{col}'. Aborting processing for this feature."
            )
            self.report[col] = {
                "status": "no_scaler_found",
                "reason": "All candidates and fallback failed; unable to determine scaler.",
            }
            return col, None

        if already_normalizes:
            transformed = transformed_t.reshape(-1, 1)
        else:
            transformed = best_scaler.transform(transformed_t.reshape(-1, 1))

        final_stats = self._compute_stats(transformed.flatten())
        final_normality = self._normality_scores(transformed.flatten())
        passes_normality = (
            final_normality["harmonic_p"] > self.alpha
            and abs(final_stats["skew"]) < self.skew_thresh
        )
        passes_normality = (
            final_normality["harmonic_p"] > relaxed_alpha
            and abs(final_stats["skew"]) < relaxed_skew_thresh
        )

        pass_flag = (
            "✅ Pass (post)"
            if passes_normality
            else "❌ Fail (post)" if not structured_multimodal else "❓ Ambiguous"
        )

        self._plot_before_after(col, x, transformed.flatten())
        report_entry = {
            "original_mean": original_stats["mean"],
            "original_std": original_stats["std"],
            "original_skew": original_stats["skew"],
            "original_kurtosis": original_stats["kurtosis"],
            "original_entropy": original_stats["entropy"],
            "dip_pvalue": dip_p,
            "chosen_method": best_metrics["method"],
            "chosen_scaler": (
                best_metrics["scaler_name"] if not already_normalizes else "skipped"
            ),
            "p_shapiro": best_metrics["p_shapiro"],
            "p_dagostino": best_metrics["p_dagostino"],
            "p_anderson": best_metrics["p_anderson"],
            "chosen_pval": best_metrics["harmonic_p"],
            "chosen_skew": best_metrics["skew"],
            "final_mean": final_stats["mean"],
            "final_std": final_stats["std"],
            "final_skew": final_stats["skew"],
            "final_kurtosis": final_stats["kurtosis"],
            "final_entropy": final_stats["entropy"],
            "num_modes": num_modes,
            "structured_multimodal": structured_multimodal,
            "post_pval": final_normality["harmonic_p"],
            "post_skew": final_stats["skew"],
            "pass_status": pass_flag,
            "detected_distribution": distribution_type,
            "fallback_reason": fallback_reason if not candidates else None,
        }
        if best_metrics["method"] == "boxcox" and isinstance(best_transformer, tuple):
            report_entry["boxcox_lambda"] = best_transformer[1]
        elif best_metrics["method"] == "yeo" and hasattr(best_transformer, "lambdas_"):
            report_entry["yeo_lambdas"] = best_transformer.lambdas_.tolist()
        else:
            report_entry["boxcox_lambda"], report_entry["yeo_lambdas"] = None, None

        logging.info(
            f"[Decision] Column: {col} | Detected: {distribution_type} | Method: {best_metrics['method']} + {'skipped' if already_normalizes else best_metrics['scaler_name']} (final p={final_normality['harmonic_p']:.4f}, skew={final_stats['skew']:.3f})"
        )
        return col, (
            best_scaler,
            best_transformer,
            best_metrics,
            candidates,
            report_entry,
        )

    @monitor(name="FeatureScalerTransformer Fit")
    def fit(self, df: pd.DataFrame) -> "FeatureScalerTransformer":
        self._fail_if_invalid(df)
        X_num = df.select_dtypes(include=[np.number])
        self.columns = list(X_num.columns)
        self._adaptive_thresholds(X_num.shape[0])

        logging.info("[GlobalCheck] Trying global scaler...")
        best_global_pval, best_scaler, pass_rate = 0, None, 0
        per_col_norms = []

        for scaler in self.available_scalers:
            try:
                X_scaled = scaler.fit_transform(X_num.fillna(X_num.mean()))
                pvals, skews = [], []
                for i, col in enumerate(self.columns):
                    arr = X_scaled[:, i]
                    normality = self._normality_scores(arr)
                    per_col_norms.append(normality)
                    pvals.append(normality["harmonic_p"])
                    sk = stats.skew(arr)
                    skews.append(abs(sk))

                avg_pval = np.mean(pvals)
                cols_passing = sum(
                    (np.array(pvals) > self.alpha)
                    & (np.array(skews) < self.skew_thresh)
                )
                rate = cols_passing / len(self.columns)
                if self.verbose:
                    logging.info(
                        f"[GlobalCheck] {type(scaler).__name__}: mean p={avg_pval:.4f}, pass={rate:.1%}"
                    )
                if rate > 0.9 and avg_pval > best_global_pval:
                    best_global_pval, best_scaler, pass_rate = avg_pval, scaler, rate
            except Exception as e:
                logging.warning(f"[GlobalCheck] {type(scaler).__name__} failed: {e}")

        if any((np.array(pvals) < 1e-4) | (np.array(skews) > 5)):
            logging.warning(
                "[GlobalCheck] Extreme stats found; overriding global scaler."
            )
            best_scaler = None

        if best_scaler:
            logging.info(
                f"[GlobalCheck] Global transform PASSED with {type(best_scaler).__name__}, {pass_rate:.1%} of columns normal/skew within threshold."
            )
            self.report["global_scaler"] = {
                "status": "global_scaler",
                "scaler_name": type(best_scaler).__name__,
                "mean_p_shapiro": np.mean([n["p_shapiro"] for n in per_col_norms]),
                "mean_p_dagostino": np.mean([n["p_dagostino"] for n in per_col_norms]),
                "mean_p_anderson": np.mean([n["p_anderson"] for n in per_col_norms]),
                "mean_harmonic_p": np.mean([n["harmonic_p"] for n in per_col_norms]),
                "mean_skew": np.mean(skews),
                "pass_rate": pass_rate,
            }
            self.global_scaler_ = best_scaler
            self._save_report()
            return self

        logging.info(
            "[GlobalCheck] Global transform failed or not suitable; falling back to per-column evaluation."
        )
        results = Parallel(n_jobs=-1)(
            delayed(self._process_column)(X_num[col], col) for col in self.columns
        )
        for col, res in results:
            if res:
                scaler, transformer, best_metrics, candidates, report_entry = res
                self.scalers_[col] = scaler
                self.transformers_[col] = transformer
                self.report[col] = report_entry
                self.report[col]["candidates"] = candidates

        self._save_report()
        return self

    @monitor(name="FeatureScalerTransformer Transform")
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        X_num = df_new.select_dtypes(include=[np.number])
        if self.global_scaler_:
            logging.info("[Transform] Applying global scaler to entire dataset.")
            transformed = self.global_scaler_.transform(X_num.fillna(X_num.mean()))
            df_new[X_num.columns] = transformed
            return df_new

        for col in self.columns:
            if col not in df_new.columns or col not in self.scalers_:
                continue

            x = df_new[col].values
            nonnull_mask = ~np.isnan(x)
            if not np.any(nonnull_mask):
                logging.warning(f"[Transform] Column '{col}' has all NaNs; skipping.")
                continue
            x_nonnull = x[nonnull_mask].reshape(-1, 1)

            best_transformer = self.transformers_[col]
            already_normalizes = isinstance(
                best_transformer, (PowerTransformer, QuantileTransformer)
            ) or (
                isinstance(best_transformer, tuple)
                and isinstance(self.scalers_[col], MinMaxScaler)
            )
            # Apply transformer if it exists
            if best_transformer:
                if (
                    isinstance(best_transformer, tuple)
                    and best_transformer[0] == "boxcox"
                ):
                    # Box-Cox transformation using saved lambda
                    x_t = boxcox(x_nonnull.flatten(), lmbda=best_transformer[1])
                    already_normalizes = True
                    logging.info(f"[Transform] Applied Box-Cox on '{col}'.")
                elif isinstance(
                    best_transformer, (PowerTransformer, QuantileTransformer)
                ):
                    # Scikit-learn transformers use .transform()
                    x_t = best_transformer.transform(x_nonnull).flatten()
                    already_normalizes = True
                    logging.info(
                        f"[Transform] Applied {type(best_transformer).__name__} on '{col}'."
                    )
                elif callable(best_transformer):
                    # This is the key: handle pre-function lambdas safely
                    x_t = best_transformer(x_nonnull.flatten())
                    already_normalizes = False
                    logging.info(
                        f"[Transform] Applied pre-function transformer on '{col}'."
                    )
                else:
                    raise ValueError(
                        f"[Transform] Unknown transformer type: {best_transformer}"
                    )
            else:
                logging.warning(
                    f"[Transform] Unexpected transformer type {type(best_transformer)}; skipping transformer."
                )
                x_t = x_nonnull.flatten()
                already_normalizes = False

            if already_normalizes:
                transformed_col = x_t
            else:
                transformed_col = (
                    self.scalers_[col].transform(x_t.reshape(-1, 1)).flatten()
                )
                logging.info(
                    f"[Transform] Applied {type(self.scalers_[col]).__name__} on '{col}'."
                )

            x_copy = x.copy()
            x_copy[nonnull_mask] = transformed_col
            df_new[col] = x_copy
        return df_new

    @monitor(name="FeatureScalerTransformer Fit-Transform")
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, filepath: str):
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "global_scaler": self.global_scaler_,
                    "scalers": self.scalers_,
                    "transformers": self.transformers_,
                    "columns": self.columns,
                    "report": self.report,
                },
                f,
            )

    def load(self, filepath: str) -> "FeatureScalerTransformer":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.global_scaler_ = data["global_scaler"]
            self.scalers_ = data["scalers"]
            self.transformers_ = data["transformers"]
            self.columns = data["columns"]
            self.report = data["report"]
        return self

    def generate_html_report(
        self, html_file: str = "scaler_transform_report.html"
    ) -> str:
        df = pd.read_csv(self.report_file)
        df["notes"] = np.where(
            df["chosen_method"].str.startswith("fallback"), "Fallback applied", "OK"
        )
        df["chart"] = df["feature"].apply(
            lambda f: (
                f'<a href="{os.path.join(VISUALS_DIR, f"{f}_before_after.png")}" target="_blank">View</a>'
                if os.path.exists(os.path.join(VISUALS_DIR, f"{f}_before_after.png"))
                else "N/A"
            )
        )
        columns_to_keep = [
            "pass_status",
            "fallback_reason",
            "detected_distribution",
            "feature",
            "chosen_method",
            "chosen_scaler",
            "p_shapiro",
            "p_dagostino",
            "p_anderson",
            "chosen_pval",
            "chosen_skew",
            "original_mean",
            "original_std",
            "original_skew",
            "original_kurtosis",
            "original_entropy",
            "final_mean",
            "final_std",
            "final_skew",
            "final_kurtosis",
            "final_entropy",
            "dip_pvalue",
            "boxcox_lambda",
            "yeo_lambdas",
            "notes",
            "chart",
        ]
        df_simple = df[columns_to_keep]
        html_table = df_simple.to_html(
            index=False,
            border=0,
            classes="table table-striped table-hover",
            escape=False,
            float_format="%.4f",
        )
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>FeatureScalerTransformer Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ padding: 20px; font-family: sans-serif; }}
        h1 {{ margin-bottom: 20px; }}
        .table {{ font-size: 0.9rem; }}
        .meta {{ margin-bottom: 20px; font-size: 0.95rem; }}
    </style>
</head>
<body>
    <h1>📊 FeatureScalerTransformer Report</h1>
    <div class="meta">
        <strong>Thresholds:</strong> alpha={self.alpha}, skew_thresh={self.skew_thresh} <br/>
        <strong>Features Processed:</strong> {len(df)} <br/>
        <strong>Visuals Directory:</strong> {VISUALS_DIR}
    </div>
    <p>This interactive table shows statistics, transformation decisions, pass/fail flags, and links to before/after charts.</p>
    {html_table}
</body>
</html>
"""
        with open(html_file, "w") as f:
            f.write(html_content)
        logging.info(f"[HTML Report] Generated → {html_file}")
        return html_content
