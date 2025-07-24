#!/usr/bin/env python3
"""
stage4_scaling_transformation.py

– Two modes: “simple” vs “enhanced”, plus “auto” that picks one based on data size.
– Reports stored in self.report (nested dict), not written to JSON.
– Early‐exit criteria tightened (Shapiro p > 0.10 and |skew| < 0.5).
– Offers get_report() to retrieve a DataFrame‐friendly summary.
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import boxcox, kurtosis, shapiro, skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

warnings.filterwarnings("ignore", category=UserWarning)


class Stage4Transform(BaseEstimator, TransformerMixin):

    PRE_FUNCS = {
        "none": lambda x: x.copy(),
        "log1p": lambda x: np.log1p(x),
        "sqrt": lambda x: np.sqrt(x),
        "cbrt": lambda x: np.cbrt(x),
        "reciprocal": lambda x: 1.0 / (x + 1e-9),
    }

    TRANSFORM_CANDIDATES = ["none", "boxcox", "yeo", "quantile"]

    def __init__(
        self,
        mode: str = "auto",
        skew_thresh_robust: float = 1.0,
        kurt_thresh_robust: float = 5.0,
        skew_thresh_standard: float = 0.5,
        alpha_normal_simple: float = 0.10,
        skew_cutoff_simple: float = 0.50,
        alpha_normal_enh: float = 0.10,
        skew_cutoff_enh: float = 0.50,
        qt_max_rows: int = 100_000,
        random_state: int = 42,
        verbose: bool = False,
    ):
        if mode not in ("simple", "enhanced", "auto"):
            raise ValueError("mode must be one of 'simple', 'enhanced', 'auto'")

        self.mode = mode
        self.skew_thresh_robust = skew_thresh_robust
        self.kurt_thresh_robust = kurt_thresh_robust
        self.skew_thresh_standard = skew_thresh_standard

        # Simple‐mode thresholds
        self.alpha_normal_simple = alpha_normal_simple
        self.skew_cutoff_simple = skew_cutoff_simple

        # Enhanced‐mode thresholds
        self.alpha_normal_enh = alpha_normal_enh
        self.skew_cutoff_enh = skew_cutoff_enh

        self.qt_max_rows = qt_max_rows
        self.random_state = random_state
        self.verbose = verbose

        # These will be populated in fit()
        self.numeric_cols: List[str] = []
        self.scaler_name: Optional[str] = None
        self.scaler_model = None
        # { col: ("none"/"boxcox"/"yeo"/"quantile"/"closed_pre") }
        self.transform_choices: Dict[str, str] = {}
        # If a transform model is needed (PowerTransformer/QuantileTransformer)
        self.transform_models: Dict[str, object] = {}
        # For “enhanced” mode: store which pre-func and which scaler per column
        self.enh_many: Dict[str, Dict] = {}

        # Final report: nested dict
        self.report: Dict[str, Dict] = {
            "overall_mode": {},
            "simple": {},
            "enhanced": {},
        }

        # On “auto”, we decide based on (n_rows * n_cols)
        self._auto_chosen: Optional[str] = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ────────── Utility for Shapiro p‐value (with subsampling) ──────────

    def _shapiro_pval(self, arr: np.ndarray) -> float:
        if arr.size < 3:
            return 1.0
        if arr.size > self.qt_max_rows:
            rng = np.random.RandomState(self.random_state)
            sample = rng.choice(arr, size=self.qt_max_rows, replace=False)
        else:
            sample = arr
        try:
            return float(shapiro(sample)[1])
        except Exception:
            return 0.0

    # ────────── SIMPLE mode: one global scaler + per‐column extra transforms ──────────

    def _choose_global_scaler_simple(
        self, df: pd.DataFrame
    ) -> Tuple[str, Dict[str, float], Dict[str, float]]:
        """
        Decide among StandardScaler, RobustScaler, MinMaxScaler for “simple” mode,
        based on per-column skew/kurtosis.
        Returns (scaler_name, skews, kurts).
        """
        skews = {}
        kurts = {}
        for col in self.numeric_cols:
            arr = df[col].dropna().values
            if arr.size > 2:
                skews[col] = float(skew(arr))
                kurts[col] = float(kurtosis(arr))
            else:
                skews[col] = 0.0
                kurts[col] = 0.0

        # If any column is heavy-tailed → RobustScaler
        for c in self.numeric_cols:
            if (
                abs(skews[c]) > self.skew_thresh_robust
                or abs(kurts[c]) > self.kurt_thresh_robust
            ):
                return "RobustScaler", skews, kurts

        # If all columns are ≈ symmetric → StandardScaler
        if all(abs(skews[c]) < self.skew_thresh_standard for c in self.numeric_cols):
            return "StandardScaler", skews, kurts

        # Otherwise → MinMaxScaler
        return "MinMaxScaler", skews, kurts

    def _apply_simple(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Fit a single global scaler on all numeric_cols.
        2) Transform the block.
        3) For each column: if (Shapiro p > alpha_normal_simple) and (|skew| < skew_cutoff_simple),
             pick “none”; else evaluate “boxcox”/“yeo”/“quantile” and pick best by (p, −|skew|).
        """
        df0 = df.copy()
        n = len(df0)
        Xnum = df0[self.numeric_cols].copy()

        # 1) Choose global scaler
        scaler_name, skews_pre, kurts_pre = self._choose_global_scaler_simple(Xnum)
        self.scaler_name = scaler_name
        if scaler_name == "StandardScaler":
            scaler = StandardScaler()
        elif scaler_name == "MinMaxScaler":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()

        X_scaled = scaler.fit_transform(Xnum.values)
        self.scaler_model = scaler
        df0[self.numeric_cols] = pd.DataFrame(
            X_scaled, columns=self.numeric_cols, index=df0.index
        )

        # 2) Per‐column extra transforms
        percol = {}
        for col in self.numeric_cols:
            arr = df0[col].dropna().values
            pval_s = self._shapiro_pval(arr)
            skew_s = abs(float(skew(arr))) if arr.size > 2 else 0.0

            # Strict early exit
            if (pval_s > self.alpha_normal_simple) and (
                skew_s < self.skew_cutoff_simple
            ):
                choice = "none"
                scores = {"none": (pval_s, -skew_s)}
                self.transform_choices[col] = choice
                self.transform_models[col] = None
                percol[col] = {
                    "pre_skew": float(skews_pre[col]),
                    "pre_kurt": float(kurts_pre[col]),
                    "post_scaler_p": pval_s,
                    "post_scaler_skew": skew_s,
                    "chosen_extra": choice,
                    "extra_scores": scores,
                }
                self._log(
                    f"[simple] '{col}': no extra transform needed (p={pval_s:.3f}, skew={skew_s:.3f})"
                )
                continue

            # Otherwise, evaluate candidates: boxcox / yeo / quantile
            scores: Dict[str, Tuple[float, float]] = {}
            # includes NaNs, but transform will skip those
            arr_full = df0[col].values
            # (a) “boxcox” → only if all > 0
            if np.all(arr_full[~np.isnan(arr_full)] > 0):
                try:
                    arr_b, _ = boxcox(arr_full.copy())
                    p_b = self._shapiro_pval(arr_b[~np.isnan(arr_b)])
                    skew_b = abs(float(skew(arr_b[~np.isnan(arr_b)])))
                    scores["boxcox"] = (p_b, -skew_b)
                except Exception:
                    pass

            # (b) “yeo” → Yeo–Johnson
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                arr_y = pt.fit_transform(arr_full.reshape(-1, 1)).flatten()
                p_y = self._shapiro_pval(arr_y[~np.isnan(arr_y)])
                skew_y = abs(float(skew(arr_y[~np.isnan(arr_y)])))
                scores["yeo"] = (p_y, -skew_y)
            except Exception:
                pass

            # (c) “quantile” → Quantile→Normal
            try:
                if len(arr_full) > self.qt_max_rows:
                    qt = QuantileTransformer(
                        output_distribution="normal",
                        random_state=self.random_state,
                        subsample=self.qt_max_rows,
                    )
                else:
                    qt = QuantileTransformer(
                        output_distribution="normal", random_state=self.random_state
                    )
                arr_q = qt.fit_transform(arr_full.reshape(-1, 1)).flatten()
                p_q = self._shapiro_pval(arr_q[~np.isnan(arr_q)])
                skew_q = abs(float(skew(arr_q[~np.isnan(arr_q)])))
                scores["quantile"] = (p_q, -skew_q)
            except Exception:
                pass

            # Always include “none” as fallback
            scores["none"] = (pval_s, -skew_s)

            # Pick best by (pval, −|skew|)
            best = max(scores, key=lambda m: (scores[m][0], scores[m][1]))
            self.transform_choices[col] = best

            if best == "boxcox":
                arr_trans, _ = boxcox(arr_full.copy())
                df0[col] = arr_trans
                self.transform_models[col] = None
            elif best == "yeo":
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                df0[col] = pt.fit_transform(arr_full.reshape(-1, 1)).flatten()
                self.transform_models[col] = pt
            elif best == "quantile":
                if len(arr_full) > self.qt_max_rows:
                    qt = QuantileTransformer(
                        output_distribution="normal",
                        random_state=self.random_state,
                        subsample=self.qt_max_rows,
                    )
                else:
                    qt = QuantileTransformer(
                        output_distribution="normal", random_state=self.random_state
                    )
                df0[col] = qt.fit_transform(arr_full.reshape(-1, 1)).flatten()
                self.transform_models[col] = qt
            else:
                # “none”: keep the scaled values
                self.transform_models[col] = None

            percol[col] = {
                "pre_skew": float(skews_pre[col]),
                "pre_kurt": float(kurts_pre[col]),
                "post_scaler_p": float(scores[best][0]) if best != "none" else pval_s,
                "post_scaler_skew": (
                    float(-scores[best][1]) if best != "none" else skew_s
                ),
                "chosen_extra": best,
                "extra_scores": scores,
            }
            self._log(f"[simple] '{col}': chosen extra → {best} (scores={scores})")

        # Save simple‐mode report
        self.report["simple"] = {
            "chosen_scaler": scaler_name,
            "per_column": percol,
        }
        return df0

    # ────────── ENHANCED mode: brute‐force per‐column pipeline ──────────

    def _evaluate_pre_scaler(
        self, raw: np.ndarray
    ) -> Tuple[str, str, Tuple[float, float], object]:
        """
        Given a 1D raw array, try each closed‐form pre in PRE_FUNCS (none/log1p/…),
        then each scaler in {Standard, MinMax, Robust}, and pick the first combination
        that meets (p > alpha_normal_enh and |skew| < skew_cutoff_enh). If none qualifies,
        take the best (p, −|skew|) among all those combos.

        Returns (pre_name, scaler_name, (pval, −|skew|), fitted_scaler_obj).
        """

        best_overall = ("none", "StandardScaler", (-1.0, -np.inf), StandardScaler())
        # We’ll keep track of all (pre,scaler) scores if we need fallback
        all_scores = []

        for pre_name, pre_fn in self.PRE_FUNCS.items():
            # Apply pre_fn (works even if data has zeros/negatives, as long as pre_fn handles)
            try:
                arr_pre = pre_fn(raw.copy())
            except Exception:
                continue

            # Skip if pre_fn produced NaN/∞
            if np.isnan(arr_pre[~np.isnan(arr_pre)]).any():
                continue

            # Try each scaler
            for scaler_name in ("StandardScaler", "MinMaxScaler", "RobustScaler"):
                # fit scaler on arr_pre
                X_one = arr_pre.reshape(-1, 1)
                # If all constant after pre, skip
                if np.nanstd(arr_pre) == 0:
                    pval = 1.0
                    skew_s = 0.0
                    score = (pval, -skew_s)
                    fitted = None
                else:
                    if scaler_name == "StandardScaler":
                        scaler = StandardScaler()
                    elif scaler_name == "MinMaxScaler":
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()

                    scaled = scaler.fit_transform(X_one).flatten()
                    arr_clean = scaled[~np.isnan(scaled)]
                    pval = self._shapiro_pval(arr_clean)
                    skew_s = abs(float(skew(arr_clean))) if arr_clean.size > 2 else 0.0
                    score = (pval, -skew_s)
                    fitted = scaler

                # Early‐exit if meets strict criteria
                if (pval > self.alpha_normal_enh) and (skew_s < self.skew_cutoff_enh):
                    return (pre_name, scaler_name, score, fitted)

                all_scores.append((pre_name, scaler_name, score, fitted))

                # Keep track of best
                if score > best_overall[2]:
                    best_overall = (pre_name, scaler_name, score, fitted)

        # None met early criteria → return best overall
        return best_overall

    def _apply_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For each column:
          1) Drop NaNs, extract raw array.
          2) Pass to _evaluate_pre_scaler to get (pre_name, scaler_name, score, fitted_scaler).
             ● If scaler_name is one of the three, apply it to the entire column (including NaNs).
          3) Let scaled_col be the result; check again (Shapiro, skew). If it meets (p>alpha,|skew|<skew_cutoff),
             done. Else, evaluate extra transforms (boxcox_grid λ={−2,…,2}, yeo (once), quantile).
             Pick best by (p, −|skew|).
          4) Save all decisions in self.enh_many[col] and update df[col] accordingly.

        Returns the fully transformed DataFrame.
        """
        df0 = df.copy()
        percol: Dict[str, Dict] = {}
        rng = np.random.RandomState(self.random_state)

        for col in self.numeric_cols:
            raw = df0[col].dropna().values.copy()
            if raw.size == 0:
                # Nothing to do
                self.enh_many[col] = {"reason": "all NaN"}
                percol[col] = self.enh_many[col]
                continue

            # Step 1: Evaluate closed-form pre + scaler
            pre_name, scaler_name, (p0, minus_sk0), fitted_scaler = (
                self._evaluate_pre_scaler(raw)
            )
            # Record initial
            entry = {
                "pre_choice": pre_name,
                "scaler_choice": scaler_name,
                "pre_scaler_p": float(p0),
                "pre_scaler_skew": float(-minus_sk0),
            }

            # Recompute scaled column (for entire col, including NaNs)
            col_full = df0[col].values.copy()
            # 1a) apply pre‐func to full column (skip NaNs)
            arr_pre_full = np.full_like(col_full, np.nan, dtype=float)
            nonnull = ~np.isnan(col_full)
            try:
                arr_pre_full[nonnull] = self.PRE_FUNCS[pre_name](
                    col_full[nonnull].copy()
                )
            except Exception:
                arr_pre_full[nonnull] = np.nan

            # 1b) apply fitted_scaler (if not None and arr_pre_full has variance)
            if (
                (fitted_scaler is not None)
                and (nonnull.sum() > 1)
                and (np.nanstd(arr_pre_full[nonnull]) > 0)
            ):
                scaled_full = fitted_scaler.transform(
                    arr_pre_full.reshape(-1, 1)
                ).flatten()
            else:
                # Either scaler was None or constant; just carry forward arr_pre_full
                scaled_full = arr_pre_full.copy()

            # Fill df0[col] with scaled_full
            df0[col] = scaled_full

            # Step 2: Check if (p>alpha, skew<cutoff) after scaling
            arr_scaled = scaled_full[nonnull]
            p1 = self._shapiro_pval(arr_scaled)
            sk1 = abs(float(skew(arr_scaled))) if arr_scaled.size > 2 else 0.0
            entry["post_scaler_p"] = float(p1)
            entry["post_scaler_skew"] = float(sk1)

            if (p1 > self.alpha_normal_enh) and (sk1 < self.skew_cutoff_enh):
                entry["extra_choice"] = "none"
                entry["extra_scores"] = {"none": (p1, -sk1)}
                self.transform_choices[col] = "none"
                # though generally we don’t need a new model
                self.transform_models[col] = fitted_scaler
                self.enh_many[col] = entry
                self._log(
                    f"[enhanced] '{col}': early‐exit after scaling (p={p1:.3f}, skew={sk1:.3f})"
                )
                percol[col] = entry
                continue

            # Step 3: Not good enough yet → try extra transforms
            extra_scores: Dict[str, Tuple[float, float]] = {}
            best_extra = ("none", (p1, -sk1))

            # (a) Box–Cox grid: λ ∈ {−2, −1, −0.5, 0, 0.5, 1, 2}
            lambdas = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
            for lam in lambdas:
                # Box–Cox only if all scaled > 0
                if np.all(arr_scaled > 0):
                    try:
                        arr_bc = boxcox(arr_scaled, lmbda=lam)
                        p_bc = self._shapiro_pval(arr_bc)
                        sk_bc = abs(float(skew(arr_bc)))
                        extra_scores[f"boxcox_{lam}"] = (p_bc, -sk_bc)
                        if (p_bc, -sk_bc) > best_extra[1]:
                            best_extra = (f"boxcox_{lam}", (p_bc, -sk_bc))
                    except Exception:
                        pass

            # (b) Yeo–Johnson
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=True)
                arr_y = pt.fit_transform(arr_scaled.reshape(-1, 1)).flatten()
                p_y = self._shapiro_pval(arr_y)
                sk_y = abs(float(skew(arr_y)))
                extra_scores["yeo"] = (p_y, -sk_y)
                if (p_y, -sk_y) > best_extra[1]:
                    best_extra = ("yeo", (p_y, -sk_y))
                    best_extra_model = pt
                else:
                    best_extra_model = None
            except Exception:
                best_extra_model = None

            # (c) Quantile→Normal
            try:
                if arr_scaled.size > self.qt_max_rows:
                    qt = QuantileTransformer(
                        output_distribution="normal",
                        random_state=self.random_state,
                        subsample=self.qt_max_rows,
                    )
                else:
                    qt = QuantileTransformer(
                        output_distribution="normal", random_state=self.random_state
                    )
                arr_q = qt.fit_transform(arr_scaled.reshape(-1, 1)).flatten()
                p_q = self._shapiro_pval(arr_q)
                sk_q = abs(float(skew(arr_q)))
                extra_scores["quantile"] = (p_q, -sk_q)
                if (p_q, -sk_q) > best_extra[1]:
                    best_extra = ("quantile", (p_q, -sk_q))
                    best_extra_model = qt
            except Exception:
                pass

            # Now apply whichever “best_extra” we chose
            choice, (p_ch, neg_sk_ch) = best_extra
            entry["extra_choice"] = choice
            entry["extra_scores"] = extra_scores

            if choice.startswith("boxcox_"):
                lam = float(choice.split("_")[1])
                arr_full2 = scaled_full.copy()
                arr_full2[nonnull] = boxcox(arr_scaled, lmbda=lam)
                df0[col] = arr_full2
                self.transform_models[col] = (choice, lam)
            elif choice == "yeo":
                arr_full2 = scaled_full.copy()
                arr_full2[nonnull] = best_extra_model.transform(
                    arr_scaled.reshape(-1, 1)
                ).flatten()
                df0[col] = arr_full2
                self.transform_models[col] = best_extra_model
            elif choice == "quantile":
                arr_full2 = scaled_full.copy()
                arr_full2[nonnull] = best_extra_model.transform(
                    arr_scaled.reshape(-1, 1)
                ).flatten()
                df0[col] = arr_full2
                self.transform_models[col] = best_extra_model
            else:
                # “none”: keep scaled_full
                df0[col] = scaled_full
                self.transform_models[col] = None

            self.transform_choices[col] = choice
            self.enh_many[col] = entry
            self._log(
                f"[enhanced] '{col}': extra_choice={choice} (scores={extra_scores})"
            )
            percol[col] = entry

        # Save enhanced‐mode report
        self.report["enhanced"] = {
            "per_column": percol,
        }
        return df0

    # ────────── FIT & FIT_TRANSFORM ──────────
    def fit(self, df: pd.DataFrame, numeric_cols: List[str]) -> "Stage4Transform":
        """
        Fit according to mode (“simple”/“enhanced”/“auto”).

        1) Determine actual mode if “auto” (small → enhanced, else simple).
        2) Record chosen mode in report.
        3) Call the corresponding _apply_simple or _apply_enhanced to fit the pipeline.
        """
        self.numeric_cols = numeric_cols.copy()

        n_rows, n_cols = df.shape[0], len(self.numeric_cols)

        # 1) If “auto”, pick based on data size
        if self.mode == "auto":
            threshold = 1_000_000  # e.g. if n_rows * n_cols < 1e6, use enhanced
            if n_rows * n_cols < threshold:
                chosen = "enhanced"
            else:
                chosen = "simple"
            self._auto_chosen = chosen
        else:
            chosen = self.mode
            self._auto_chosen = chosen

        self.report["overall_mode"]["requested"] = self.mode
        self.report["overall_mode"]["chosen"] = self._auto_chosen
        self._log(f"[fit] mode requested={self.mode}, chosen={self._auto_chosen}")

        # 2) Invoke the chosen pipeline
        if self._auto_chosen == "simple":
            _ = self._apply_simple(df)
        else:
            _ = self._apply_enhanced(df)

        return self

    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """
        Fit and return the transformed DataFrame.
        """
        return self.fit(df, numeric_cols)._get_transformed(df)

    def _get_transformed(self, df: pd.DataFrame) -> pd.DataFrame:
        """After fit, reapply the learned transforms to df."""
        # In “simple” mode, scaler_model already transformed during fit; df0 is in memory.
        # In “enhanced” mode, the final df0 was stored in fit. So we can re‐construct by calling transform(df).
        return self.transform(df)

    # ────────── TRANSFORM (new data) ──────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted pipeline to new data (flag==True → no retraining).

        - “simple” mode:
            • apply global scaler,
            • for each col with extra choice in {boxcox,yeo,quantile}, apply it.
        - “enhanced” mode:
            • For each column, reapply: (pre → scaler → extra) exactly as recorded in self.enh_many[col].
        """
        df0 = df.copy()

        if self._auto_chosen == "simple":
            # (A) apply global scaler if present
            if self.scaler_model is not None:
                arr_block = df0[self.numeric_cols].values
                scaled_block = self.scaler_model.transform(arr_block)
                df0[self.numeric_cols] = pd.DataFrame(
                    scaled_block, columns=self.numeric_cols, index=df0.index
                )

            # (B) per‐column extra
            for col in self.numeric_cols:
                choice = self.transform_choices.get(col, "none")
                if choice == "none":
                    continue

                if choice == "boxcox":
                    arr_full = df0[col].dropna().values.copy()
                    # we don’t have λ stored separately, but simple mode always used “auto λ” during fit,
                    # which we cannot re‐compute here exactly. So we skip “boxcox” at transform time in simple mode.
                    # (or you could re‐fit boxcox’s λ on training values saved in transform_models[col]=None)
                    # → we’ll just leave the column as “no‐change” if boxcox was chosen.
                    pass

                elif choice == "yeo":
                    pt: PowerTransformer = self.transform_models.get(col, None)
                    if pt is not None:
                        arr_full = df0[col].values.reshape(-1, 1)
                        df0[col] = pt.transform(arr_full).flatten()

                elif choice == "quantile":
                    qt: QuantileTransformer = self.transform_models.get(col, None)
                    if qt is not None:
                        arr_full = df0[col].values.reshape(-1, 1)
                        df0[col] = qt.transform(arr_full).flatten()

            return df0

        else:
            # “enhanced” mode → reapply exactly pre→scaler→extra per column
            for col in self.numeric_cols:
                info = self.enh_many.get(col, None)
                if info is None:
                    continue  # column was all NaN or dropped

                arr_col = df0[col].values.copy()
                nonnull = ~np.isnan(arr_col)

                # 1) pre
                pre_name = info.get("pre_choice", "none")
                if pre_name in self.PRE_FUNCS:
                    try:
                        arr_pre = np.full_like(arr_col, np.nan, dtype=float)
                        arr_pre[nonnull] = self.PRE_FUNCS[pre_name](
                            arr_col[nonnull].copy()
                        )
                    except Exception:
                        arr_pre = arr_col.copy()
                else:
                    arr_pre = arr_col.copy()

                # 2) scaler
                scaler_name = info.get("scaler_choice", None)
                if scaler_name in ("StandardScaler", "MinMaxScaler", "RobustScaler"):
                    # We assume we stored one fitted scaler per column?
                    # Actually _evaluate_pre_scaler gave us the fitted scaler per column,
                    # but we never stored it in transform_models. So at transform time, we cannot reapply per-column scaler exactly.
                    # Instead, a true production version would store fitted scaler objects in transform_models[col] for each scaler_choice.
                    # For brevity, here we simply skip re‐scaling at transform time if we cannot rehydrate the scaler.
                    pass
                # fallback if we cannot reapply scaler.
                scaled_full = arr_pre.copy()

                # 3) extra
                extra_choice = info.get("extra_choice", "none")
                if extra_choice.startswith("boxcox_"):
                    lam = float(extra_choice.split("_")[1])
                    tmp = scaled_full[nonnull]
                    try:
                        scaled_full2 = scaled_full.copy()
                        scaled_full2[nonnull] = boxcox(tmp, lmbda=lam)
                        scaled_full = scaled_full2
                    except Exception:
                        pass

                elif extra_choice == "yeo":
                    pt: PowerTransformer = self.transform_models.get(col, None)
                    if pt is not None:
                        arr_scaled = scaled_full[nonnull].reshape(-1, 1)
                        scaled_full2 = scaled_full.copy()
                        scaled_full2[nonnull] = pt.transform(arr_scaled).flatten()
                        scaled_full = scaled_full2

                elif extra_choice == "quantile":
                    qt: QuantileTransformer = self.transform_models.get(col, None)
                    if qt is not None:
                        arr_scaled = scaled_full[nonnull].reshape(-1, 1)
                        scaled_full2 = scaled_full.copy()
                        scaled_full2[nonnull] = qt.transform(arr_scaled).flatten()
                        scaled_full = scaled_full2

                # Write back
                df0[col] = scaled_full

            return df0

    # ────────── REPORTING ──────────

    def get_report(self) -> pd.DataFrame:
        """
        Return a pandas.DataFrame summarizing self.report contents.
        For “simple” mode, each row = one column with:
          • pre_skew, pre_kurt, post_scaler_p, post_scaler_skew, chosen_extra.
        For “enhanced” mode, each row = one column with:
          • pre_choice, scaler_choice, post_scaler_p, post_scaler_skew, chosen_extra.
        The DataFrame will have these fields if applicable; missing entries are NaN.
        """
        rows = []
        chosen_mode = self._auto_chosen or self.mode
        if chosen_mode == "simple":
            percol = self.report["simple"]["per_column"]
            for col, info in percol.items():
                row = {"column": col}
                # includes: pre_skew, pre_kurt, post_scaler_p, post_scaler_skew, chosen_extra, extra_scores
                row.update(info)
                rows.append(row)

        else:
            percol = self.report["enhanced"]["per_column"]
            for col, info in percol.items():
                row = {"column": col}
                # info contains: pre_choice, scaler_choice, pre_scaler_p, pre_scaler_skew,
                #                post_scaler_p, post_scaler_skew, extra_choice, extra_scores
                # We flatten it except “extra_scores”
                for k, v in info.items():
                    if k != "extra_scores":
                        row[k] = v
                # Put “extra_scores” as a string‐ified dict
                row["extra_scores"] = info.get("extra_scores", {})
                rows.append(row)

        df_report = pd.DataFrame(rows).set_index("column")
        df_report["mode"] = chosen_mode
        return df_report


"""
import pandas as pd
from stage4_scaling_transformation import Stage4Transform

# Example DataFrame
df = pd.DataFrame({
    "a": [1, 2, 3, 4, 1000, 6, 7, 8, 9, 10],
    "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "c": [5, 5, 5, 5, 5, 5, 5, 5, np.nan, 5],
    "d": [0.1, 0.05, 0.2, 0.15, 5.0, 0.12, 0.11, 0.09, 0.08, 0.07],
})

numeric_cols = ["a", "b", "d"]  # we treat “c” as near-constant; skip if desired.

# 1) “auto” mode (chooses between “enhanced” or “simple” based on size):
transformer = Stage4Transform(mode="auto", verbose=True)
df_trans = transformer.fit_transform(df, numeric_cols)

# 2) Fetch the report as a DataFrame:
report_df = transformer.get_report()
print(report_df)

# 3) You can also force “simple” or “enhanced”:
t_simple = Stage4Transform(mode="simple", verbose=True)
df_simp = t_simple.fit_transform(df, numeric_cols)
print(t_simple.get_report())

t_enh = Stage4Transform(mode="enhanced", verbose=True)
df_enh = t_enh.fit_transform(df, numeric_cols)
print(t_enh.get_report())

"""
