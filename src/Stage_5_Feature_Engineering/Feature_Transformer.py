# Filename: smart_feature_transformer.py

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, shapiro, boxcox
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
    QuantileTransformer,
)
from sklearn.base import BaseEstimator, TransformerMixin
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any
import warnings
from src.utils.scripts.monitor import monitor

warnings.filterwarnings("ignore", category=UserWarning)


class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Intelligent feature transformer with 'adaptive', 'simple', or 'enhanced' modes.
    Applies optimal scaler + transformation per column or globally using statistical tests.
    """

    PRE_FUNCS = {
        "none": lambda x: x.copy(),
        "log1p": lambda x: np.log1p(x),
        "sqrt": lambda x: np.sqrt(x),
        "cbrt": lambda x: np.cbrt(x),
        "reciprocal": lambda x: 1.0 / (x + 1e-9),
    }

    EXTRA_TRANSFORMS = ["none", "boxcox", "yeo", "quantile"]

    def __init__(
        self,
        mode: str = "adaptive",
        alpha: float = 0.10,
        skew_thresh: float = 0.5,
        qt_max_rows: int = 100_000,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.mode = mode
        self.alpha = alpha
        self.skew_thresh = skew_thresh
        self.qt_max_rows = qt_max_rows
        self.random_state = random_state
        self.verbose = verbose

        self.columns: List[str] = []
        self.plan: Dict[str, Dict] = {}
        self._global_transform: Optional[Dict[str, Any]] = None

    def _log(self, msg):
        if self.verbose:
            print(f"[SmartFeatureTransformer] {msg}")

    def _shapiro(self, x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        if len(x) < 3:
            return 1.0
        if len(x) > self.qt_max_rows:
            x = np.random.RandomState(self.random_state).choice(
                x, self.qt_max_rows, replace=False
            )
        try:
            return shapiro(x)[1]
        except Exception:
            return 0.0

    def _is_valid_for(self, method: str, x: np.ndarray) -> bool:
        x = x[~np.isnan(x)]
        if method == "boxcox":
            return np.all(x > 0)
        elif method in ("sqrt", "log1p"):
            return np.all(x >= 0)
        elif method == "reciprocal":
            return not np.any(np.isclose(x, 0))
        return True

    def _evaluate_candidate(
        self, x: np.ndarray, method: str, scaler_name: str
    ) -> Optional[Dict]:
        try:
            if not self._is_valid_for(method, x):
                return None
            if method == "none":
                x_t = x.copy()
                transformer = None
            elif method == "boxcox":
                x_t, lmbda = boxcox(x)
                transformer = ("boxcox", lmbda)
            elif method == "yeo":
                pt = PowerTransformer(method="yeo-johnson")
                x_t = pt.fit_transform(x.reshape(-1, 1)).flatten()
                transformer = pt
            elif method == "quantile":
                qt = QuantileTransformer(
                    output_distribution="normal",
                    subsample=self.qt_max_rows,
                    random_state=self.random_state,
                )
                x_t = qt.fit_transform(x.reshape(-1, 1)).flatten()
                transformer = qt
            elif method in self.PRE_FUNCS:
                x_t = self.PRE_FUNCS[method](x)
                transformer = None
            else:
                return None

            scaler = {
                "standard": StandardScaler(),
                "robust": RobustScaler(),
                "minmax": MinMaxScaler(),
            }.get(scaler_name)
            x_scaled = scaler.fit_transform(x_t.reshape(-1, 1)).flatten()
            x_clean = x_scaled[~np.isnan(x_scaled)]

            pval = self._shapiro(x_clean)
            skew_val = abs(float(skew(x_clean))) if len(x_clean) > 2 else 0.0

            return {
                "score": (pval, -skew_val),
                "pval": pval,
                "skew": skew_val,
                "method": method,
                "scaler_name": scaler_name,
                "scaler": scaler,
                "transformer": transformer,
            }

        except Exception as e:
            self._log(f"[ERROR] {method}+{scaler_name} failed: {e}")
            return None

    def _evaluate_global(self, df: pd.DataFrame) -> Optional[Dict]:
        x = df.dropna().values
        results = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._evaluate_candidate, x.flatten(), method, scaler)
                for method in self.EXTRA_TRANSFORMS + list(self.PRE_FUNCS)
                for scaler in ["standard", "robust", "minmax"]
            ]
            for f in as_completed(futures):
                r = f.result()
                if r:
                    results.append(r)
        if not results:
            return None
        best = max(results, key=lambda r: r["score"])
        # Check overall performance on each column
        transformed = best["scaler"].fit_transform(df.values)
        df_t = pd.DataFrame(transformed, columns=df.columns)
        pass_count = 0
        for col in df_t.columns:
            arr = df_t[col].dropna().values
            if len(arr) < 3:
                continue
            pval = self._shapiro(arr)
            sk = abs(float(skew(arr))) if len(arr) > 2 else 0.0
            if (pval > self.alpha) and (sk < self.skew_thresh):
                pass_count += 1
        score = pass_count / df_t.shape[1]
        if score >= 0.90:  # 90% columns pass
            self._log(f"Global transform PASSED for {score:.2%} columns.")
            return best
        return None

    @monitor(
        name="SFT.fit",
        log_args=True,
        track_memory=True,
        track_input_size=True,
        retries=1,
    )
    def fit(self, df: pd.DataFrame, numeric_cols: List[str]):
        self.columns = numeric_cols.copy()
        df_num = df[numeric_cols].copy()
        if self.mode == "adaptive":
            global_candidate = self._evaluate_global(df_num)
            if global_candidate:
                self._global_transform = global_candidate
                self._log("Using GLOBAL transform.")
                return self
        # Per-column fallback
        with ThreadPoolExecutor() as executor:
            for col in numeric_cols:
                x = df[col].dropna().values
                if len(x) < 3:
                    continue
                futures = [
                    executor.submit(self._evaluate_candidate, x.copy(), method, scaler)
                    for method in self.EXTRA_TRANSFORMS + list(self.PRE_FUNCS)
                    for scaler in ["standard", "robust", "minmax"]
                    if self._is_valid_for(method, x)
                ]
                candidates = [f.result() for f in as_completed(futures) if f.result()]
                if candidates:
                    best = max(candidates, key=lambda r: r["score"])
                    self.plan[col] = best
                    self._log(f"[{col}] best: {best['method']} + {best['scaler_name']}")
        return self

    @monitor(name="SFT.transform", log_args=False, track_memory=True, force_gpu=True)
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        if self._global_transform:
            method = self._global_transform["method"]
            transformer = self._global_transform["transformer"]
            scaler = self._global_transform["scaler"]
            x = df_new[self.columns].values
            if method == "none":
                x_t = x
            elif method == "boxcox":
                x_t = np.apply_along_axis(
                    lambda c: (
                        boxcox(c[~np.isnan(c)])[0] if np.all(c[~np.isnan(c)] > 0) else c
                    ),
                    axis=0,
                    arr=x,
                )
            elif method in self.PRE_FUNCS:
                x_t = np.apply_along_axis(self.PRE_FUNCS[method], axis=0, arr=x)
            else:
                x_t = transformer.transform(x)
            df_new[self.columns] = scaler.transform(x_t)
        else:
            for col in self.columns:
                if col not in self.plan:
                    continue
                x = df_new[col].values
                nonnull = ~np.isnan(x)
                method = self.plan[col]["method"]
                transformer = self.plan[col]["transformer"]
                scaler = self.plan[col]["scaler"]
                if method == "none":
                    x_t = x.copy()
                elif method == "boxcox":
                    x_t = x.copy()
                    x_t[nonnull] = boxcox(x[nonnull], lmbda=transformer[1])
                elif method in self.PRE_FUNCS:
                    x_t = self.PRE_FUNCS[method](x)
                else:
                    x_t = transformer.transform(x.reshape(-1, 1)).flatten()
                df_new[col] = scaler.transform(x_t.reshape(-1, 1)).flatten()
        return df_new

    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        return self.fit(df, numeric_cols).transform(df)

    def get_plan(self) -> Dict[str, Any]:
        return {
            "global": self._global_transform,
            "columns": {
                k: {
                    "method": v["method"],
                    "scaler": v["scaler_name"],
                    "pval": v["pval"],
                    "skew": v["skew"],
                }
                for k, v in self.plan.items()
            },
        }

    def get_report(self) -> pd.DataFrame:
        rows = []
        for col, plan in self.plan.items():
            row = {"column": col}
            row.update(plan["metrics"])
            row["method"] = plan["method"]
            row["scaler"] = plan["scaler_name"]
            rows.append(row)
        return pd.DataFrame(rows).set_index("column")


"""
# Filename: smart_feature_transformer.py

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, shapiro, boxcox
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    PowerTransformer, QuantileTransformer
)
from sklearn.base import BaseEstimator, TransformerMixin
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, List, Any, Tuple
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class SmartFeatureTransformer(BaseEstimator, TransformerMixin):

    PRE_FUNCS = {
        "none": lambda x: x.copy(),
        "log1p": lambda x: np.log1p(x),
        "sqrt": lambda x: np.sqrt(x),
        "cbrt": lambda x: np.cbrt(x),
        "reciprocal": lambda x: 1.0 / (x + 1e-9),
    }

    EXTRA_TRANSFORMS = ["none", "boxcox", "yeo", "quantile"]

    def __init__(
        self,
        mode: str = "enhanced",  # 'simple', 'enhanced', or 'auto'
        alpha: float = 0.10,
        skew_thresh: float = 0.5,
        qt_max_rows: int = 100_000,
        skew_thresh_robust: float = 1.0,
        kurt_thresh_robust: float = 5.0,
        skew_thresh_standard: float = 0.5,
        alpha_normal_simple: float = 0.10,
        skew_cutoff_simple: float = 0.50,
        alpha_normal_enh: float = 0.10,
        skew_cutoff_enh: float = 0.50,
        random_state: int = 42,
        verbose: bool = False
    ):
        assert mode in ("simple", "enhanced", "auto")
        self.mode = mode
        self.alpha = alpha
        self.skew_thresh = skew_thresh
        self.qt_max_rows = qt_max_rows
        self.random_state = random_state
        self.verbose = verbose
        self.skew_thresh_robust = skew_thresh_robust
        self.kurt_thresh_robust = kurt_thresh_robust
        self.skew_thresh_standard = skew_thresh_standard
        self.alpha_normal_simple = alpha_normal_simple
        self.skew_cutoff_simple = skew_cutoff_simple
        self.alpha_normal_enh = alpha_normal_enh
        self.skew_cutoff_enh = skew_cutoff_enh

        self.plan: Dict[str, Dict] = {}
        self.report: Dict[str, Dict] = {}
        self.columns: List[str] = []
        self._auto_chosen: Optional[str] = None

    def _log(self, msg: str):
        if self.verbose:
            print(f"[SmartFeatureTransformer] {msg}")

    def _shapiro(self, x: np.ndarray) -> float:
        x = x[~np.isnan(x)]
        if len(x) < 3:
            return 1.0
        if len(x) > self.qt_max_rows:
            x = np.random.RandomState(self.random_state).choice(x, self.qt_max_rows, replace=False)
        try:
            return shapiro(x)[1]
        except Exception:
            return 0.0

    def _is_valid_for(self, method: str, x: np.ndarray) -> bool:
        x = x[~np.isnan(x)]
        if method == "boxcox":
            return np.all(x > 0)
        elif method in ("sqrt", "log1p"):
            return np.all(x >= 0)
        elif method == "reciprocal":
            return not np.any(np.isclose(x, 0))
        return True

    def _column_is_eligible(self, x: pd.Series) -> bool:
        x = x.dropna()
        if len(x) < 5 or x.std() == 0:
            return False
        if x.nunique() <= 2 or x.isnull().mean() > 0.5:
            return False
        if x.max() - x.min() > 1000 * x.std():
            return False
        if x.nunique() > 0.95 * len(x):
            return False
        if x.skew() > 30:
            return False
        if "timestamp" in str(x.name).lower():
            return False
        return True

    def _evaluate_candidate(self, x: np.ndarray, method: str, scaler_name: str) -> Optional[Dict]:
        try:
            if not self._is_valid_for(method, x):
                return None
            if method == "none":
                x_t = x.copy()
                transformer = None
            elif method == "boxcox":
                x_t, lmbda = boxcox(x)
                transformer = ("boxcox", lmbda)
            elif method == "yeo":
                pt = PowerTransformer(method="yeo-johnson")
                x_t = pt.fit_transform(x.reshape(-1, 1)).flatten()
                transformer = pt
            elif method == "quantile":
                qt = QuantileTransformer(output_distribution="normal", subsample=self.qt_max_rows, random_state=self.random_state)
                x_t = qt.fit_transform(x.reshape(-1, 1)).flatten()
                transformer = qt
            elif method in self.PRE_FUNCS:
                x_t = self.PRE_FUNCS[method](x)
                transformer = None
            else:
                return None

            scaler = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}.get(scaler_name)
            if scaler is None:
                return None

            x_scaled = scaler.fit_transform(x_t.reshape(-1, 1)).flatten()
            x_clean = x_scaled[~np.isnan(x_scaled)]

            pval = self._shapiro(x_clean)
            skew_val = abs(float(skew(x_clean))) if len(x_clean) > 2 else 0.0
            kurt = float(kurtosis(x_clean)) if len(x_clean) > 2 else 0.0

            return {
                "score": (pval, -skew_val),
                "pval": pval,
                "skew": skew_val,
                "kurtosis": kurt,
                "method": method,
                "scaler_name": scaler_name,
                "scaler": scaler,
                "transformer": transformer
            }

        except Exception as e:
            self._log(f"[ERROR] {method}+{scaler_name} failed: {e}")
            return None

    def _evaluate_column(self, x: np.ndarray) -> Dict:
        futures = []
        with ThreadPoolExecutor() as executor:
            for method in self.EXTRA_TRANSFORMS + list(self.PRE_FUNCS.keys()):
                if not self._is_valid_for(method, x):
                    continue
                for scaler in ["standard", "robust", "minmax"]:
                    futures.append(executor.submit(self._evaluate_candidate, x.copy(), method, scaler))

            candidates = [f.result() for f in as_completed(futures) if f.result()]

        if not candidates:
            return {
                "method": "none",
                "scaler_name": "standard",
                "scaler": StandardScaler(),
                "transformer": None,
                "metrics": {},
                "candidates": []
            }

        best = max(candidates, key=lambda r: r["score"])
        return {
            "method": best["method"],
            "scaler_name": best["scaler_name"],
            "scaler": best["scaler"],
            "transformer": best["transformer"],
            "metrics": {
                "pval": best["pval"],
                "skew": best["skew"],
                "kurtosis": best["kurtosis"]
            },
            "candidates": [
                {
                    "method": c["method"],
                    "scaler": c["scaler_name"],
                    "pval": c["pval"],
                    "skew": c["skew"],
                    "kurtosis": c["kurtosis"]
                } for c in candidates
            ]
        }

    def _choose_global_scaler(self, df: pd.DataFrame) -> str:
        skews = {c: skew(df[c].dropna()) for c in df.columns}
        kurts = {c: kurtosis(df[c].dropna()) for c in df.columns}
        for c in df.columns:
            if abs(skews[c]) > self.skew_thresh_robust or abs(kurts[c]) > self.kurt_thresh_robust:
                return "robust"
        if all(abs(skews[c]) < self.skew_thresh_standard for c in df.columns):
            return "standard"
        return "minmax"

    def fit(self, df: pd.DataFrame, numeric_cols: List[str]):
        self.columns = []
        df = df.copy()
        if self.mode == "auto":
            self._auto_chosen = "enhanced" if df.shape[0] * len(numeric_cols) < 1_000_000 else "simple"
        else:
            self._auto_chosen = self.mode

        if self._auto_chosen == "simple":
            scaler_name = self._choose_global_scaler(df[numeric_cols])
            scaler = {"standard": StandardScaler(), "robust": RobustScaler(), "minmax": MinMaxScaler()}[scaler_name]
            df_scaled = pd.DataFrame(scaler.fit_transform(df[numeric_cols]), columns=numeric_cols, index=df.index)
            self.plan["__global_scaler__"] = scaler
            for col in numeric_cols:
                arr = df_scaled[col].values
                best = self._evaluate_column(arr)
                self.plan[col] = best
                self.columns.append(col)
        else:
            for col in numeric_cols:
                x = df[col]
                if not pd.api.types.is_numeric_dtype(x) or not self._column_is_eligible(x):
                    self._log(f"Skipping {col}")
                    continue
                x_clean = x.dropna().values
                self.columns.append(col)
                self.plan[col] = self._evaluate_column(x_clean)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_new = df.copy()
        if "__global_scaler__" in self.plan:
            df_new[self.columns] = self.plan["__global_scaler__"].transform(df_new[self.columns])
        for col in self.columns:
            plan = self.plan[col]
            x = df_new[col].values
            nonnull = ~np.isnan(x)
            try:
                if plan["method"] == "none":
                    x_t = x
                elif plan["method"] == "boxcox":
                    lmbda = plan["transformer"][1]
                    x_t = np.full_like(x, np.nan)
                    x_t[nonnull] = boxcox(x[nonnull], lmbda=lmbda)
                elif plan["method"] in ("yeo", "quantile"):
                    x_t = plan["transformer"].transform(x.reshape(-1, 1)).flatten()
                elif plan["method"] in self.PRE_FUNCS:
                    x_t = self.PRE_FUNCS[plan["method"]](x)
                else:
                    x_t = x
                df_new[col] = plan["scaler"].transform(x_t.reshape(-1, 1)).flatten()
            except Exception as e:
                self._log(f"Transform failed for {col}: {e}")
                df_new[col] = x
        return df_new

    def fit_transform(self, df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        return self.fit(df, numeric_cols).transform(df)

    def get_plan(self, minimal: bool = True) -> Dict:
        if minimal:
            return {col: {
                "method": p["method"],
                "scaler": p["scaler_name"],
                "metrics": p["metrics"]
            } for col, p in self.plan.items() if col != "__global_scaler__"}
        else:
            return self.plan

    def get_report(self) -> pd.DataFrame:
        rows = []
        for col, plan in self.plan.items():
            if col == "__global_scaler__":
                continue
            row = {"column": col}
            row.update(plan["metrics"])
            row["method"] = plan["method"]
            row["scaler"] = plan["scaler_name"]
            rows.append(row)
        return pd.DataFrame(rows).set_index("column")
"""
