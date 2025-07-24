#!/usr/bin/env python3
# auto_dr.py  – Time-efficient automatic dimensionality reduction
# Author: ChatGPT • 2025-06-07

from __future__ import annotations
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import cond
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    TruncatedSVD,
    NMF,
    KernelPCA,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import trustworthiness  # t-SNE dropped for OOS-transform
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class AutoDR(BaseEstimator, TransformerMixin):
    # ————————————————————————————————— parameters / thresholds
    VARIANCE_THRESHOLD = 0.90
    MAX_TSNE_SAMPLES = 2_000  # <-- kept for reference
    MAX_TSNE_FEATURES = 50
    SMALL_FEATURE_LIMIT = 20
    BACKWARD_FEATURE_LIMIT = 40
    RP_FEATURE_THRESHOLD = 1_000
    COND_THRESH = 1e6
    MIN_SAMPLES_PER_FEATURE = 5
    MAX_TSVD_COMPONENTS = 300
    RANDOM_STATE = 0

    # voting: metric → (higher_is_better, weight)
    METRIC_INFO = {
        "cv_score": (True, 2.0),
        "explained_variance": (True, 1.0),
        "reconstruction_error": (False, 1.0),
        "trustworthiness": (True, 1.0),
    }
    BASE_CV_SPLITS = 5

    # ————————————————————————————————— constructor
    def __init__(
        self,
        target: Optional[str] = None,
        variance_threshold: float = VARIANCE_THRESHOLD,
        random_state: int = RANDOM_STATE,
        verbose: bool = True,
    ):
        self.target = target
        self.variance_threshold = variance_threshold
        self.random_state = random_state
        self.verbose = verbose

        # populated during fit
        self.numeric_cols: List[str] = []
        self.report: Dict[str, Any] = {}
        self.models: Dict[str, Tuple[Optional[StandardScaler], Any]] = {}
        self.chosen_technique: Optional[str] = None
        self._X_train: Optional[np.ndarray] = None
        self.task_type: str = "unsupervised"
        self.y: Optional[pd.Series] = None

    # ════════════════════════════════════════════════════════════════════
    # helper utilities
    def _detect_task(self, df: pd.DataFrame) -> str:
        if self.target and self.target in df.columns:
            y = df[self.target]
            if pd.api.types.is_numeric_dtype(y):
                return "classification" if y.nunique() <= 10 else "regression"
            return "classification"
        return "unsupervised"

    def _validate(self, X: np.ndarray) -> Optional[str]:
        n_samples, n_feats = X.shape
        if np.isnan(X).any():
            return "NaNs present"
        if n_feats < 2:
            return "fewer than two numeric features"
        # if n_samples < self.MIN_SAMPLES_PER_FEATURE * n_feats:
        #     return f"too few samples ({n_samples}) for {n_feats} features"
        # if cond(np.cov(X, rowvar=False)) > self.COND_THRESH:
        #     return "ill-conditioned covariance matrix"
        return None

    # ════════════════════════════════════════════════════════════════════
    # technique implementations  →  (X_red, info_dict) and store (scaler, model)

    # ------- PCA
    def _apply_pca(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        full = PCA(random_state=self.random_state).fit(Xs)
        cum = np.cumsum(full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, self.variance_threshold) + 1)
        model = PCA(n_components=n_comp, random_state=self.random_state)
        Xr = model.fit_transform(Xs)
        info = {
            "type": "PCA",
            "n_components": n_comp,
            "explained_variance": float(cum[n_comp - 1]),
        }
        self.models["PCA"] = (scaler, model)
        return Xr, info

    # ------- Incremental PCA
    def _apply_incremental_pca(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        full = IncrementalPCA(random_state=self.random_state).fit(Xs)
        cum = np.cumsum(full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, self.variance_threshold) + 1)
        model = IncrementalPCA(n_components=n_comp, random_state=self.random_state)
        Xr = model.fit_transform(Xs)
        info = {
            "type": "IncrementalPCA",
            "n_components": n_comp,
            "explained_variance": float(cum[n_comp - 1]),
        }
        self.models["IncrementalPCA"] = (scaler, model)
        return Xr, info

    # ------- Truncated SVD
    def _apply_tsvd(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        max_comp = min(Xs.shape[1], self.MAX_TSVD_COMPONENTS)
        full = TruncatedSVD(n_components=max_comp, random_state=self.random_state).fit(
            Xs
        )
        cum = np.cumsum(full.explained_variance_ratio_)
        n_comp = int(np.searchsorted(cum, self.variance_threshold) + 1)
        model = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        Xr = model.fit_transform(Xs)
        info = {
            "type": "TruncatedSVD",
            "n_components": n_comp,
            "explained_variance": float(cum[n_comp - 1]),
        }
        self.models["TruncatedSVD"] = (scaler, model)
        return Xr, info

    # ------- Kernel PCA (RBF)
    def _apply_kpca(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        gamma = 1.0 / X.shape[1]
        model = KernelPCA(
            kernel="rbf",
            gamma=gamma,
            fit_inverse_transform=False,
            random_state=self.random_state,
            n_components=min(10, X.shape[1]),
        )
        Xr = model.fit_transform(X)  # KernelPCA expects raw, not scaled
        info = {"type": "KernelPCA", "params": {"kernel": "rbf", "gamma": gamma}}
        self.models["KernelPCA"] = (None, model)
        return Xr, info

    # ------- Linear Discriminant Analysis
    def _apply_lda(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.task_type != "classification":
            raise ValueError("LDA only for classification")
        model = LinearDiscriminantAnalysis()
        Xr = model.fit_transform(X, self.y)
        info = {
            "type": "LDA",
            "n_components": Xr.shape[1],
            "classes": int(self.y.nunique()),
        }
        self.models["LDA"] = (None, model)
        return Xr, info

    # ------- Supervised PCA via PLSRegression
    def _apply_supervised_pca(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if self.task_type == "regression":
            pls = PLSRegression(n_components=min(X.shape[1], 10))
            Xr = pls.fit_transform(X, self.y)[0]
        else:
            y_enc = pd.get_dummies(self.y).values
            pls = PLSRegression(n_components=min(X.shape[1], 10))
            pls.fit(X, y_enc)
            Xr = pls.transform(X)
        info = {"type": "SupervisedPCA", "n_components": Xr.shape[1]}
        self.models["SupervisedPCA"] = (None, pls)
        return Xr, info

    # ════════════════════════════════════════════════════════════════════
    # scoring & voting
    def _score(self, Xr: np.ndarray, info: Dict) -> Dict[str, float]:
        scores: Dict[str, float] = {}

        # direct metrics from info
        for m in self.METRIC_INFO:
            if m in info:
                up, _ = self.METRIC_INFO[m]
                scores[m] = info[m] if up else -info[m]

        # supervised CV score
        if self.task_type != "unsupervised":
            # choose estimator + CV object
            splits = (
                min(self.BASE_CV_SPLITS, int(self.y.value_counts().min()))
                if self.task_type == "classification"
                else self.BASE_CV_SPLITS
            )
            splits = max(2, splits)
            if self.task_type == "classification":
                est = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        solver="liblinear", random_state=self.random_state
                    ),
                )
                cv = StratifiedKFold(
                    n_splits=splits, shuffle=True, random_state=self.random_state
                )
                val = cross_val_score(est, Xr, self.y, cv=cv, n_jobs=-1).mean()
            else:
                est = make_pipeline(
                    StandardScaler(), Ridge(random_state=self.random_state)
                )
                cv = KFold(
                    n_splits=splits, shuffle=True, random_state=self.random_state
                )
                val = cross_val_score(
                    est, Xr, self.y, cv=cv, scoring="r2", n_jobs=-1
                ).mean()
            scores["cv_score"] = val

        # trustworthiness fallback (only if ≤3 dims and valid neighbours)
        if "trustworthiness" not in scores and Xr.shape[1] <= 3 and Xr.shape[0] > 5:
            n_ngh = min(5, Xr.shape[0] - 1)
            scores["trustworthiness"] = trustworthiness(
                self._X_train, Xr, n_neighbors=n_ngh
            )

        # explained variance fallback
        if (
            "explained_variance" not in scores
            and "model" in info
            and hasattr(info["model"], "explained_variance_ratio_")
        ):
            scores["explained_variance"] = info["model"].explained_variance_ratio_.sum()

        # ensure sign convention (↑ good)
        for m, (up, _) in self.METRIC_INFO.items():
            if m in scores and not up:
                scores[m] = -scores[m]
        return scores

    def _borda_vote(self, metric_table: Dict[str, Dict[str, float]]) -> str:
        rank_df = pd.DataFrame(metric_table).T
        borda = pd.Series(0.0, index=rank_df.index)
        for m, (up, w) in self.METRIC_INFO.items():
            if m not in rank_df.columns:
                continue
            r = rank_df[m].rank(ascending=not up, method="min")
            borda += (r.max() - r) * w
        return borda.idxmax()

    # ════════════════════════════════════════════════════════════════════
    # main API
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        df = df.copy()
        self.numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if self.target and self.target in self.numeric_cols:
            self.numeric_cols.remove(self.target)

        X = df[self.numeric_cols].values
        self._X_train = X
        self.task_type = self._detect_task(df)
        if y is None and self.target:
            y = df[self.target]
        self.y = y

        # sanity-check data
        if err := self._validate(X):
            raise ValueError(f"Data invalid for DR: {err}")

        # -------------------- build candidate list (fast filters) ---------
        cand: List[Tuple[str, Callable[[np.ndarray], Tuple[np.ndarray, Dict]]]] = [
            ("PCA", self._apply_pca)
        ]
        n_samples, n_feats = X.shape
        non_neg = (X >= 0).all()

        if n_samples > 5_000:
            cand.append(("IncrementalPCA", self._apply_incremental_pca))
        if n_feats > n_samples:
            cand.append(("TruncatedSVD", self._apply_tsvd))
        if n_feats <= 10:
            cand.append(("KernelPCA", self._apply_kpca))
        if non_neg and n_feats <= 100:
            # defined later for voting completeness
            cand.append(("NMF", self._apply_nmf))
        if n_feats > self.RP_FEATURE_THRESHOLD:
            cand.append(("RandomProj", self._apply_random_proj))
        if (
            self.task_type == "classification"
            and self.y is not None
            and 2 <= self.y.nunique() < n_feats
        ):
            cand.append(("LDA", self._apply_lda))
            cand.append(("SupervisedPCA", self._apply_supervised_pca))
        elif self.task_type == "regression":
            cand.append(("SupervisedPCA", self._apply_supervised_pca))

        if self.verbose:
            log.info(f"Candidates: {[n for n, _ in cand]}")

        # -------------------- evaluate & vote ----------------------------
        metric_table: Dict[str, Dict[str, float]] = {}
        for name, func in cand:
            try:
                Xr, info = func(X)
                scr = self._score(Xr, info)
                self.report[name] = {"info": info, "metrics": scr}
                metric_table[name] = scr
                if self.verbose:
                    log.info(f"{name}: {scr}")
            except Exception as e:
                self.report[name] = {"error": str(e)}
                if self.verbose:
                    log.warning(f"{name} failed → {e}")

        if not metric_table:
            raise RuntimeError("No DR technique succeeded on the data.")

        winner = self._borda_vote(metric_table)
        self.chosen_technique = winner
        self.report["winner"] = winner
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self.fit(df)
        return self.transform(df)

    # --------------------------- transform -------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.chosen_technique is None:
            raise RuntimeError("Call fit() before transform().")

        if self.chosen_technique == "TSNE":
            raise RuntimeError(
                "t-SNE cannot project new samples – "
                "call fit_transform on the full data."
            )

        scaler, model = self.models[self.chosen_technique]
        X = df[self.numeric_cols].values
        if scaler is not None:
            X = scaler.transform(X)
        Xr = model.transform(X)

        cols = [f"{self.chosen_technique}_C{i+1}" for i in range(Xr.shape[1])]
        df_red = pd.DataFrame(Xr, columns=cols, index=df.index)
        if self.target and self.target in df.columns:
            df_red[self.target] = df[self.target]
        return df_red

    # ════════════════════════════════════════════════════════════════════
    # extra techniques not in your list but referenced in filters
    def _apply_nmf(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        if (X < 0).any():
            raise ValueError("NMF requires non-negative data.")
        n_comp = min(X.shape[1], int(self.variance_threshold * X.shape[1]))
        nmf = NMF(
            n_components=n_comp,
            init="nndsvda",
            max_iter=200,
            random_state=self.random_state,
        )
        W = nmf.fit_transform(X)
        H = nmf.components_
        err = float(np.linalg.norm(X - W @ H, ord="fro"))
        info = {"type": "NMF", "n_components": n_comp, "reconstruction_error": err}
        self.models["NMF"] = (None, nmf)
        return W, info

    def _apply_random_proj(self, X: np.ndarray) -> Tuple[np.ndarray, Dict]:
        n_comp = min(X.shape[1], max(2, int(self.variance_threshold * X.shape[1])))
        rp = GaussianRandomProjection(
            n_components=n_comp, random_state=self.random_state
        )
        Xr = rp.fit_transform(X)
        info = {"type": "RandomProj", "n_components": n_comp}
        self.models["RandomProj"] = (None, rp)
        return Xr, info

    def _apply_forward_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.numeric_cols]
        y = df[self.target]
        task = self._detect_task(df)
        if task == "classification":
            est, scoring = (
                LogisticRegression(solver="liblinear", random_state=self.random_state),
                "accuracy",
            )
        else:
            est, scoring = Ridge(random_state=self.random_state), "r2"
        sfs = SequentialFeatureSelector(
            est,
            n_features_to_select="auto",
            direction="forward",
            scoring=scoring,
            cv=3,
            n_jobs=-1,
        )
        sfs.fit(X, y)
        sel = X.columns[sfs.get_support()].tolist()
        self.report["dim_reduction"] = {
            "type": "forward_selection",
            "selected_features": sel,
        }
        self._dr_model = sfs
        df_sel = df[sel].copy()
        if self.target:
            df_sel[self.target] = df[self.target]
        return df_sel

    def _apply_backward_selection(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.numeric_cols]
        y = df[self.target]
        task = self._detect_task(df)
        if task == "classification":
            est, scoring = (
                LogisticRegression(solver="liblinear", random_state=self.random_state),
                "accuracy",
            )
        else:
            est, scoring = Ridge(random_state=self.random_state), "r2"
        sfs = SequentialFeatureSelector(
            est,
            n_features_to_select="auto",
            direction="backward",
            scoring=scoring,
            cv=3,
            n_jobs=-1,
        )
        sfs.fit(X, y)
        sel = X.columns[sfs.get_support()].tolist()
        self.report["dim_reduction"] = {
            "type": "backward_selection",
            "selected_features": sel,
        }
        self._dr_model = sfs
        df_sel = df[sel].copy()
        if self.target:
            df_sel[self.target] = df[self.target]
        return df_sel


"""
Below is a **step-by-step recipe** (with code snippets) that plugs the four routines you pasted into the “voting” framework of the last `AutoDR` class.
None of the other logic needs to change—you only have to:

1. **Expose each routine as a *candidate* wrapper that returns** `(X_red, info_dict)`.
2. **Register those wrappers** in the place where the class builds its `cand` list.
3. **(For FS) keep track of the selected columns** so `transform()` can reproduce the reduction on new data.

I’ll show the exact diff you need to add. Copy-paste the blocks into your existing file and you’re done.

---

## 1 · Turn the functions into “candidate” wrappers

```python
# add right after _apply_random_proj or wherever convenient
# ---------------------------------------------------------

def _cand_nmf(self, X: np.ndarray):
    X_red, info = self._apply_nmf(X)
    return X_red, info

def _cand_random_proj(self, X: np.ndarray):
    X_red, info = self._apply_random_proj(X)
    return X_red, info

def _cand_forward_sel(self, X: np.ndarray):
    # needs DataFrame – wrap in internal call using self.raw_df
    df_sel = self._apply_forward_selection(self.raw_df)
    X_red = df_sel[self.numeric_cols].values
    info = {
        "type": "ForwardSelection",
        "selected_features": df_sel.columns.tolist()
    }
    # store model so transform() works
    self.models["ForwardSelection"] = (None, self._dr_model)
    return X_red, info

def _cand_backward_sel(self, X: np.ndarray):
    df_sel = self._apply_backward_selection(self.raw_df)
    X_red = df_sel[self.numeric_cols].values
    info = {
        "type": "BackwardSelection",
        "selected_features": df_sel.columns.tolist()
    }
    self.models["BackwardSelection"] = (None, self._dr_model)
    return X_red, info
```

*`self.raw_df` will be assigned in **fit()** (see next section) so the wrappers can reuse the DataFrame.*

---

## 2 · Register the new candidates in `fit`

Inside `fit()` you already build `cand = [...]`.
Append the new wrappers **according to simple heuristics** (so you don’t waste time evaluating hopeless options):

```python
# before if self.verbose: log.info(...)
if non_neg and n_feats <= 100:
    cand.append(("NMF", self._cand_nmf))

if n_feats > self.RP_FEATURE_THRESHOLD:
    cand.append(("RandomProj", self._cand_random_proj))

# Wrapper Feature Selection – cheap, only try when #features is small
if self.task_type != "unsupervised" and n_feats <= self.SMALL_FEATURE_LIMIT:
    cand.append(("ForwardSelection", self._cand_forward_sel))
elif self.task_type != "unsupervised" and n_feats <= self.BACKWARD_FEATURE_LIMIT:
    cand.append(("BackwardSelection", self._cand_backward_sel))

# Wrapper Feature-selection heuristics based on the decision matrix
# (p = n_feats, n = n_samples)
p, n = n_feats, n_samples
forward_ok  = (p <= 20) and (n >= 2 * p)                  # small & many rows
backward_ok = ((20 < p <= 60) or (p <= 20 and n < 2 * p)) \
              and self.task_type != "unsupervised"

if forward_ok:
    cand.append(("ForwardSelection", self._cand_forward_sel))
if backward_ok:
    cand.append(("BackwardSelection", self._cand_backward_sel))
```

(If you also want the *backward* selector when features are tiny, include both—you can’t hurt much.)

---

## 3 · Remember the original DataFrame for FS wrappers

At the very **top** of `fit()` (right after `df = df.copy()`), store the DataFrame:

```python
self.raw_df = df           # so the wrappers can access non-numeric cols
```

---

## 4 · Teach `transform()` how to apply FS reductions

Add this block just before the final return in `transform()`:

```python
# -------- Forward / Backward Selection -----------
if self.chosen_technique in {"ForwardSelection", "BackwardSelection"}:
    mask = self.models[self.chosen_technique][1].get_support()
    sel_cols = [c for c, m in zip(self.numeric_cols, mask) if m]
    df_red = df[sel_cols].copy()
    if self.target and self.target in df.columns:
        df_red[self.target] = df[self.target]
    return df_red
```

Everything else in `transform()` can stay as is.

---

## 5 · Recap: full integration checklist

1. **Paste the 4 × `_cand_*` wrappers** from section 1 into the class.
2. **Insert the three `cand.append()` lines** shown in section 2 into your `fit()` filter block.
3. **Add `self.raw_df = df`** near the start of `fit()`.
4. **Extend `transform()`** with the “FS branch” from section 4.

That’s all—the selector will automatically:

* run only the techniques that make sense for the data,
* score them on the same metrics,
* vote, and
* remember the winner so `transform()` reproduces the reduction.

---

### Quick usage test

```python
dr = AutoDR(target="label", verbose=True)
df_red = dr.fit_transform(df_train)
print("Winner:", dr.chosen_technique)
df_new_red = dr.transform(df_new)    # works for PCA/FS/etc.
```

If the winner is `ForwardSelection` or `BackwardSelection`, the returned DataFrames will contain only the selected numeric columns (plus the target). For PCA-like winners they’ll contain the new component columns.

Feel free to tweak the filter rules (`n_feats` limits, non-negativity) and metric weights; the skeleton above drops straight into the class you already have.

"""
