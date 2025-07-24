# extended_feature_selector.py

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SequentialFeatureSelector,
    VarianceThreshold,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import (
    ElasticNetCV,
    LassoCV,
    LogisticRegression,
    LogisticRegressionCV,
)
from sklearn.preprocessing import LabelEncoder


class ExtendedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Comprehensive, parallel feature selector combining multiple methods:
      - Univariate: mutual information & ANOVA/F-test
      - Filter: variance threshold & correlation threshold
      - Embedded: RandomForest & GradientBoosting importances
      - Permutation importance
      - L1-based: LassoCV & LogisticRegressionCV
      - Wrapper: RFE & SequentialFeatureSelector

    Keeps features with ≥ vote_threshold fraction of “keep” votes.

    Parameters
    ----------
    is_classification : Optional[bool]
        If None, inferred from y.
    vote_threshold : float
        Fraction of methods that must vote “keep” (0–1).
    mi_k : Optional[int]
        Keep top-k by mutual information (None → keep all >0).
    f_k : Optional[int]
        Keep top-k by ANOVA F-score.
    variance_threshold : float
        Drop features with variance ≤ this.
    corr_threshold : float
        Drop features with max |corr| ≥ this.
    rf_n_estimators : int
    gb_n_estimators : int
    perm_n_repeats : int
    l1_C : float
    enet_l1_ratio : float
    rfe_n_features : Optional[int]
    sfs_n_features : Optional[int]
    cv : int
    n_jobs : int
    random_state : int
    """

    def __init__(
        self,
        is_classification: Optional[bool] = None,
        vote_threshold: float = 0.5,
        mi_k: Optional[int] = None,
        f_k: Optional[int] = None,
        variance_threshold: float = 0.0,
        corr_threshold: float = 0.9,
        rf_n_estimators: int = 100,
        gb_n_estimators: int = 100,
        perm_n_repeats: int = 10,
        l1_C: float = 1.0,
        enet_l1_ratio: float = 0.5,
        rfe_n_features: Optional[int] = None,
        sfs_n_features: Optional[int] = None,
        cv: int = 5,
        n_jobs: int = 1,
        random_state: int = 42,
    ):
        self.is_classification = is_classification
        self.vote_threshold = vote_threshold
        self.mi_k = mi_k
        self.f_k = f_k
        self.variance_threshold = variance_threshold
        self.corr_threshold = corr_threshold
        self.rf_n_estimators = rf_n_estimators
        self.gb_n_estimators = gb_n_estimators
        self.perm_n_repeats = perm_n_repeats
        self.l1_C = l1_C
        self.enet_l1_ratio = enet_l1_ratio
        self.rfe_n_features = rfe_n_features
        self.sfs_n_features = sfs_n_features
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.selected_features_: List[str] = []
        self.report_: Dict[str, Set[str]] = {}
        self.timings_: Dict[str, float] = {}

    def _infer_task(self, y: Union[pd.Series, np.ndarray]) -> bool:
        """Return True if classification, False if regression."""
        y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y
        return not (y_ser.dtype.kind in "ifu" and y_ser.nunique() > 20)

    def _univariate_mi(self, X: pd.DataFrame, y) -> Set[str]:
        mi_func = (
            mutual_info_classif if self.is_classification else mutual_info_regression
        )
        mi = mi_func(X, y, discrete_features="auto", random_state=self.random_state)
        s = pd.Series(mi, index=X.columns).sort_values(ascending=False)
        if self.mi_k:
            return set(s.iloc[: self.mi_k].index)
        else:
            return set(s[s > 0].index)

    def _univariate_f(self, X: pd.DataFrame, y) -> Set[str]:
        f_func = f_classif if self.is_classification else f_regression
        f_vals, _ = f_func(X, y)
        s = pd.Series(f_vals, index=X.columns).sort_values(ascending=False)
        if self.f_k:
            return set(s.iloc[: self.f_k].index)
        else:
            return set(s[s > 0].index)

    def _variance_keep(self, X: pd.DataFrame, y=None) -> Set[str]:
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X)
        return set(X.columns[vt.get_support()])

    def _correlation_keep(self, X: pd.DataFrame, y=None) -> Set[str]:
        corr = X.corr().abs()
        np.fill_diagonal(corr.values, 0)
        drop = set(corr.columns[corr.max() >= self.corr_threshold])
        return set(X.columns) - drop

    def _rf_keep(self, X: pd.DataFrame, y) -> Set[str]:
        Model = (
            RandomForestClassifier if self.is_classification else RandomForestRegressor
        )
        m = Model(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        imp = pd.Series(m.feature_importances_, index=X.columns)
        return set(imp[imp >= imp.median()].index)

    def _gb_keep(self, X: pd.DataFrame, y) -> Set[str]:
        Model = (
            GradientBoostingClassifier
            if self.is_classification
            else GradientBoostingRegressor
        )
        m = Model(
            n_estimators=self.gb_n_estimators, random_state=self.random_state
        ).fit(X, y)
        imp = pd.Series(m.feature_importances_, index=X.columns)
        return set(imp[imp >= imp.median()].index)

    def _perm_keep(self, X: pd.DataFrame, y) -> Set[str]:
        Model = (
            RandomForestClassifier if self.is_classification else RandomForestRegressor
        )
        base = Model(
            n_estimators=self.rf_n_estimators,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        perm = permutation_importance(
            base,
            X,
            y,
            n_repeats=self.perm_n_repeats,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        imp = pd.Series(perm.importances_mean, index=X.columns)
        return set(imp[imp >= imp.median()].index)

    def _l1_keep(self, X: pd.DataFrame, y) -> Set[str]:
        if self.is_classification:
            model = LogisticRegressionCV(
                penalty="l1",
                solver="saga",
                Cs=[self.l1_C],
                cv=self.cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        else:
            model = LassoCV(
                alphas=[1 / self.l1_C],
                cv=self.cv,
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
        # wrap in SelectFromModel, threshold='median' → median coefficient/magnitude
        sfm = SelectFromModel(model).fit(X, y)
        return set(X.columns[sfm.get_support()])

    def _enet_keep(self, X: pd.DataFrame, y) -> Set[str]:
        model = ElasticNetCV(
            l1_ratio=self.enet_l1_ratio,
            cv=self.cv,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        sfm = SelectFromModel(model).fit(X, y)
        return set(X.columns[sfm.get_support()])

    def _rfe_keep(self, X: pd.DataFrame, y) -> Set[str]:
        if not self.rfe_n_features:
            return set(X.columns)
        if self.is_classification:
            est = LogisticRegression(
                penalty="none",
                solver="lbfgs",
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        else:
            est = LassoCV(
                cv=self.cv, n_jobs=self.n_jobs, random_state=self.random_state
            )
        rfe = RFE(est, n_features_to_select=self.rfe_n_features).fit(X, y)
        return set(X.columns[rfe.support_])

    def _sfs_keep(self, X: pd.DataFrame, y) -> Set[str]:
        if not self.sfs_n_features:
            return set(X.columns)
        if self.is_classification:
            est = LogisticRegressionCV(
                cv=self.cv,
                solver="liblinear",
                n_jobs=self.n_jobs,
                random_state=self.random_state,
            )
            scoring = "accuracy"
        else:
            est = ElasticNetCV(
                cv=self.cv, n_jobs=self.n_jobs, random_state=self.random_state
            )
            scoring = "r2"
        sfs = SequentialFeatureSelector(
            est,
            n_features_to_select=self.sfs_n_features,
            direction="forward",
            scoring=scoring,
            cv=self.cv,
            n_jobs=self.n_jobs,
        ).fit(X, y)
        return set(X.columns[sfs.get_support()])

    def fit(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        X = X.copy()
        y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y.copy()
        if self.is_classification is None:
            self.is_classification = self._infer_task(y_ser)
        if self.is_classification and y_ser.dtype.kind not in "iu":
            y_ser = LabelEncoder().fit_transform(y_ser)

        methods = {
            "mi": self._univariate_mi,
            "f_test": self._univariate_f,
            "variance": self._variance_keep,
            "correlation": self._correlation_keep,
            "rf": self._rf_keep,
            "gb": self._gb_keep,
            "perm": self._perm_keep,
            "l1": self._l1_keep,
            "enet": self._enet_keep,
            "rfe": self._rfe_keep,
            "sfs": self._sfs_keep,
        }

        votes: Dict[str, int] = {c: 0 for c in X.columns}
        self.report_ = {}
        self.timings_ = {}
        total_methods = len(methods)

        with ThreadPoolExecutor(max_workers=self.n_jobs) as exe:
            futures = {}
            for name, fn in methods.items():
                t0 = time.perf_counter()
                futures[exe.submit(fn, X, y_ser)] = (name, t0)
            for fut in as_completed(futures):
                name, t0 = futures[fut]
                kept = fut.result()
                self.report_[name] = kept
                self.timings_[name] = time.perf_counter() - t0
                for c in X.columns:
                    if c in kept:
                        votes[c] += 1

        min_votes = int(np.ceil(self.vote_threshold * total_methods))
        self.selected_features_ = [c for c, v in votes.items() if v >= min_votes]
        self.report_["final_keep"] = set(self.selected_features_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.selected_features_].copy()

    def fit_transform(
        self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
