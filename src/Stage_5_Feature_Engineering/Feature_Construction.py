import re
import json
import threading
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Union


class FeatureConstructor(BaseEstimator, TransformerMixin):
    """
    Standalone feature constructor: group aggregations, ratios, differences, crosses with hashing,
    frequency counts, text stats, date diffs, rolling windows, custom funcs, plus evaluate + filtering.
    """

    def __init__(
        self,
        group_aggs: Optional[Dict[str, List[str]]] = None,
        ratio_pairs: Optional[List[Tuple[str, str]]] = None,
        diff_pairs: Optional[List[Tuple[str, str]]] = None,
        crosses: Optional[List[Tuple[str, str]]] = None,
        freq_count_cols: Optional[List[str]] = None,
        text_stat_cols: Optional[List[str]] = None,
        date_diff_pairs: Optional[List[Tuple[str, str]]] = None,
        rolling_windows: Optional[Dict[str, int]] = None,
        custom_funcs: Optional[Dict[str, callable]] = None,
        n_jobs: int = 1,
        min_score: Optional[float] = None,
    ):
        self.group_aggs = group_aggs or {}
        self.ratio_pairs = ratio_pairs or []
        self.diff_pairs = diff_pairs or []
        self.crosses = crosses or []
        self.freq_count_cols = freq_count_cols or []
        self.text_stat_cols = text_stat_cols or []
        self.date_diff_pairs = date_diff_pairs or []
        self.rolling_windows = rolling_windows or {}
        self.custom_funcs = custom_funcs or {}
        self.n_jobs = n_jobs
        self.min_score = min_score

        self._lock = threading.RLock()
        self.decisions_: Dict[str, Dict[str, Union[bool, float, str]]] = {}
        self.report_: Dict[str, float] = {}

        import hashlib

        self._sha256 = hashlib.sha256
        try:
            from datasketch import MinHash

            self._MinHash = MinHash
        except ImportError:
            self._MinHash = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _base_transform(self, X: pd.DataFrame):
        df = X.copy()

        # 1) group aggregations
        for key, aggs in self.group_aggs.items():
            if key in df.columns:
                grp = df.groupby(key).agg(aggs)
                grp.columns = [f"{key}_{col}_{func}" for col, func in grp.columns]
                df = df.join(grp, on=key)

        # 2) ratio pairs
        for a, b in self.ratio_pairs:
            if a in df.columns and b in df.columns:
                denom = df[b].replace({0: np.nan})
                df[f"{a}_to_{b}_ratio"] = df[a] / denom

        # 3) diff pairs
        for a, b in self.diff_pairs:
            if a in df.columns and b in df.columns:
                df[f"{a}_minus_{b}"] = df[a] - df[b]

        # 4) crosses + hashing + MinHash
        for a, b in self.crosses:
            if a in df.columns and b in df.columns:
                combined = df[a].astype(str) + "_" + df[b].astype(str)
                df[f"{a}_x_{b}"] = combined
                df[f"{a}_x_{b}_hash"] = combined.map(
                    lambda s: self._sha256(s.encode()).hexdigest()
                )
                if self._MinHash:

                    def make_mh(s: str):
                        m = self._MinHash()
                        for token in s.split():
                            m.update(token.encode())
                        return m.hashvalues

                    df[f"{a}_x_{b}_mhsig"] = combined.map(make_mh)

        # 5) frequency count
        for col in self.freq_count_cols:
            if col in df.columns:
                counts = df[col].map(df[col].value_counts())
                df[f"{col}_freq_count"] = counts

        # 6) text stats
        for c in self.text_stat_cols:
            if c in df.columns:
                s = df[c].fillna("").astype(str)
                df[f"{c}_char_len"] = s.str.len()
                df[f"{c}_word_count"] = s.str.split().str.len()
                df[f"{c}_uniq_words"] = s.apply(lambda x: len(set(x.split())))
                df[f"{c}_punc_count"] = s.apply(
                    lambda x: sum(ch in ".,;:!?" for ch in x)
                )

        # 7) date differences
        for a, b in self.date_diff_pairs:
            if a in df.columns and b in df.columns:
                da = pd.to_datetime(df[a], errors="coerce")
                db = pd.to_datetime(df[b], errors="coerce")
                df[f"{a}_to_{b}_days"] = (da - db).dt.days

        # 8) rolling windows
        for c, win in self.rolling_windows.items():
            if c in df.columns and np.issubdtype(df[c].dtype, np.number):
                df[f"{c}_roll_mean_{win}"] = (
                    df[c].rolling(window=win, min_periods=1).mean()
                )
                df[f"{c}_roll_std_{win}"] = (
                    df[c].rolling(window=win, min_periods=1).std()
                )

        # 9) custom functions
        for name, func in self.custom_funcs.items():
            df[name] = func(df)

        return df

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ):
        df_feats = self._base_transform(X)
        new_cols = [c for c in df_feats.columns if c not in X.columns]

        # no filtering if no target or no threshold
        if y is None or self.min_score is None:
            return df_feats

        scores = self.evaluate(X, y)
        kept = [c for c, score in scores.items() if score >= self.min_score]
        self.decisions_ = {}

        for c, score in scores.items():
            applied = c in kept
            reason = f"score {score:.4f} {'≥' if applied else '<'} threshold {self.min_score}"
            self.decisions_[c] = {"score": score, "applied": applied, "reason": reason}

        return df_feats[X.columns.tolist() + kept]

    def evaluate(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]):
        df_feats = self._base_transform(X)
        new_cols = [c for c in df_feats.columns if c not in X.columns]
        y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y
        is_reg = y_ser.dtype.kind in "ifu" and y_ser.nunique() > 20

        scores: Dict[str, float] = {}
        if is_reg:
            for c in new_cols:
                scores[c] = abs(df_feats[c].corr(y_ser))
        else:
            y_enc = LabelEncoder().fit_transform(y_ser)
            mi = mutual_info_classif(
                df_feats[new_cols], y_enc, discrete_features="auto"
            )
            for c, m in zip(new_cols, mi):
                scores[c] = float(m)

        self.report_ = scores
        return scores
