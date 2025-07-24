import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


class FeatureSplitter(BaseEstimator, TransformerMixin):
    """
    Standalone feature splitter integrating mixed-pattern splits, delimiter splits,
    multi-label splits, regex extraction, JSON/XML flattening, URL parsing,
    IP/color/filepath/email/phone parsing, plus evaluate + threshold filtering.
    """

    def __init__(
        self,
        mixed_patterns: Dict[str, str] = None,
        delimiter_splits: Optional[Dict[str, str]] = None,
        multi_label_delimiter: Optional[str] = None,
        regex_extract: Optional[Dict[str, str]] = None,
        url_columns: Optional[List[str]] = None,
        ip_columns: Optional[List[str]] = None,
        color_hex_columns: Optional[List[str]] = None,
        filepath_columns: Optional[List[str]] = None,
        email_columns: Optional[List[str]] = None,
        phone_columns: Optional[List[str]] = None,
        flatten_json: bool = True,
        flatten_xml: bool = False,
        strip_html: bool = True,
        cache_json: bool = True,
        drop_original: bool = True,
        n_jobs: int = 1,
        min_mutual_info: Optional[float] = None,
        min_abs_corr: Optional[float] = None,
    ):
        # splitting params
        self.mixed_patterns = mixed_patterns or {
            "alpha": r"([A-Za-z]+)",
            "digits": r"(\d+)",
            "punct": r"([^A-Za-z0-9]+)",
        }
        self.delimiter_splits = delimiter_splits or {}
        self.multi_label_delimiter = multi_label_delimiter
        self.regex_extract = regex_extract or {}
        self.url_columns = url_columns or []
        self.ip_columns = ip_columns or []
        self.color_hex_columns = color_hex_columns or []
        self.filepath_columns = filepath_columns or []
        self.email_columns = email_columns or []
        self.phone_columns = phone_columns or []

        # JSON/XML/html params
        self.flatten_json = flatten_json
        self.flatten_xml = flatten_xml
        self.strip_html = strip_html
        self.cache_json = cache_json

        # behavior params
        self.drop_original = drop_original
        self.n_jobs = n_jobs

        # evaluation thresholds
        self.min_mutual_info = min_mutual_info
        self.min_abs_corr = min_abs_corr

        # internals
        self._lock = threading.RLock()
        if self.cache_json:
            self._json_loads = lru_cache(maxsize=10_000)(json.loads)
        else:
            self._json_loads = json.loads
        if self.flatten_xml:
            self._xml_tag = re.compile(r"<(\w+)[^>]*>(.*?)</\1>", re.DOTALL)

        # to be populated during split/evaluate
        self.splits_: Dict[str, List[str]] = {}
        self.report_: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.decisions_: Dict[str, Dict[str, Union[bool, str, Dict[str, float]]]] = {}

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _split_mixed(self, df: pd.DataFrame, src: str, strip_html_fn):
        out: Dict[str, pd.Series] = {}
        text = df[src].astype(str).map(strip_html_fn)
        for label, pat in self.mixed_patterns.items():
            arr = text.str.extractall(pat)[0].unstack().fillna("")
            for i in range(arr.shape[1]):
                out[f"{src}_{label}_{i}"] = arr.iloc[:, i]
        if self.drop_original:
            with self._lock:
                df.drop(columns=[src], inplace=True, errors="ignore")
        return out

    def _flatten_json_col(self, df: pd.DataFrame, col: str):
        try:
            parsed = df[col].astype(str).map(self._json_loads)
            flat = pd.json_normalize(parsed).add_prefix(f"{col}_")
            return col, flat
        except Exception:
            return col, None

    def _flatten_xml_col(self, df: pd.DataFrame, col: str):
        try:
            texts = df[col].astype(str)
            matches = texts.map(
                lambda t: {m.group(1): m.group(2) for m in self._xml_tag.finditer(t)}
            )
            if matches.empty:
                return col, None
            flat = pd.json_normalize(matches).add_prefix(f"{col}_")
            return col, flat
        except Exception:
            return col, None

    def _full_split_pipeline(self, X: pd.DataFrame):
        df = X.copy()
        new_frames_info: List[Tuple[str, pd.DataFrame]] = []
        self.splits_ = {}

        # helper: strip HTML if needed
        def strip_html_fn(text: str) -> str:
            return re.sub(r"<[^>]+>", "", text) if self.strip_html else text

        # 1) mixed, JSON, XML splits in parallel
        tasks: List[Tuple[threading.Future, str, str]] = []
        with ThreadPoolExecutor(max_workers=self.n_jobs) as exec:
            # mixed-pattern splits
            for src in df.select_dtypes(include=["object"]).columns:
                fut = exec.submit(self._split_mixed, df, src, strip_html_fn)
                tasks.append((fut, "mixed", src))
            # JSON flatten
            if self.flatten_json:
                for col in df.columns:
                    fut = exec.submit(self._flatten_json_col, df, col)
                    tasks.append((fut, "json", col))
            # XML flatten
            if self.flatten_xml:
                for col in df.columns:
                    fut = exec.submit(self._flatten_xml_col, df, col)
                    tasks.append((fut, "xml", col))
            # collect results
            for fut, kind, group in tasks:
                res = fut.result()
                if kind == "mixed" and isinstance(res, dict):
                    frame = pd.DataFrame(res)
                    new_frames_info.append((group, frame))
                    self.splits_.setdefault(group, []).extend(frame.columns.tolist())
                elif kind in ("json", "xml") and isinstance(res, tuple):
                    col, flat = res
                    if flat is not None:
                        new_frames_info.append((col, flat))
                        self.splits_.setdefault(col, []).extend(flat.columns.tolist())
                        if self.drop_original:
                            df.drop(columns=[col], inplace=True, errors="ignore")

        # 2) V3-like splits: delimiter
        for col, delim in self.delimiter_splits.items():
            if col in df.columns:
                parts = df[col].astype(str).str.split(delim, expand=True)
                cols = [f"{col}_delim_{i}" for i in range(parts.shape[1])]
                parts.columns = cols
                new_frames_info.append((col, parts))
                self.splits_.setdefault(col, []).extend(cols)
                if self.drop_original:
                    df.drop(columns=[col], inplace=True, errors="ignore")

        # 3) multi-label delimiter → get_dummies
        if self.multi_label_delimiter:
            for col in df.select_dtypes(include=["object"]).columns:
                if self.multi_label_delimiter in df[col].astype(str).str.cat(
                    sep=self.multi_label_delimiter
                ):
                    dummies = (
                        df[col]
                        .astype(str)
                        .str.get_dummies(sep=self.multi_label_delimiter)
                    )
                    dummies = dummies.add_prefix(f"{col}_lbl_")
                    new_frames_info.append((col, dummies))
                    self.splits_.setdefault(col, []).extend(dummies.columns.tolist())
                    if self.drop_original:
                        df.drop(columns=[col], inplace=True, errors="ignore")

        # 4) regex extract
        for col, pat in self.regex_extract.items():
            if col in df.columns:
                arr = df[col].astype(str).str.extractall(pat)[0].unstack().fillna("")
                cols = [f"{col}_rex_{i}" for i in range(arr.shape[1])]
                arr.columns = cols
                new_frames_info.append((col, arr))
                self.splits_.setdefault(col, []).extend(cols)
                if self.drop_original:
                    df.drop(columns=[col], inplace=True, errors="ignore")

        # 5) URL parse
        for col in self.url_columns:
            if col in df.columns:
                parsed = df[col].astype(str).apply(urlparse)
                df_url = pd.DataFrame(
                    {
                        f"{col}_scheme": parsed.map(lambda p: p.scheme),
                        f"{col}_netloc": parsed.map(lambda p: p.netloc),
                        f"{col}_path": parsed.map(lambda p: p.path),
                        f"{col}_params": parsed.map(lambda p: p.params),
                        f"{col}_query": parsed.map(lambda p: p.query),
                        f"{col}_fragment": parsed.map(lambda p: p.fragment),
                    }
                )
                new_frames_info.append((col, df_url))
                self.splits_.setdefault(col, []).extend(df_url.columns.tolist())
                if self.drop_original:
                    df.drop(columns=[col], inplace=True, errors="ignore")

        # 6) IP extract
        ip_pat = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")
        for col in self.ip_columns:
            if col in df.columns:
                extracted = df[col].astype(str).str.extract(ip_pat)[0].fillna("")
                df_ip = pd.DataFrame({f"{col}_ip": extracted})
                new_frames_info.append((col, df_ip))
                self.splits_.setdefault(col, []).append(f"{col}_ip")
                if self.drop_original:
                    df.drop(columns=[col], inplace=True, errors="ignore")

        # 7) Color hex extract
        hex_pat = re.compile(r"#(?:[0-9a-fA-F]{3}){1,2}")
        for col in self.color_hex_columns:
            if col in df.columns:
                extracted = df[col].astype(str).str.extract(hex_pat)[0].fillna("")
                df_hex = pd.DataFrame({f"{col}_hex": extracted})
                new_frames_info.append((col, df_hex))
                self.splits_.setdefault(col, []).append(f"{col}_hex")
                if self.drop_original:
                    df.drop(columns=[col], inplace=True, errors="ignore")

        # 8) Filepath parse
        for col in self.filepath_columns:
            if col in df.columns:
                dirs = df[col].astype(str).map(os.path.dirname)
                files = df[col].astype(str).map(os.path.basename)
                exts = df[col].astype(str).map(lambda x: os.path.splitext(x)[1])
                df_fp = pd.DataFrame(
                    {f"{col}_dir": dirs, f"{col}_file": files, f"{col}_ext": exts}
                )
                new_frames_info.append((col, df_fp))
                self.splits_.setdefault(col, []).extend(df_fp.columns.tolist())
                if self.drop_original:
                    df.drop(columns=[col], inplace=True, errors="ignore")

        # 9) Email parse
        email_pat = re.compile(r"([^@]+)@(.+)")
        for col in self.email_columns:
            if col in df.columns:
                arr = df[col].astype(str).str.extract(email_pat).fillna("")
                arr.columns = [f"{col}_user", f"{col}_domain"]
                new_frames_info.append((col, arr))
                self.splits_.setdefault(col, []).extend(arr.columns.tolist())
                if self.drop_original:
                    df.drop(columns=[col], inplace=True, errors="ignore")

        # 10) Phone digits
        for col in self.phone_columns:
            if col in df.columns:
                digits = df[col].astype(str).str.replace(r"\D+", "", regex=True)
                df_ph = pd.DataFrame({f"{col}_digits": digits})
                new_frames_info.append((col, df_ph))
                self.splits_.setdefault(col, []).append(f"{col}_digits")
                if self.drop_original:
                    df.drop(columns=[col], inplace=True, errors="ignore")

        return df, new_frames_info

    def transform(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ):
        base_df, new_frames_info = self._full_split_pipeline(X)

        # no filtering if no target or no thresholds
        if y is None or (self.min_mutual_info is None and self.min_abs_corr is None):
            return pd.concat(
                [base_df] + [frame for _, frame in new_frames_info], axis=1
            )

        # apply threshold-based filtering
        metrics = self.evaluate(X, y)
        kept_frames: List[pd.DataFrame] = []
        self.decisions_ = {}

        for group, frame in new_frames_info:
            kept_cols = []
            for col in frame.columns:
                m = metrics.get(group, {}).get(col, {})
                applied = True
                reason = "no metric"
                if "mutual_info" in m and self.min_mutual_info is not None:
                    mi = m["mutual_info"]
                    if mi < self.min_mutual_info:
                        applied = False
                        reason = (
                            f"mutual_info {mi:.4f} < threshold {self.min_mutual_info}"
                        )
                    else:
                        reason = (
                            f"mutual_info {mi:.4f} ≥ threshold {self.min_mutual_info}"
                        )
                elif "abs_corr" in m and self.min_abs_corr is not None:
                    corr = m["abs_corr"]
                    if corr < self.min_abs_corr:
                        applied = False
                        reason = f"abs_corr {corr:.4f} < threshold {self.min_abs_corr}"
                    else:
                        reason = f"abs_corr {corr:.4f} ≥ threshold {self.min_abs_corr}"
                self.decisions_[col] = {
                    "applied": applied,
                    "metric": m,
                    "reason": reason,
                }
                if applied:
                    kept_cols.append(col)
            if kept_cols:
                kept_frames.append(frame[kept_cols])

        return pd.concat([base_df] + kept_frames, axis=1)

    def evaluate(
        self, X: pd.DataFrame, y: Optional[Union[pd.Series, np.ndarray]] = None
    ):
        base_df, new_frames_info = self._full_split_pipeline(X)
        df_new = pd.concat([base_df] + [frame for _, frame in new_frames_info], axis=1)

        metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        is_class = False
        y_ser = None
        y_enc = None
        if y is not None:
            y_ser = pd.Series(y) if not isinstance(y, pd.Series) else y
            is_class = not (y_ser.dtype.kind in "ifu" and y_ser.nunique() > 20)
            if is_class and y_ser.dtype.kind not in "iu":
                y_enc = LabelEncoder().fit_transform(y_ser)
            else:
                y_enc = y_ser.values

        for group, cols in self.splits_.items():
            grp_metrics: Dict[str, Dict[str, float]] = {}
            for c in cols:
                arr = df_new[c]
                m: Dict[str, float] = {
                    "non_null_ratio": float(arr.notnull().mean()),
                    "unique_count": float(arr.nunique()),
                }
                if y is not None:
                    if is_class:
                        mi = mutual_info_classif(
                            arr.to_frame(), y_enc, discrete_features="auto"
                        )[0]
                        m["mutual_info"] = float(mi)
                    else:
                        corr = abs(float(arr.corr(y_ser)))
                        m["abs_corr"] = corr
                grp_metrics[c] = m
            metrics[group] = grp_metrics

        self.report_ = metrics
        return metrics
