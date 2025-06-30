#!/usr/bin/env python3
"""
Stage 5: Categorical Encoding Variants

  • For each categorical column, decide how to encode:
      – One‐Hot if unique_frac ≤ onehot_thresh
      – Ordinal if onehot_thresh < unique_frac ≤ ordinal_thresh
      – Frequency if ordinal_thresh < unique_frac ≤ freq_thresh
      – Else: add to “suggest_target_encode” list (user can pursue WOE or LOO)
  • Outputs three processed CSVs (linear, tree, knn variants) into working directory.
  • Exposes thresholds as class‐level constants or __init__ parameters.
"""

import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict

from sklearn.preprocessing import OrdinalEncoder

log = logging.getLogger("stage5")
REPORT_PATH = Path("reports/encoding")
REPORT_PATH.mkdir(parents=True, exist_ok=True)


class FeatureEncoder:
    """
    Parameters
    ----------
      onehot_frac_thresh : float
          If unique_frac ≤ this, use one‐hot (default 0.05).
      ordinal_frac_thresh : float
          If onehot_frac_thresh < unique_frac ≤ this, use ordinal (default 0.20).
      freq_frac_thresh : float
          If ordinal_frac_thresh < unique_frac ≤ this, use frequency (default 0.50).
    """

    ONEHOT_FRAC_THRESH: float = 0.05
    ORDINAL_FRAC_THRESH: float = 0.20
    FREQ_FRAC_THRESH: float = 0.50

    def __init__(
        self,
        onehot_frac_thresh: float = ONEHOT_FRAC_THRESH,
        ordinal_frac_thresh: float = ORDINAL_FRAC_THRESH,
        freq_frac_thresh: float = FREQ_FRAC_THRESH,
    ):
        self.onehot_frac_thresh = onehot_frac_thresh
        self.ordinal_frac_thresh = ordinal_frac_thresh
        self.freq_frac_thresh = freq_frac_thresh

        # Will be filled during encoding
        self.categorical_cols: List[str] = []
        self.unique_frac: Dict[str, float] = {}
        self.suggestions: List[str] = []

    @staticmethod
    def _onehot_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        return pd.get_dummies(df[cols], prefix=cols, drop_first=False, dtype=float)

    @staticmethod
    def _frequency_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        out: Dict[str, pd.Series] = {}
        for col in cols:
            freq = df[col].value_counts(normalize=True)
            out[col + "_freq"] = df[col].map(freq).fillna(0.0)
        return pd.DataFrame(out, index=df.index)

    @staticmethod
    def _ordinal_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        arr = enc.fit_transform(df[cols].astype(object))
        return pd.DataFrame(arr.astype(int), columns=cols, index=df.index)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1) Identify categorical columns.
        2) Compute unique_frac = nunique / n_rows for each.
        3) For “linear” variant:
            – one‐hot if frac ≤ onehot_frac_thresh
            – else frequency if frac ≤ freq_frac_thresh
            – else suggest target encode
        4) For “tree” variant:
            – one‐hot if frac ≤ onehot_frac_thresh
            – ordinal if onehot_frac_thresh < frac ≤ ordinal_frac_thresh
            – freq if ordinal_frac_thresh < frac ≤ freq_frac_thresh
            – else suggest target encode
        5) For “knn” variant:
            – freq‐encode all
        6) Write three CSVs in working dir:
            processed_train_linear.parquet, processed_train_tree.parquet, processed_train_knn.parquet
        7) Write a JSON summary (“encoding_report.json”) under REPORT_PATH.
        """
        df0 = df.copy().reset_index(drop=True)
        n_rows = len(df0)
        self.categorical_cols = df0.select_dtypes(
            include=["category", "object"]
        ).columns.tolist()
        self.unique_frac = {
            col: df0[col].nunique(dropna=False) / n_rows
            for col in self.categorical_cols
        }

        report: Dict[str, Dict] = {}

        # — LINEAR VARIANT —
        linear_onehot, linear_freq = [], []
        linear_sugg: List[str] = []
        for col in self.categorical_cols:
            frac = self.unique_frac[col]
            if frac <= self.onehot_frac_thresh:
                linear_onehot.append(col)
            elif frac <= self.freq_frac_thresh:
                linear_freq.append(col)
            else:
                linear_sugg.append(col)

        # Build DataFrame
        oh_lin = (
            self._onehot_encode(df0, linear_onehot)
            if linear_onehot
            else pd.DataFrame(index=df0.index)
        )
        freq_lin = (
            self._frequency_encode(df0, linear_freq)
            if linear_freq
            else pd.DataFrame(index=df0.index)
        )
        df_lin = pd.concat(
            [df0.drop(columns=self.categorical_cols), oh_lin, freq_lin], axis=1
        )
        df_lin.to_parquet("processed_train_linear.parquet", index=False)
        report["linear"] = {
            "onehot": linear_onehot,
            "frequency": linear_freq,
            "suggest_target_encode": linear_sugg,
        }
        log.info(
            f"[LINEAR] onehot={linear_onehot}, freq={linear_freq}, sugg={linear_sugg}"
        )

        # — TREE VARIANT —
        tree_onehot, tree_ordinal, tree_freq = [], [], []
        tree_sugg: List[str] = []
        for col in self.categorical_cols:
            frac = self.unique_frac[col]
            if frac <= self.onehot_frac_thresh:
                tree_onehot.append(col)
            elif frac <= self.ordinal_frac_thresh:
                tree_ordinal.append(col)
            elif frac <= self.freq_frac_thresh:
                tree_freq.append(col)
            else:
                tree_sugg.append(col)

        df_tree = df0.copy()
        if tree_onehot:
            oh2 = self._onehot_encode(df0, tree_onehot)
            df_tree = df_tree.drop(columns=tree_onehot)
            df_tree = pd.concat([df_tree, oh2], axis=1)
        if tree_ordinal:
            ord_df = self._ordinal_encode(df_tree, tree_ordinal)
            df_tree = df_tree.drop(columns=tree_ordinal)
            df_tree = pd.concat([df_tree, ord_df], axis=1)
        if tree_freq:
            freq_df2 = self._frequency_encode(df_tree, tree_freq)
            df_tree = df_tree.drop(columns=tree_freq)
            df_tree = pd.concat([df_tree, freq_df2], axis=1)

        df_tree.to_parquet("processed_train_tree.parquet", index=False)
        report["tree"] = {
            "onehot": tree_onehot,
            "ordinal": tree_ordinal,
            "frequency": tree_freq,
            "suggest_target_encode": tree_sugg,
        }
        log.info(
            f"[TREE] onehot={tree_onehot}, ordinal={tree_ordinal}, freq={tree_freq}, sugg={tree_sugg}"
        )

        # — KNN VARIANT —
        freq_all = (
            self._frequency_encode(df0, self.categorical_cols)
            if self.categorical_cols
            else pd.DataFrame(index=df0.index)
        )
        df_knn = pd.concat([df0.drop(columns=self.categorical_cols), freq_all], axis=1)
        df_knn.to_parquet("processed_train_knn.parquet", index=False)
        report["knn"] = {"frequency_all": self.categorical_cols}
        log.info(f"[KNN] freq-encode all: {self.categorical_cols}")

        # Write summary JSON
        outpath = REPORT_PATH / "encoding_report.json"
        with open(outpath, "w") as f:
            json.dump(report, f, indent=2)
        log.info(f"Encoding report → {outpath}")

        return df_lin  # return linear variant by default

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        For a new DataFrame, apply the SAME encoding decisions as train:
          – For “linear,” load processed_train_linear.parquet and reindex/harmonize columns.
          – Similarly for “tree” and “knn” if needed.
        NOTE: Here we implement only “linear” transformation for brevity.
        """
        # This is a stub. In practice, you’d load the parquet template, align columns, etc.
        raise NotImplementedError(
            "Stage5Encoder.transform() is not implemented. Use fit_transform on train."
        )


if __name__ == "__main__":
    # === Quick Self-Test ===
    df_test = pd.DataFrame(
        {
            "cat1": ["a", "b", "a", "c", "b", "b", "a", "d", "e", "f"],
            "cat2": [
                "low",
                "medium",
                "high",
                "low",
                "low",
                "medium",
                "high",
                "low",
                "medium",
                "medium",
            ],
            "num": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    )
    encoder = Stage5Encoder(
        onehot_frac_thresh=0.2, ordinal_frac_thresh=0.5, freq_frac_thresh=0.8
    )
    df_lin = encoder.fit_transform(df_test)
    print("\nLinear‐encoded DataFrame (parquet on disk):")
    print(df_lin.head())
