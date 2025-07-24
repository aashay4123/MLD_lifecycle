import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import BaseEstimator, TransformerMixin
from configs import global_conf


class LeakageDetector(BaseEstimator, TransformerMixin):
    """
    A single, pipeline‐friendly transformer that performs these leakage checks:

      1) Pearson correlation (numeric → numeric target) or “constant‐target‐in‐group”
         (categorical → categorical target) as before.
      2) AUC(target‐leakage): For each feature, compute AUC(feature → target).
         If AUC ≥ self.auc_threshold, flag as “target leakage.”
      3) AUC(train/test separation): concatenate train/test with a 0/1 label,
         compute AUC(feature → is_train); if ≥ self.auc_threshold, flag as “train/test leakage.”
      4) “Unseen categories” in categorical features (test vs train), as before.

    Any findings are collected in self.leakage_report_ after .fit(...).  .transform(X)
    just returns X unchanged so you can integrate it into an sklearn Pipeline.

    Example:
        detector = LeakageDetector(auc_threshold=0.99, corr_threshold=0.99)
        detector.fit(
            X_train, y_train,
            X_test=X_val, y_test=y_val,
            categorical_cols=[…], numeric_cols=[…]
        )
        print(detector.leakage_report_)
    """

    def __init__(
        self,
        auc_threshold: float = 0.99,
        corr_threshold: float = 0.99,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        auc_threshold : float (0 < auc_threshold < 1)
            If a feature’s one‐vs‐rest AUC (feature → target) or AUC(feature → is_train)
            exceeds this value, it’s flagged as potential leakage.
        corr_threshold : float (0 < corr_threshold < 1)
            If abs(Pearson‐corr(feature, target)) ≥ corr_threshold for a numeric target,
            or if any categorical level perfectly predicts the target, then we flag that too.
        verbose : bool
            If True, prints warnings when leakage is detected.
        """
        self.auc_threshold = auc_threshold
        self.corr_threshold = corr_threshold
        self.verbose = verbose
        self.leakage_report_: Dict[str, Union[Dict, List]] = {}
        self._fitted = False

    def _is_classification_target(self, y: pd.Series) -> bool:
        return not pd.api.types.is_numeric_dtype(y.dtype) or y.nunique() <= 2

    def _check_pearson_or_constant_group(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> None:
        """
        1) If target is numeric with >2 levels: compute Pearson(feature, y).
           If |corr| ≥ corr_threshold, flag that feature.
        2) Else (classification): For each feature (cat or num), check if any single
           value or bin of that feature yields only one target class (i.e. constant‐target group).
        """
        corr_leaks: Dict[str, float] = {}
        constant_group_leaks: Dict[str, List] = {}

        is_classification = self._is_classification_target(y)

        if not is_classification:
            # numeric (regression) target
            for col in numeric_cols:
                if X[col].var(ddof=0) == 0:
                    continue
                try:
                    corr = np.corrcoef(X[col].astype(float), y.astype(float))[0, 1]
                except Exception:
                    continue
                if np.isnan(corr):
                    continue
                if abs(corr) >= self.corr_threshold:
                    corr_leaks[col] = float(corr)
        else:
            # classification target
            for col in numeric_cols + categorical_cols:
                series = X[col]
                temp = pd.DataFrame({"feature": series, "target": y})
                total_rows = len(temp)
                min_bin_size = 30  # Adjust as needed

                n_bins = max(2, min(10, len(temp) // min_bin_size))

                try:
                    temp["bin"] = pd.qcut(temp["feature"], q=n_bins, duplicates="drop")
                    groups = temp.groupby("bin")["target"]
                except Exception:
                    groups = temp.groupby("feature")["target"]

                bad_vals = []
                for val, grp in groups:
                    if len(grp) < min_bin_size:
                        continue
                    frac = len(grp) / total_rows
                    if grp.nunique(dropna=False) == 1 and frac >= 0.01:
                        bad_vals.append(val)

                n_flagged_rows = sum(len(groups.get_group(b)) for b in bad_vals)
                if len(bad_vals) >= 2 and n_flagged_rows / total_rows >= 0.1:
                    constant_group_leaks[col] = bad_vals

        self.leakage_report_["corr_or_constant_group"] = {
            "high_corr_numeric": corr_leaks,
            "constant_target_in_group": constant_group_leaks,
        }

    def _check_target_leakage_auc(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> None:
        """
        For each feature in numeric_cols + categorical_cols:
          – If numeric: scale to [0,1], compute AUC(feature → y).
          – If categorical: one‐hot each level, compute AUC(dummy → y) per level, take max.
        If max AUC ≥ auc_threshold, flag as potential leakage.
        """
        from sklearn.metrics import roc_auc_score

        leaky_feats: List[str] = []
        auc_scores: Dict[str, float] = {}

        is_classification = self._is_classification_target(y)

        for col in numeric_cols + categorical_cols:
            try:
                if col in categorical_cols:
                    # one‐hot encode, find per‐level AUC
                    dummies = pd.get_dummies(X[col], prefix=col, dummy_na=False)
                    per_level_aucs: List[float] = []
                    for dummy_col in dummies.columns:
                        try:
                            # if y is continuous in regression: use AUC by thresholding y? skip that case
                            auc_val = roc_auc_score(y, dummies[dummy_col])
                            per_level_aucs.append(float(auc_val))
                        except Exception:
                            continue
                    if not per_level_aucs:
                        continue
                    max_auc = max(per_level_aucs)
                else:
                    # numeric: scale to [0,1]
                    vals = X[col].astype(float)
                    if vals.nunique() <= 1:
                        continue
                    xv = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
                    auc_val = roc_auc_score(y, xv.fillna(0.5))
                    max_auc = float(auc_val)

                if max_auc >= max(0.8, self.auc_threshold):
                    auc_scores[col] = max_auc

                if max_auc >= self.auc_threshold:
                    leaky_feats.append(col)

            except Exception:
                continue

        self.leakage_report_["target_leakage_auc"] = {
            "auc_per_feature": auc_scores,
            "leaky_features": leaky_feats,
        }

    def _check_train_test_separation_auc(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        numeric_cols: List[str],
        categorical_cols: List[str],
    ) -> None:
        """
        Concatenate train_df and test_df, label train rows as 0 and test rows as 1.
        For each feature:
          – If numeric: scale to [0,1], compute AUC(feature → is_train).
          – If categorical: one‐hot encode levels, compute AUC(dummy → is_train) per level, take max.
        If max AUC ≥ auc_threshold, flag as “train/test leakage.”
        """
        from sklearn.metrics import roc_auc_score

        combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        labels = np.concatenate([np.zeros(len(train_df)), np.ones(len(test_df))])
        sep_feats: List[str] = []
        auc_scores: Dict[str, float] = {}

        for col in numeric_cols + categorical_cols:
            try:
                values = combined[col]
                if col in categorical_cols:
                    dummies = pd.get_dummies(values, prefix=col, dummy_na=False)
                    per_level_aucs: List[float] = []
                    for dummy_col in dummies.columns:
                        try:
                            auc_val = roc_auc_score(labels, dummies[dummy_col].values)
                            per_level_aucs.append(float(auc_val))
                        except Exception:
                            continue
                    if not per_level_aucs:
                        continue
                    max_auc = max(per_level_aucs)
                else:
                    # numeric → scale to [0,1]
                    xv = values.astype(float)
                    if xv.nunique() <= 1:
                        continue
                    xv_scaled = (xv - xv.min()) / (xv.max() - xv.min() + 1e-9)
                    max_auc = float(roc_auc_score(labels, xv_scaled.fillna(0.5)))
                if max_auc >= max(0.8, self.auc_threshold):
                    auc_scores[col] = max_auc

                if max_auc >= self.auc_threshold:
                    sep_feats.append(col)

            except Exception:
                continue

        self.leakage_report_["train_test_leakage_auc"] = {
            "auc_per_feature": auc_scores,
            "leaky_features": sep_feats,
        }

    def _check_unseen_categories(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> None:
        """
        For each categorical column, list which categories appear in test_df but not in train_df.
        """
        unseen: Dict[str, List] = {}
        for col in categorical_cols:
            if col not in test_df.columns:
                continue
            train_vals = set(train_df[col].dropna().unique())
            test_vals = set(test_df[col].dropna().unique())
            diff = test_vals - train_vals
            if diff:
                unseen[col] = list(diff)

        self.leakage_report_["unseen_test_categories"] = unseen

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        *,
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> "LeakageDetector":
        """
        Runs all leakage checks. Populates self.leakage_report_ with:

          • "corr_or_constant_group": {
                "high_corr_numeric": {...},
                "constant_target_in_group": {...}
             }
          • "target_leakage_auc": {
                "auc_per_feature": {...},
                "leaky_features": [...]
             }
          • "train_test_leakage_auc": {
                "auc_per_feature": {...},
                "leaky_features": [...]
             }
          • "unseen_test_categories": {col: [levels], ...}

        Parameters
        ----------
        X : DataFrame or ndarray
            Training features.
        y : Series or ndarray
            Training target.
        X_test : DataFrame or ndarray, optional
            Validation/test features.
        y_test : Series or ndarray, optional
            Validation/test target (unused except for shape consistency).
        categorical_cols : List[str], optional
            List of categorical column names in X (required if X is DataFrame; else all non-numeric).
        numeric_cols : List[str], optional
            List of numeric column names in X (required if X is DataFrame; else all non-numeric).
        """
        # Convert y to a pandas Series if necessary
        if isinstance(y, np.ndarray):
            y_train = pd.Series(y)
        else:
            y_train = y.copy()

        # Convert X to DataFrame if possible; else skip checks requiring column names
        if isinstance(X, np.ndarray):
            X_train_df = None
        else:
            X_train_df = X.copy()

        # Determine numeric/categorical columns if not provided
        if X_train_df is not None:
            if numeric_cols is None:
                numeric_cols = X_train_df.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
            if categorical_cols is None:
                categorical_cols = [
                    c for c in X_train_df.columns if c not in numeric_cols
                ]
        else:
            # If X is ndarray, we cannot run column‐wise leakage; leave these empty
            numeric_cols = [] if numeric_cols is None else numeric_cols
            categorical_cols = [] if categorical_cols is None else categorical_cols

        # 1) Pearson‐corr or “constant‐target‐in‐group” check
        if X_train_df is not None and y_train is not None:
            self._check_pearson_or_constant_group(
                X_train_df, y_train, numeric_cols, categorical_cols
            )
        else:
            self.leakage_report_["corr_or_constant_group"] = {
                "high_corr_numeric": {},
                "constant_target_in_group": {},
            }

        # 2) AUC(target‐leakage)
        if X_train_df is not None and y_train is not None:
            self._check_target_leakage_auc(
                X_train_df, y_train, numeric_cols, categorical_cols
            )
        else:
            self.leakage_report_["target_leakage_auc"] = {
                "auc_per_feature": {},
                "leaky_features": [],
            }

        # 3) AUC(train/test separation)
        if (
            X_train_df is not None
            and isinstance(X_test, pd.DataFrame)
            and y_train is not None
            and y_test is not None
        ):
            self._check_train_test_separation_auc(
                X_train_df, X_test, numeric_cols, categorical_cols
            )
        else:
            self.leakage_report_["train_test_leakage_auc"] = {
                "auc_per_feature": {},
                "leaky_features": [],
            }

        # 4) Unseen categories (if categoricals present)
        if (
            X_train_df is not None
            and isinstance(X_test, pd.DataFrame)
            and categorical_cols
        ):
            self._check_unseen_categories(X_train_df, X_test, categorical_cols)
        else:
            self.leakage_report_["unseen_test_categories"] = {}

        self._fitted = True
        self.dump_report(global_conf.HEALTH_CHECK_REPORT_PATH)
        return X

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        This transformer does not modify X; it only flags leakage during fit().
        Returns X unchanged. Must be called after .fit(...).
        """
        if not self._fitted:
            raise RuntimeError(
                "LeakageDetector must be fitted before calling transform()."
            )
        return X

    def fit_transform(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_test: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_test: Optional[Union[pd.Series, np.ndarray]] = None,
        categorical_cols: Optional[List[str]] = None,
        numeric_cols: Optional[List[str]] = None,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Convenience method: runs all leakage checks, then returns X unchanged.
        """
        df = self.fit(
            X,
            y,
            X_test=X_test,
            y_test=y_test,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
        ).transform(X)
        self.dump_report(global_conf.HEALTH_CHECK_REPORT_PATH)

        return df

    def serialize_for_json(self, obj):
        """Recursively convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self.serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.serialize_for_json(v) for v in obj]
        elif isinstance(obj, pd.Interval):
            return str(obj)
        else:
            return obj

    def dump_report(self, path: str = global_conf.HEALTH_CHECK_REPORT_PATH) -> None:
        """
        Write self.leakage_report_ to JSON for later inspection.
        """
        report_Path = f"{path}/leakage_report.json"
        Path(report_Path).parent.mkdir(parents=True, exist_ok=True)
        serializable_report = self.serialize_for_json(self.leakage_report_)
        with open(report_Path, "w") as f:
            json.dump(serializable_report, f, indent=2)
