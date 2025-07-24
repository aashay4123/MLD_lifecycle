# /merged_missing_imputer.py
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from statsmodels.imputation.mice import MICEData
from statsmodels.stats.missing import test_missing

from configs import global_conf
from src.utils.perfkit import PerfMixin, perfclass

# ==== Global Constants ====
REPORT_DIR = Path(global_conf.PREPROCESSOR_REPORT_PATH) / "missingness"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL_PATH = Path(global_conf.MODEL_ARTIFACTS_PATH) / "missing_model.pkl"
DEFAULT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
RARE_LEVEL_CUTOFF = 0.01
MAX_MISSING_FRAC_DROP = 0.90
VARIANCE_RATIO_CUTOFF = 0.50
COV_CHANGE_CUTOFF = 0.20


class MissingnessAnalyzer:
    """
    Advanced missingness analysis using logistic regression & Little's MCAR test.
    """

    @staticmethod
    def analyze(df: pd.DataFrame) -> Dict[str, Dict]:
        results = {}

        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue

            y = df[col].isna().astype(int)
            X = df.drop(columns=[col])
            try:
                X = X.notna().astype(int)
            except Exception:
                X = X.notna().applymap(int)

            if X.shape[1] == 0 or y.sum() == 0:
                results[col] = {
                    "fraction_missing": y.mean(),
                    "mechanism": "undetermined",
                    "p_value": np.nan,
                }
                continue

            try:
                lr = LogisticRegression(solver="liblinear", max_iter=100)
                lr.fit(X, y)

                Xmat = X.values
                XtX = Xmat.T @ Xmat
                inv_XtX = np.linalg.inv(XtX + np.eye(X.shape[1]) * 1e-6)
                z_scores, pvals = [], []
                for idx, coef in enumerate(lr.coef_[0]):
                    se = np.sqrt(inv_XtX[idx, idx])
                    z = coef / se if se > 0 else 0
                    pvals.append(2 * (1 - stats.norm.cdf(abs(z))))

                p_combined = max(pvals)
                mechanism = "MCAR" if p_combined > ALPHA else "MAR/MNAR"

                results[col] = {
                    "fraction_missing": y.mean(),
                    "mechanism": mechanism,
                    "p_value": p_combined,
                }
            except Exception:
                results[col] = {
                    "fraction_missing": y.mean(),
                    "mechanism": "fit_failed",
                    "p_value": np.nan,
                }

        # Optional: Little’s MCAR global test if available
        try:
            res = test_missing(df)
            results["_global_Little_MCAR"] = {
                "statistic": res.statistic,
                "p_value": res.pvalue,
                "is_MCAR": bool(res.pvalue > ALPHA),
            }
        except Exception:
            results["_global_Little_MCAR"] = {
                "statistic": np.nan,
                "p_value": np.nan,
                "is_MCAR": False,
            }

        return results

    @staticmethod
    def save_report(results: Dict, outpath: Union[str, Path], html: bool = False):
        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)

        if html:
            html_lines = ["<html><body><h2>Missingness Analysis</h2><table border='1'>"]
            html_lines.append(
                "<tr><th>Column</th><th>Fraction Missing</th><th>P-Value</th><th>Mechanism</th></tr>"
            )
            for col, vals in results.items():
                if col.startswith("_global"):
                    continue
                frac = f"{vals['fraction_missing']:.2%}"
                pval = (
                    f"{vals['p_value']:.4f}" if vals["p_value"] is not None else "N/A"
                )
                mech = vals["mechanism"]
                html_lines.append(
                    f"<tr><td>{col}</td><td>{frac}</td><td>{pval}</td><td>{mech}</td></tr>"
                )
            html_lines.append("</table></body></html>")
            outpath.with_suffix(".html").write_text("\n".join(html_lines))
        else:
            with open(outpath.with_suffix(".json"), "w") as f:
                json.dump(results, f, indent=2)
