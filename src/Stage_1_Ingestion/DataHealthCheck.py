import os
import json
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List
from itertools import combinations
from statsmodels.stats.outliers_influence import variance_inflation_factor
from configs import global_conf
import mlflow

logger = logging.getLogger("DataHealthLogger")
logger.setLevel(logging.INFO)


class DataHealthCheck:
    """
    ✅ World-class data diagnostics tool.
    - Outputs JSON, Markdown, and diagnostic PNG charts
    - Supports MLflow integration
    - Use `.get_results()` and `.get_chart_paths()` for PipelineReporter
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "label",
        batch_col: Optional[str] = None,
        datetime_cols: Optional[List[str]] = None,
        max_charts: int = 50,
        save_dir: str = global_conf.HEALTH_CHECK_REPORT_PATH,
    ):
        self.df = df.copy()
        self.target_col = target_col
        self.batch_col = batch_col
        self.datetime_cols = datetime_cols or []
        self.max_charts = max_charts
        self.save_dir = save_dir
        self.results = {}

        os.makedirs(save_dir, exist_ok=True)
        logger.info("DataHealthCheck initialized.")

    def run_all(self):
        """Run all checks and write report files."""
        self._check_dimensions()
        self._check_missingness()
        self._check_dtypes()
        self._check_skewness()
        self._check_cardinality()
        self._check_outliers()
        self._check_correlations()
        self._check_vif()
        self._check_target_imbalance()
        self._check_datetime_coverage()
        self._check_batch_distribution()
        self._save_all_reports()
        logger.info("DataHealthCheck completed.")

    def _check_dimensions(self):
        n, p = self.df.shape
        self.results["dimensions"] = {
            "rows": n,
            "columns": p,
            "ratio": round(p / n, 2),
            "regime": "p≫n" if p > n else "n≫p" if n > p else "p≈n",
        }

    def _check_missingness(self):
        miss = self.df.isna().mean()
        self.results["missingness"] = (
            miss.sort_values(ascending=False).head(20).to_dict()
        )
        self._plot_bar(miss, "Missingness per column", "missingness.png")

    def _check_dtypes(self):
        self.results["dtypes"] = self.df.dtypes.value_counts().astype(
            str).to_dict()

    def _check_skewness(self):
        num = self.df.select_dtypes(include=np.number)
        skew = num.skew()
        self.results["skewness"] = skew.sort_values(
            ascending=False).head(10).to_dict()
        self._plot_bar(
            skew.abs(), "Skewness of numeric columns", "skewness.png")

    def _check_cardinality(self):
        cat = self.df.select_dtypes(include=["object", "category"])
        card = {col: cat[col].nunique() for col in cat.columns}
        top_card = dict(
            sorted(card.items(), key=lambda x: x[1], reverse=True)[:10])
        self.results["cardinality"] = top_card
        self._plot_bar(
            pd.Series(top_card), "Categorical cardinality", "cardinality.png"
        )

    def _check_outliers(self):
        out = {}
        num = self.df.select_dtypes(include=np.number)
        for col in num.columns[: self.max_charts]:
            q1, q3 = np.nanpercentile(num[col], [25, 75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            out[col] = ((num[col] < low) | (num[col] > high)).sum()
        self.results["outliers"] = dict(
            sorted(out.items(), key=lambda x: x[1], reverse=True)[:10]
        )

    def _check_correlations(self):
        num = self.df.select_dtypes(include=np.number)
        if num.shape[1] > 1:
            corr = num.corr().abs()
            pairs = [
                (i, j, corr.loc[i, j])
                for i, j in combinations(corr.columns, 2)
                if corr.loc[i, j] > 0.9
            ]
            self.results["correlations"] = [
                {"pair": (i, j), "corr": round(c, 2)}
                for i, j, c in sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
            ]
            self._plot_heatmap(corr, "correlation_matrix.png")

    def _check_vif(self):
        num = self.df.select_dtypes(include=np.number).dropna().iloc[:, :20]
        X = num.values
        vifs = {
            num.columns[i]: variance_inflation_factor(X, i) for i in range(X.shape[1])
        }
        self.results["vif"] = dict(
            sorted(vifs.items(), key=lambda x: x[1], reverse=True)[:10]
        )

    def _check_target_imbalance(self):
        if self.target_col in self.df.columns:
            counts = self.df[self.target_col].value_counts(normalize=True)
            self.results["imbalance"] = counts.round(3).to_dict()
            self._plot_bar(counts, "Target Imbalance", "imbalance.png")

    def _check_datetime_coverage(self):
        info = {}
        for col in self.datetime_cols:
            if col in self.df:
                dates = pd.to_datetime(self.df[col], errors="coerce")
                info[col] = {
                    "parsed_pct": round(dates.notna().mean(), 2),
                    "min": str(dates.min()),
                    "max": str(dates.max()),
                }
        self.results["datetime_coverage"] = info

    def _check_batch_distribution(self):
        if self.batch_col and self.batch_col in self.df:
            counts = self.df[self.batch_col].value_counts(normalize=True)
            self.results["batch_distribution"] = counts.round(3).to_dict()
            self._plot_bar(counts, "Batch Distribution",
                           "batch_distribution.png")

    def _plot_bar(self, series: pd.Series, title: str, filename: str):
        try:
            plt.figure(figsize=(10, 6))
            sns.barplot(
                x=series.values[: self.max_charts], y=series.index[: self.max_charts]
            )
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, filename))
            plt.close()
        except Exception as e:
            logger.warning(f"Plotting failed for {title}: {e}")

    def _plot_heatmap(self, corr_df, filename: str):
        try:
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_df, cmap="coolwarm", annot=False)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_dir, filename))
            plt.close()
        except Exception as e:
            logger.warning(f"Heatmap failed: {e}")

    @staticmethod
    def _convert_json_serializable(obj):
        """Recursively convert pandas/numpy types to Python types."""
        if isinstance(obj, dict):
            return {
                str(
                    DataHealthCheck._convert_json_serializable(k)
                ): DataHealthCheck._convert_json_serializable(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, (list, tuple)):
            return [DataHealthCheck._convert_json_serializable(i) for i in obj]
        elif hasattr(obj, "item"):  # catches numpy scalars like np.int64, np.float64
            return obj.item()
        else:
            return obj

    def _save_all_reports(self):
        json_path = os.path.join(self.save_dir, "report.json")
        md_path = os.path.join(self.save_dir, "report.md")

        serializable_results = self._convert_json_serializable(self.results)

        with open(json_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        with open(md_path, "w") as f:
            f.write("# 📋 Data Health Report\n")
            for key, val in self.results.items():
                f.write(f"\n## {key}\n")
                if isinstance(val, dict):
                    for k, v in val.items():
                        f.write(f"- **{k}**: {v}\n")
                elif isinstance(val, list):
                    for item in val:
                        f.write(f"- {item}\n")

        if mlflow:
            try:
                mlflow.log_artifact(json_path)
                mlflow.log_artifact(md_path)
                for f in os.listdir(self.save_dir):
                    if f.endswith(".png"):
                        mlflow.log_artifact(os.path.join(self.save_dir, f))
            except Exception as e:
                logger.warning(f"MLflow logging failed: {e}")

    def get_results(self) -> dict:
        """Return the health check results dictionary."""
        return self.results

    def get_chart_paths(self) -> list:
        """Return list of all generated PNG chart paths."""
        return [
            os.path.join(self.save_dir, f)
            for f in os.listdir(self.save_dir)
            if f.endswith(".png")
        ]
