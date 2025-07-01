import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Any, Dict, Union, Optional
import logging
import os
from configs import global_conf
try:
    import mlflow
except ImportError:
    mlflow = None

log = logging.getLogger("PipelineReporter")
log.setLevel(logging.INFO)


class PipelineReporter:
    """
    World-class reporting utility for ML pipeline diagnostics.
    - Supports OutlierDetector, MissingImputer, and any compatible class.
    - Outputs Markdown + JSON + interactive HTML reports.
    - Integrates with MLflow, supports PerfMixin summary.
    """

    def __init__(
        self,
        max_charts: int = 50,
        report_dir: Union[str, Path] = global_conf.PREPROCESSOR_REPORT_PATH,
        enable_mlflow: bool = True,
    ):
        self.max_charts = max_charts
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir = self.report_dir / "figures"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.enable_mlflow = enable_mlflow
        self.components: Dict[str, Any] = {}
        self.component_reports: Dict[str, Dict] = {}
        self.chart_paths: Dict[str, list] = {}
        self.static_reports: Dict[str, Dict] = {}

    def _report_outlier_detector(self, name: str, component: Any) -> Dict:
        report = component.report if hasattr(component, "report") else {}
        outlier_indices = set(report.get(
            "real_outliers", {}).get("indices", []))
        df = getattr(component, "df", None)
        scores = getattr(component, "votes_table_",
                         {}).get("total_votes", None)
        cols = getattr(component, "numeric_cols", [])

        charts = []
        if df is not None and scores is not None:
            stds = {
                col: df[col].loc[list(outlier_indices)].std()
                for col in cols
                if col in df.columns
            }
            top_cols = sorted(stds.items(), key=lambda x: x[1], reverse=True)[
                : self.max_charts
            ]
            for col, _ in top_cols:
                chart_path = self._plot_histogram(
                    df[col],
                    title=f"{name}: {col} (highlighted outliers)",
                    hue=df.index.isin(outlier_indices),
                    filename=f"{name}_{col}_outliers.png",
                )
                charts.append(chart_path)
        return {"charts": charts, "summary": report}

    def _report_missing_imputer(self, name: str, component: Any) -> Dict:
        report = component.report if hasattr(component, "report") else {}
        df = getattr(component, "df", None)
        cols = getattr(component, "numeric_cols", [])

        charts = []
        if df is not None:
            missing_frac = df[cols].isna().mean().sort_values(ascending=False)
            top_cols = missing_frac.head(self.max_charts).index
            for col in top_cols:
                chart_path = self._plot_histogram(
                    df[col],
                    title=f"{name}: {col} (missing values)",
                    filename=f"{name}_{col}_missing.png",
                )
                charts.append(chart_path)
        return {"charts": charts, "summary": report}

    def register(
        self, name: str, component: Any = None, report: dict = None, charts: list = None
    ):
        """Register either a component (with .report or .get_pipeline_report()) OR a precomputed report dict."""
        if component is not None:
            self.components[name] = component
        elif report is not None:
            self.static_reports[name] = report
            if charts:
                self.chart_paths[name] = charts

    def _plot_histogram(
        self,
        data: pd.Series,
        title: str,
        filename: str,
        hue: Optional[pd.Series] = None,
    ) -> str:
        plt.figure(figsize=(6, 4))
        if hue is not None:
            sns.histplot(data, hue=hue, kde=True, palette="muted")
        else:
            sns.histplot(data, kde=True, color="steelblue")
        plt.title(title)
        plt.tight_layout()
        filepath = self.plots_dir / filename
        plt.savefig(filepath)
        plt.close()
        return str(filepath)

    def _extract_summary_and_charts(self, name: str, comp: Any) -> Dict:
        summary = {}
        charts = []

        # Prefer .get_pipeline_report() if present
        if hasattr(comp, "get_pipeline_report"):
            try:
                result = comp.get_pipeline_report(report_dir=self.report_dir)
                if isinstance(result, dict):
                    summary = result.get("summary", result)
                    charts = result.get("charts", [])
            except Exception as e:
                log.warning(f"[{name}] Failed get_pipeline_report: {e}")
        elif hasattr(comp, "report"):
            try:
                summary = comp.report()
            except Exception as e:
                log.warning(f"[{name}] Failed .report(): {e}")

        # Optional visual diagnostics (e.g. histogram of numeric features)
        df = getattr(comp, "df", None)
        numeric_cols = getattr(comp, "numeric_cols", [])
        if isinstance(df, pd.DataFrame) and numeric_cols:
            try:
                top_cols = (
                    df[numeric_cols]
                    .std()
                    .sort_values(ascending=False)
                    .head(self.max_charts)
                    .index
                )
                for col in top_cols:
                    chart = self._plot_histogram(
                        df[col],
                        title=f"{name}: {col} distribution",
                        filename=f"{name}_{col}_dist.png",
                    )
                    charts.append(chart)
            except Exception as e:
                log.warning(f"[{name}] Failed to plot visuals: {e}")

        return {"summary": summary, "charts": charts}

    def _handle_perf(self, component: Any, name: str) -> Optional[str]:
        if hasattr(component, "report"):
            try:
                perf = component.report()
                if perf:
                    path = self.report_dir / f"{name}_perf_summary.json"
                    with open(path, "w") as f:
                        json.dump(perf, f, indent=2)
                    return str(path)
            except Exception:
                pass
        return None

    def generate_report(self, output_name: str = "pipeline_report") -> Dict:
        final_report = {}

        markdown = ["# 🧾 Full Pipeline Diagnostic Report\n"]

        # Static report blocks
        for name, report in self.static_reports.items():
            markdown.append(f"## 📌 {name} (static report)\n")
            markdown.append(
                "```json\n" + json.dumps(report, indent=2) + "\n```\n")
            for chart in self.chart_paths.get(name, []):
                markdown.append(f"![{name}]({chart})\n")
            final_report[name] = {
                "summary": report,
                "charts": self.chart_paths.get(name, []),
            }

        # Component-based blocks
        for name, comp in self.components.items():
            cls_name = comp.__class__.__name__
            markdown.append(f"## 📦 {name} ({cls_name})\n")

            if hasattr(comp, "plot_summary"):
                try:
                    chart_path = comp.plot_summary()
                    markdown.append(f"![{name}]({chart_path})\n")
                    if self.enable_mlflow and mlflow:
                        mlflow.log_artifact(chart_path)
                except Exception as e:
                    log.warning(f"[{name}] plot_summary failed: {e}")

            result = self._extract_summary_and_charts(name, comp)
            if result["summary"]:
                markdown.append(
                    "```json\n"
                    + json.dumps(result["summary"], indent=2) + "\n```\n"
                )
            for chart in result["charts"]:
                markdown.append(f"![{name}]({chart})\n")

            final_report[name] = result

            perf_file = self._handle_perf(comp, name)
            if perf_file:
                markdown.append(f"\n📊 **Perf Summary**: `{perf_file}`\n")

        # Save all files
        json_path = self.report_dir / f"{output_name}.json"
        # md_path = self.report_dir / f"{output_name}.md"
        # html_path = self.report_dir / f"{output_name}.html"

        with open(json_path, "w") as f:
            json.dump(final_report, f, indent=2)
        # with open(md_path, "w") as f:
        #     f.write("\n".join(markdown))
        # with open(html_path, "w") as f:
        #     f.write("<html><body>"
        #             + "<br><hr><br>".join(markdown) + "</body></html>")

        if self.enable_mlflow and mlflow:
            mlflow.log_artifact(json_path)
            # mlflow.log_artifact(md_path)
            # mlflow.log_artifact(html_path)
            for sect in final_report.values():
                for chart in sect.get("charts", []):
                    mlflow.log_artifact(chart)

        return final_report
