# optuna_hpo/reporter.py

import os
import optuna.visualization as vis
import pandas as pd
from pathlib import Path
import mlflow


class OptunaReporter:
    """
    Generates visual & tabular reports for each model's Optuna study,
    and logs them optionally to MLflow.
    """

    def __init__(self, best_models, output_dir="reports", ensemble=None, mlflow_run=None):
        """
        best_models: dict of model_name -> (best_model, study)
        ensemble: trained ensemble model (optional)
        mlflow_run: active MLflow run (optional)
        """
        self.best_models = best_models
        self.ensemble = ensemble
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.mlflow_run = mlflow_run

    def generate_reports(self):
        report_data = {}

        for name, (_, study) in self.best_models.items():
            print(f"📊 Generating report for model: {name}")
            report_data[name] = {
                "best_value": study.best_value,
                "best_params": study.best_params,
            }

            vis_plots = [
                ("opt_history", vis.plot_optimization_history),
                ("param_importance", vis.plot_param_importances),
                ("slice", vis.plot_slice),
                ("parallel_coords", vis.plot_parallel_coordinate),
            ]

            for suffix, plot_func in vis_plots:
                try:
                    fig = plot_func(study)
                    out_file = self.output_dir / f"{name}_{suffix}.html"
                    fig.write_html(str(out_file))
                    if self.mlflow_run:
                        mlflow.log_artifact(str(out_file))
                except Exception as e:
                    print(
                        f"[WARN] Could not create {suffix} plot for {name}: {e}")

        if self.ensemble:
            print("📊 Adding ensemble summary")
            report_data["ensemble"] = str(self.ensemble)

        df_report = pd.DataFrame(report_data).T
        csv_path = self.output_dir / "summary.csv"
        df_report.to_csv(csv_path)
        if self.mlflow_run:
            mlflow.log_artifact(str(csv_path))

        print(f"✅ Reports saved in {self.output_dir.resolve()}")
