from typing import Any, Dict, Tuple

import great_expectations as ge
import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import (
    ClassificationPerformancePreset,
    DataDriftPreset,
    DataQualityPreset,
)
from evidently.report import Report
from great_expectations.checkpoint import CheckpointResult
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from zenml import pipeline, step

console = Console()

DataFrame = pd.DataFrame


@step
def ge_validator_step(data: DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """
    Step for data validation using Great Expectations.
    Runs a checkpoint and returns success status and detailed results.
    """
    console.rule("[bold blue]Great Expectations Validation Step[/bold blue]")
    context = ge.data_context.DataContext()

    checkpoint_name = "my_checkpoint"

    # Use batch request with runtime batch_data (recommended way)
    batch_request = {
        "datasource_name": "my_datasource",
        "data_connector_name": "my_data_connector",
        "data_asset_name": "my_data_asset",
        "runtime_parameters": {"batch_data": data},
        "batch_identifiers": {"default_identifier_name": "default_identifier"},
    }

    try:
        results: CheckpointResult = context.run_checkpoint(
            checkpoint_name=checkpoint_name,
            batch_request=batch_request,
            run_name="ge_validation_run",
        )
        success = results["success"]
    except Exception as e:
        console.print(f"[red]GE validation failed with exception: {e}[/red]")
        return False, {"error": str(e)}

    # Display summary of validation results in a table
    table = Table(title="GE Validation Results Summary", show_lines=True)
    table.add_column("Expectation", justify="left", style="cyan", no_wrap=True)
    table.add_column("Success", justify="center", style="green")
    table.add_column("Details", justify="left", style="magenta")

    for res in results.list_validation_results():
        exp_name = res.expectation_config.expectation_type
        exp_success = "✅" if res.success else "❌"
        detail = str(res.result)
        # Limit detail length to keep table readable
        if len(detail) > 50:
            detail = detail[:47] + "..."
        table.add_row(exp_name, exp_success, detail)

    console.print(table)
    console.print(
        f"[bold green]Great Expectations validation success: {success}[/bold green]\n"
    )
    return success, results


@step
def evidently_validator_step(
    reference_data: DataFrame, current_data: DataFrame
) -> Tuple[bool, str]:
    """
    Runs Evidently report for data drift, quality, and classification performance.
    Returns success flag and path to HTML report.
    """
    console.rule("[bold blue]Evidently Validation Step[/bold blue]")
    column_mapping = ColumnMapping()
    # TODO: Adjust column mapping as per dataset (e.g., column_mapping.target = "target")

    report = Report(
        metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
            ClassificationPerformancePreset(),
        ]
    )
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )

    report_path = "evidently_report.html"
    report.save_html(report_path)

    # Enhanced safe extraction of drift and quality flags
    metrics = report.as_dict().get("metrics", [])
    drift_detected = False
    quality_issues = False

    for metric in metrics:
        result = metric.get("result", {})
        if isinstance(result, dict):
            drift = result.get("dataset_drift", False)
            issues = result.get("quality_issues", False)
            if drift:
                drift_detected = True
            if issues:
                quality_issues = True

    success = not (drift_detected or quality_issues)
    status = "[green]PASSED[/green]" if success else "[red]FAILED[/red]"
    console.print(f"Evidently Validation: {status}")
    console.print(f"Report saved to: [bold blue]{report_path}[/bold blue]\n")
    return success, report_path


@step
def training_step(data: DataFrame, labels: pd.Series) -> str:
    """
    Trains a RandomForestClassifier on the input data.
    Prints training and validation accuracy.
    Returns a model version string (placeholder).
    """
    console.rule("[bold blue]Model Training Step[/bold blue]")
    X_train, X_val, y_train, y_val = train_test_split(
        data, labels, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    val_score = model.score(X_val, y_val)

    console.print(f"Training accuracy: [bold green]{train_score:.4f}[/bold green]")
    console.print(f"Validation accuracy: [bold green]{val_score:.4f}[/bold green]")
    # TODO: Save model to registry or artifact store and return model version or URI
    return "model_v1"


@pipeline
def full_pipeline(
    reference_data: DataFrame, current_data: DataFrame, labels: pd.Series
):
    """
    Full pipeline combining multiple validation steps and training.
    Aborts training if any validation fails.
    """
    ge_success, ge_results = ge_validator_step(data=current_data)
    evidently_success, evidently_report = evidently_validator_step(
        reference_data=reference_data, current_data=current_data
    )

    # Proceed to training only if all validations pass
    if ge_success and evidently_success:
        training_step(data=current_data, labels=labels)
    else:
        # Summarize failed validators for easier debugging
        failures = []
        if not ge_success:
            failures.append("Great Expectations")
        if not evidently_success:
            failures.append("Evidently")

        failed_str = ", ".join(failures)
        console.print(
            f"[bold red]Pipeline aborted! Failed validations: {failed_str}[/bold red]"
        )
        raise RuntimeError(f"Pipeline validation failed: {failed_str}")


# Example usage (for local or direct run)
if __name__ == "__main__":
    import sys

    # TODO: Replace with actual data loading logic and paths
    reference_df = pd.read_csv("reference.csv")
    current_df = pd.read_csv("current.csv")
    labels = current_df.pop("target")

    pipeline_instance = full_pipeline(
        reference_data=reference_df, current_data=current_df, labels=labels
    )
    pipeline_instance.run()
