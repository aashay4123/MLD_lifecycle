#!/usr/bin/env python3
import os
import sys
import yaml
import argparse
import importlib.util
from pathlib import Path


def load_global_conf(conf_path: Path):
    """
    Dynamically load a Python module from the given path
    and return it.
    """
    spec = importlib.util.spec_from_file_location(
        "global_conf", str(conf_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_constants(module):
    """
    Return a dict of all-uppercase attributes in the module.
    """
    return {
        name: value
        for name, value in vars(module).items()
        if name.isupper()
    }


def build_structure(constants: dict):
    """
    Build both a freeform 'globals' section of *all* constants
    and a hand‑crafted mapping for DVC stages.
    """
    struct = {
        # for ad‑hoc use
        "globals": constants,
        # explicit stage params
        "ingestion": {
            "csv_path": constants.get("CSV_PATH"),
            "target_col": constants.get("DATASET_TARGET_COLUMN_NAME"),
            "raw_parquet_path": constants.get("RAW_PARQUET_PATH"),
        },
        "health_check": {
            "report_path": constants.get("HEALTH_CHECK_REPORT_PATH"),
        },
        "eda": {
            "epda_report_path": constants.get("EPDA_REPORT_PATH"),
        },
        "splitting": {
            "split_parquet_path": constants.get("SPLIT_PARQUET_PATH"),
            "random_state": constants.get("RANDOM_STATE"),
            "train_test_split": constants.get("TRAIN_TEST_SPLIT"),
        },
        "preprocessing": {
            "preprocessor_report_path": constants.get("PREPROCESSOR_REPORT_PATH"),
            "train_parquet": constants.get("TRAIN_PARQUET_PATH"),
            "val_parquet": constants.get("VAL_PARQUET_PATH"),
            "test_parquet": constants.get("TEST_PARQUET_PATH"),
        },
        "dimensionality_reduction": {
            "dr_report_path": constants.get("DR_REPORT_PATH"),
        },
        "modeling": {
            "model_name": constants.get("MODEL_NAME"),
            "final_model_path": constants.get("FINAL_MODEL_PATH"),
            "artifacts_path": constants.get("MODEL_ARTIFACTS_PATH"),
        },
        "baseline": {
            "baseline_report_path": constants.get("BASELINE_REPORT_PATH"),
        },
        "evaluation": {
            "min_train_accuracy": constants.get("MIN_TRAIN_ACCURACY"),
            "min_test_accuracy": constants.get("MIN_TEST_ACCURACY"),
            "max_serve_train_diff": constants.get("MAX_SERVE_TRAIN_ACCURACY_DIFF"),
            "max_serve_test_diff": constants.get("MAX_SERVE_TEST_ACCURACY_DIFF"),
        },
        "monitoring": {
            "pipeline_log_path": constants.get("PIPELINE_LOGS_PATH"),
            "mlflow_report_path": constants.get("MLFLOW_REPORT_PATH"),
            "optuna_report_path": constants.get("OPTUNA_REPORT_PATH"),
            "logs_path": constants.get("LOGS_PATH"),
        },
        "hpo": {
            "ensemble_top_n": constants.get("ENSEMBLE_TOP_N"),
            "scoring": constants.get("DEFAULT_SCORING"),
            "ensemble_artifact_path": constants.get("MLFLOW_ENSEMBLE_ARTIFACT_PATH"),
            "n_trials": constants.get("DEFAULT_N_TRIALS"),
            "cv_folds": constants.get("DEFAULT_CV_FOLDS"),
            "metric_classification": constants.get("DEFAULT_METRIC_CLASSIFICATION"),
            "metric_regression": constants.get("DEFAULT_METRIC_REGRESSION"),
        },
    }
    return struct


def main():
    p = argparse.ArgumentParser(
        description="Generate params.yaml from your global_conf.py"
    )
    p.add_argument(
        "--conf", "-c",
        type=Path,
        default=Path(__file__).parents[2] / "configs" / "global_conf.py",
        help="path to global_conf.py"
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).parents[2] / "params.yaml",
        help="where to write params.yaml"
    )
    args = p.parse_args()

    if not args.conf.exists():
        sys.exit(f"ERROR: config file not found at {args.conf}")

    module = load_global_conf(args.conf)
    consts = extract_constants(module)
    structure = build_structure(consts)

    # ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w") as f:
        yaml.dump(structure, f, sort_keys=False)

    print(f"➡️  Generated params.yaml at {args.output}")


if __name__ == "__main__":
    main()
