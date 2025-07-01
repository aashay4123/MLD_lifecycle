# These global parameters should be the same across all workflow stages.
CSV_PATH = "Data/merged_all_3_datasets.csv"
DATASET_TARGET_COLUMN_NAME = "label"

RAW_PARQUET_PATH = "Data/raw"
SPLIT_PARQUET_PATH = "Data/split"


INJESTION_LOGS_PATH = "reports/logs/ingest.log"
HEALTH_CHECK_REPORT_PATH = "reports/health_report"
EPDA_REPORT_PATH = "reports/epda"
BASELINE_REPORT_PATH = "reports/baseline_report"
PREPROCESSOR_REPORT_PATH = "reports/preprocessor_report"

MODEL_ARTIFACTS_PATH = "reports/Models"

PIPELINE_LOGS_PATH = "reports/logs/pipeline_monitor.log"
MLFLOW_REPORT_PATH = "reports/mlflow/"

# Model training and evaluation parameters
RANDOM_STATE = 23
TRAIN_TEST_SPLIT = 0.2
MIN_TRAIN_ACCURACY = 0.9
MIN_TEST_ACCURACY = 0.9
MAX_SERVE_TRAIN_ACCURACY_DIFF = 0.1
MAX_SERVE_TEST_ACCURACY_DIFF = 0.05
WARNINGS_AS_ERRORS = False
MODEL_NAME = "gitflow_model"
