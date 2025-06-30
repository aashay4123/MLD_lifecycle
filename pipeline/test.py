import warnings
from zenml import pipeline
from src.Stage_1_Ingestion.data_loaders import dataLoader, dataCheck
from src.Stage_2_EPD_Analysis.PED_Analysis import UnifiedPEDAnalyze
from src.Stage_3_Split_data.data_split import data_splitter, baseline
from src.Stage_4_Preprocessor.preprocessor import missing_imputer, outlier_detector
from src.utils.PipelineReporter import PipelineReporter
from configs import global_conf
import mlflow
import os
from zenml.client import Client

warnings.filterwarnings("ignore")
mlflow.set_tracking_uri("http://localhost:7000")

secret_response = Client().get_secret("mlflow_secret")

# Access as a dict of key-value pairs
secrets = secret_response.values

os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:7000"
print(mlflow.get_tracking_uri())


@pipeline(enable_cache=False)
def Pipeline():
    # ────────────── Step 1: Data Loader ──────────────
    data_df = dataLoader(global_conf.CSV_PATH, "Breast Cancer")

    # ────────────── Step 2: Data Health Check ──────────────
    # dataCheck(data_df)

    # ────────────── Step 3: Unified EPD Analyze ──────────────
    UnifiedPEDAnalyze(data_df)

    # ────────────── Step 4: Data Split ──────────────
    train, test, val = data_splitter(data=data_df)

    # ────────────── Step 5: Baseline ──────────────
    baseline(train, val)

    # ────────────── Step 6: Missing Imputer ──────────────
    train, test, val = missing_imputer(train, test, val)

    # ────────────── Step 7: Outlier Detector ──────────────
    train, test, val = outlier_detector(train, test, val)


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:7000")
    with mlflow.start_run(run_name="ML_Pipeline_Run"):
        mlflow.log_param("project", "Breast Cancer")
        mlflow.log_param("pipeline_name", "MLD_Lifecycle")
        mlflow.log_param("version", "1.0.0")

        # Run the pipeline
        pipeline_instance = Pipeline()
        pipeline_instance.run()
