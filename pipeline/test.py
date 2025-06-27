import mlflow
import warnings
import pandas as pd
from zenml import pipeline
from src.Stage_1_Ingestion.data_loaders import dataLoader, dataCheck
from src.Stage_2_EPD_Analysis.PED_Analysis import UnifiedPEDAnalyze
from src.Stage_3_Split_data.data_split import data_splitter, baseline
from src.Stage_4_Preprocessor.preprocessor import missing_imputer, outlier_detector
from src.utils.PipelineReporter import PipelineReporter
from configs import global_conf
warnings.filterwarnings("ignore")


@pipeline
def main():
    # Load the data
    mlflow.set_tracking_uri('http://localhost:5000')
    data_loader = dataLoader(
        global_conf.CSV_PATH, "Breast Cancer")
    # reporter = PipelineReporter()

    # TODO: Add parellel processing for data Analysis

    dataCheck(data_loader)
    print("Data health check completed successfully.")
    UnifiedPEDAnalyze(data_loader)
    train, test, val = data_splitter(data=data_loader)
    baseline(train, val)
    print("Data split and baseline model training completed successfully.")

    # Register components with `.report` or `.get_pipeline_report()` method
    train, test, val = missing_imputer(train, test, val)
    train, test, val = outlier_detector(train, test, val)

    # Generate Markdown + HTML + JSON reports and log to MLflow
    # report = reporter.generate_report(output_name="final_pipeline_report")
    print("Pipeline reporting completed successfully.")
