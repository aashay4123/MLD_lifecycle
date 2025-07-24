import json
import os
from typing import Any, Tuple

import mlflow
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from src.Stage_2_ED_Analysis.EDAnalyzer import EDAnalyzer
from src.Stage_2_ED_Analysis.PDAnalysis import ProbabilisticAnalysis
from src.Stage_2_ED_Analysis.UnifiedEPDA import UnifiedEPDA
from src.stage_4_preprocessing.Missing_Imputer import MissingImputer
from src.stage_4_preprocessing.Outlier_Detection import OutlierDetector
from zenml.integrations.mlflow.mlflow_step_decorator import enable_mlflow
from zenml.pipelines import pipeline
from zenml.steps import Output, step

from src.Stage_1_Ingestion.data_injection.DataCollector import DataCollector
from src.Stage_1_Ingestion.DataHealthCheck import DataHealthCheck
from src.Stage_3_Split_data.base import SplitThreeWay
from src.Stage_3_Split_data.ThreeWaySplit import SplitThreeWay
from src.Stage_5_Feature_Engineering.Dimensionality_Reduction import AutoDR
from src.Stage_5_Feature_Engineering.Feature_Construction import FeatureConstructor
from src.Stage_5_Feature_Engineering.Feature_Encoding import FeatureEncoder
from src.Stage_5_Feature_Engineering.Feature_Splitting import FeatureSplitter
from src.Stage_5_Feature_Engineering.Feature_Transformer import FeatureTransformer

# from src.stage_6_modeling import ModelTrainer
from src.Stage_7_Evaluation.Evaluate_register import evaluate_and_register
from src.utils.monitor import monitor

mlflow.set_tag("zenml_pipeline", "training_pipeline_v1")


@step
@monitor(
    name="ingest_data",
    log_args=True,
    log_result=True,
    track_input_size=True,
    track_memory=True,
    retries=1,
)
def ingest_data() -> Output(data=pd.DataFrame):
    df = DataCollector(pii_mask=True).read_file("data.csv")
    return df


@step
@monitor(name="inspect_data", log_args=True, log_result=False, track_input_size=True)
def inspect_data(data: pd.DataFrame) -> Output(valid_data=pd.DataFrame):
    checker = DataHealthCheck(data)
    checker.run_all_checks()
    return data


@step
@monitor(
    name="split_data",
    log_args=True,
    log_result=False,
    track_input_size=True,
    track_memory=True,
)
def split_data(
    data: pd.DataFrame,
) -> Output(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame):
    from sklearn.model_selection import train_test_split

    train_val, test = train_test_split(data, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=0.1, random_state=42)

    # Save splits to disk
    os.makedirs("artifacts/splits", exist_ok=True)
    train.to_parquet("artifacts/splits/train.parquet", index=False)
    val.to_parquet("artifacts/splits/val.parquet", index=False)
    test.to_parquet("artifacts/splits/test.parquet", index=False)

    return train, val, test


@enable_mlflow
@step
@monitor(
    name="impute_data",
    log_args=True,
    track_input_size=True,
    track_memory=True,
    retries=1,
)
def impute_data(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Output(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame):
    imputer = MissingImputer()
    imputer.fit(train)

    # Save imputer report to MLflow
    with open("imputer_report.json", "w") as f:
        json.dump(imputer.report, f)
    mlflow.log_artifact("imputer_report.json")

    return imputer.transform(train), imputer.transform(val), imputer.transform(test)


@enable_mlflow
@step
@monitor(
    name="scale_transform",
    log_args=True,
    track_input_size=True,
    track_memory=True,
    log_result=False,
)
def scale_transform(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Output(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame):
    numeric = train.select_dtypes(include="number").columns.tolist()
    transformer = SmartFeatureTransformer(mode="adaptive", verbose=False)

    X_train = transformer.fit_transform(train, numeric)
    X_val = transformer.transform(val)
    X_test = transformer.transform(test)

    mlflow.sklearn.log_model(transformer, artifact_path="scaler_model")

    return X_train, X_val, X_test


@enable_mlflow
@step
@monitor(
    name="detect_outliers", log_args=True, track_input_size=True, track_memory=True
)
def detect_outliers(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Output(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame):
    detector = OutlierDetector()
    train_o = detector.fit_transform(train)

    with open("outlier_log.json", "w") as f:
        json.dump(detector.report, f)
    mlflow.log_artifact("outlier_log.json")

    return train_o, detector.transform(val), detector.transform(test)


@step
@monitor(name="feature_split", log_args=True, track_input_size=True, track_memory=True)
def feature_split(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Output(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame):
    splitter = AdvancedFeatureSplitterV4(n_jobs=2)
    return splitter.transform(train), splitter.transform(val), splitter.transform(test)


@step
@monitor(
    name="feature_construct", log_args=True, track_input_size=True, track_memory=True
)
def feature_construct(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Output(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame):
    constructor = AdvancedFeatureConstructorV4()
    return (
        constructor.transform(train),
        constructor.transform(val),
        constructor.transform(test),
    )


@enable_mlflow
@step
@monitor(
    name="encode",
    log_args=True,
    log_result=True,
    track_input_size=True,
    track_memory=True,
)
def encode(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame
) -> Output(train=pd.DataFrame, val=pd.DataFrame, test=pd.DataFrame):
    encoder = FeatureEncoder()
    train_e = encoder.encode_train(train)
    val_e = encoder.encode_test(val)
    test_e = encoder.encode_test(test)

    with open("encoding_report.json", "w") as f:
        json.dump(encoder.report, f)
    mlflow.log_artifact("encoding_report.json")

    # Save processed final versions
    os.makedirs("artifacts/final_data", exist_ok=True)
    train_e.to_parquet("artifacts/final_data/train_encoded.parquet", index=False)
    val_e.to_parquet("artifacts/final_data/val_encoded.parquet", index=False)
    test_e.to_parquet("artifacts/final_data/test_encoded.parquet", index=False)

    return train_e, val_e, test_e


@enable_mlflow
@step
@monitor(name="baseline_model")
def train_baseline(train: pd.DataFrame, val: pd.DataFrame) -> Output(score=float):
    model = DummyClassifier(strategy="most_frequent")
    TARGET_COL = os.getenv("TARGET_COL", "target")

    model.fit(train.drop(columns=TARGET_COL), train[TARGET_COL])
    preds = model.predict(val.drop(columns=TARGET_COL))
    acc = accuracy_score(val[TARGET_COL], preds)

    mlflow.log_metric("baseline_accuracy", acc)
    mlflow.sklearn.log_model(model, artifact_path="baseline_model")

    return acc


@pipeline(enable_cache=True)
def full_training_pipeline(
    ingest_data,
    inspect_data,
    split_data,
    impute_data,
    scale_transform,
    detect_outliers,
    feature_split,
    feature_construct,
    encode,
    train_baseline,
):
    data = ingest_data()
    checked = inspect_data(data)
    train, val, test = split_data(checked)
    train, val, test = impute_data(train, val, test)
    train, val, test = scale_transform(train, val, test)
    train, val, test = detect_outliers(train, val, test)
    train, val, test = feature_split(train, val, test)
    train, val, test = feature_construct(train, val, test)
    train, val, test = encode(train, val, test)
    train_baseline(train, val)


def TrainingPipeline():
    full_training_pipeline(
        ingest_data=ingest_data(),
        inspect_data=inspect_data(),
        split_data=split_data(),
        impute_data=impute_data(),
        scale_transform=scale_transform(),
        detect_outliers=detect_outliers(),
        feature_split=feature_split(),
        feature_construct=feature_construct(),
        encode=encode(),
        train_baseline=train_baseline(),
    ).run()
