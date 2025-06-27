from pathlib import Path
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extensions import Annotated
from src.Stage_4_Preprocessor.Outlier_Detection import OutlierDetector
from src.Stage_4_Preprocessor.Missing_Imputer import MissingImputer
from src.utils.monitor import monitor
from src.utils.PipelineReporter import PipelineReporter
DATASET_TARGET_COLUMN_NAME = "label"


@step
@monitor(name="missing_imputer_step", track_memory=True, track_input_size=True)
def missing_imputer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
    reporter: PipelineReporter = None
) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"], Annotated[pd.DataFrame, "Val"]]:

    missing_imputer = MissingImputer(
        max_missing_frac_drop=0.8,
        knn_neighbors=2,
        knn_mice_max_rows=10,
        knn_mice_max_columns=3,
        var_ratio_cutoff=0.5,
        cov_change_cutoff=0.2,
        rare_freq_cutoff=0.1,
        random_state=0,
        verbose=True
    )
    train_imp = missing_imputer.fit_transform(train)
    test_imp = missing_imputer.transform(test)
    val_imp = missing_imputer.transform(val)
    reporter.register(name="missing_imputer", component=missing_imputer)

    print("\nImputed Training DataFrame:")
    print("train_imp", train_imp.shape)
    print("test_imp", test_imp.shape)
    print("val_imp", val_imp.shape)
    print("\nDropped columns:", missing_imputer.cols_to_drop)
    print("Numeric columns:", missing_imputer.numeric_cols)
    print("Categorical columns:", missing_imputer.categorical_cols)
    print("Numeric imputers:", missing_imputer.numeric_imputers)
    print("Categorical imputers:", missing_imputer.categorical_imputers)

    return train_imp, test_imp, val_imp


@step
def outlier_imputer(
    train: pd.DataFrame,
    test: pd.DataFrame,
    val: pd.DataFrame,
    reporter: PipelineReporter = None
) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"], Annotated[pd.DataFrame, "Val"]]:

    detector = OutlierDetector(
        outlier_threshold=3,
        robust_covariance=True,
        cap_outliers=None,       # will force winsorize unless cap_outliers=False
        model_family="linear",
        random_state=0,
        verbose=True
    )

    train_clean = detector.fit_transform(train)
    reporter.register(name="Outlier Detector", component=detector)

    print(detector.report["univariate_outliers"])
    # chosen method, etc.
    print(detector.report["multivariate_outliers"])
    print(detector.report["real_outliers"])
    # what was done: drop/winsorize
    print(detector.report["treatment"])
    # see each row’s votes
    print(detector.votes_table_.head())
    # how many clipped per column
    print(detector.clipped_counts_)

    # 4) On new (validation/test) data, simply flag:
    test_clean = detector.transform(test)
    test_outliers = detector.outlier_flags_

    print("Test Outliers:", test_outliers.sum(),
          "out of", len(test_clean).sum())

    val_clean = detector.transform(val)
    val_outliers = detector.outlier_flags_
    print("Test Outliers:", val_outliers.sum(),
          "out of", len(val_clean).sum())
    return train_clean, test_clean, val_clean


@step
def fit_data_preprocessor_step(
    train_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Path, Path]:
    """Fit on training data: detects outliers, imputes, and saves both models."""
    # Step 1: Outlier Detection
    outlier_model = OutlierDetector(verbose=True)
    # this saves internally to outlier_model_state.pkl
    cleaned_df = outlier_model.fit_transform(train_df)

    # Step 2: Missing Imputation
    imputer = MissingImputer()
    # saves to missing_model_state.pkl
    imputed_df = imputer.fit_transform(cleaned_df)

    # Return final df and paths for ZenML tracking
    return imputed_df, Path("outlier_model_state.pkl"), Path("missing_model_state.pkl")


@step
def transform_data_preprocessor_step(
    input_df: pd.DataFrame,
    outlier_model_path: Path,
    missing_model_path: Path
) -> pd.DataFrame:
    """Apply saved outlier + imputer models to validation/test data."""
    # Load OutlierDetector from disk
    outlier_model = OutlierDetector()
    # just use existing loader
    outlier_model._load_state(str(outlier_model_path))
    df_cleaned = outlier_model.transform(input_df)

    # Load MissingImputer
    imputer = MissingImputer()
    imputer._load_state(str(missing_model_path))  # same
    df_final = imputer.transform(df_cleaned)

    return df_final
