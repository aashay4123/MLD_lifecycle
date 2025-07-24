#!/usr/bin/env python3

import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

from src.Stage_4_Preprocessor.Feature_Transformer import FeatureScalerTransformer


def load_and_prepare_datasets() -> pd.DataFrame:
    """Load multiple datasets, combine them, add synthetic issues."""
    iris = load_iris()
    wine = load_wine()
    cancer = load_breast_cancer()

    df_iris = pd.DataFrame(iris.data, columns=[f"iris_{f}" for f in iris.feature_names])
    df_wine = pd.DataFrame(wine.data, columns=[f"wine_{f}" for f in wine.feature_names])
    df_cancer = pd.DataFrame(
        cancer.data, columns=[f"cancer_{f}" for f in cancer.feature_names]
    )

    # Add constant column
    df_iris["constant_col"] = 5

    # Add categorical column
    df_iris["species"] = pd.Series(iris.target).map(
        {0: "setosa", 1: "versicolor", 2: "virginica"}
    )

    # Inject missing values into wine
    df_wine.iloc[::10, 0] = np.nan

    # Concatenate them
    df_all = pd.concat([df_iris, df_wine, df_cancer], ignore_index=True)
    return df_all


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Impute, remove outliers, drop unusable columns before scaling."""

    # 1️⃣ Drop constant columns (if all non-null values are identical)
    nunique = df.nunique(dropna=True)
    constant_cols = nunique[nunique <= 1].index
    if len(constant_cols) > 0:
        print(f"[Preprocessing] Dropping constant columns: {list(constant_cols)}")
        df = df.drop(columns=constant_cols)

    # 2️⃣ Drop non-numeric columns (categorical/string/object)
    df = df.select_dtypes(include=[np.number])

    # 3️⃣ Impute missing values with mean (numeric columns only)
    if df.isnull().any().any():
        print("[Preprocessing] Filling missing values with mean imputation.")
        imputer = SimpleImputer(strategy="mean")
        df[:] = imputer.fit_transform(df)

    # 4️⃣ Detect outliers with IsolationForest and remove them
    print("[Preprocessing] Detecting and removing outliers with IsolationForest.")
    iso = IsolationForest(contamination=0.05, random_state=42)
    preds = iso.fit_predict(df)
    outliers = preds == -1
    print(f"[Preprocessing] Dropping {outliers.sum()} rows identified as outliers.")
    df_clean = df.loc[~outliers].reset_index(drop=True)

    return df_clean


def sanity_check_transformation(
    X_transformed: pd.DataFrame, fst: FeatureScalerTransformer
):
    print("\n=== Sample of transformed data ===")
    print(X_transformed.head())

    assert os.path.isfile(
        fst.report_file
    ), f"❌ Report file {fst.report_file} not created!"
    visuals = os.listdir("scaler_visuals")
    assert len(visuals) > 0, "❌ No before/after visualizations were saved!"

    report_df = pd.read_csv(fst.report_file)
    print("\n=== Sample of generated report ===")
    print(report_df.head())

    print("\n✅ Full preprocessing + scaling pipeline ran successfully.")


if __name__ == "__main__":
    X = load_and_prepare_datasets()
    X_preprocessed = preprocess_data(X)
    fst = FeatureScalerTransformer()

    try:
        X_transformed = fst.fit_transform(X_preprocessed)
        fst.generate_html_report("my_scaler_reports.html")
        sanity_check_transformation(X_transformed, fst)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
