import pandas as pd
from zenml import step
from src.Stage_2_EPD_Analysis.EDAnalyzer import EDAnalyzer
from src.Stage_2_EPD_Analysis.EPDA import UnifiedEPDA
from src.Stage_2_EPD_Analysis.PDAnalysis import ProbabilisticAnalysis


@step
def EDAnalyze(df: pd.DataFrame, project: str = "Default") -> pd.DataFrame:
    """
    Placeholder for EPD Analysis step.
    Currently just returns the DataFrame unchanged.
    """
    print(f"EPD Analysis completed for project '{project}'.")
    EDAnalyse = EDAnalyzer(df)
    asasa = EDAnalyse.run()
    return df


@step
def PDAnalyze(df: pd.DataFrame, project: str = "Default") -> pd.DataFrame:
    """
    Placeholder for PD Analysis step.
    Currently just returns the DataFrame unchanged.
    """
    print(f"PD Analysis completed for project '{project}'.")
    EDAnalyse = ProbabilisticAnalysis(df)
    asasa = EDAnalyse.run()
    return df


@step
def UnifiedPEDAnalyze(df: pd.DataFrame, project: str = "Default") -> pd.DataFrame:
    """
    Placeholder for EPD Analysis step.
    Currently just returns the DataFrame unchanged.
    """
    print(f"EPD Analysis completed for project '{project}'.")
    PEDAnalyse = UnifiedEPDA(df)
    asasa = PEDAnalyse.run()
    return df
