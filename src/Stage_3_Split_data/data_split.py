from typing_extensions import Annotated
from zenml import step
from typing import Tuple
import pandas as pd
from .BaselineModel import AutoBaseline
from .ThreeWaySplit import SplitThreeWay

DATASET_TARGET_COLUMN_NAME = "label"


@step
def baseline(train, test):
    baselineModel = AutoBaseline(target=DATASET_TARGET_COLUMN_NAME)
    baselineModel.run(train, test)
    # reporter.register("auto_baseline", baseline)


@step
def data_splitter(
    data: pd.DataFrame,
    target: str = DATASET_TARGET_COLUMN_NAME,
    stratify: bool = True,
    oversample: bool = False,
    seed: int = 42,
) -> Tuple[Annotated[pd.DataFrame, "train"], Annotated[pd.DataFrame, "test"], Annotated[pd.DataFrame, "Val"]]:
    if not DATASET_TARGET_COLUMN_NAME:
        raise ValueError(
            "DATASET_TARGET_COLUMN_NAME must be set to generate a baseline model.")
    splitData = SplitThreeWay(
        data=data,
        stratify=stratify,
        seed=seed,
        oversample=oversample,
        target=target)
    train, test, val = splitData.split_data()
    return train, test, val
