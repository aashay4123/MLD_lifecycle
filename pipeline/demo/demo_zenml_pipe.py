from typing import Annotated, Tuple

import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from zenml import log_metadata, pipeline, step
from zenml.integrations.sklearn.steps import sklearn_evaluator, sklearn_trainer


@step
def load_data() -> (
    Tuple[
        Annotated[np.ndarray, "training_data"], Annotated[np.ndarray, "training_labels"]
    ]
):
    data = np.random.rand(100, 2)
    labels = np.random.rand(100)
    return data, labels


@step
def train_model(
    data: np.ndarray,
    labels: np.ndarray,
) -> Annotated[LinearRegression, "trained_model"]:
    model = LinearRegression().fit(data, labels)
    print(f"Model coefficients: {model.coef_}, intercept: {model.intercept_}")
    log_metadata(
        metadata={
            "coefficients": model.coef_.tolist(),
            "intercept": float(model.intercept_),
        }
    )
    return model


@pipeline
def basic_pipeline():
    train_model(*load_data())


# if __name__ == "__main__":
#     basic_pipeline()


@step
def data_loader():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


@step
def get_model():
    # Here you define which sklearn model and params to use
    model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=42)
    return model


@pipeline
def sklearn_pipeline(data_loader, get_model, trainer, evaluator):
    X_train, X_test, y_train, y_test = data_loader()
    model = get_model()
    # trainer trains the model
    trained_model = trainer(X_train, y_train, model=model)
    evaluator(model=trained_model, X=X_test, y=y_test)  # evaluator evaluates it


if __name__ == "__main__":
    pipeline_instance = sklearn_pipeline(
        data_loader=data_loader(),
        get_model=get_model(),
        trainer=sklearn_trainer(),
        evaluator=sklearn_evaluator(),
    )
    pipeline_instance.run()
