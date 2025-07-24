import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class ModelEvaluation:
    """
    Section 7: Model Evaluation and Deployment.
    Evaluates the trained model on the test set and handles model saving for deployment.
    """

    def __init__(self):
        """Initialize ModelEvaluation."""
        self.results = {}

    def evaluate(self, model, X_test, y_test, baseline_model=None):
        """
        Evaluate the model on the test set, optionally comparing with a baseline model.
        :param model: Trained model to evaluate.
        :param X_test: Test features.
        :param y_test: Test target.
        :param baseline_model: A baseline model for comparison (optional).
        :return: Dictionary of evaluation metrics (for model and baseline if provided).
        """
        results = {}
        # Evaluate main model
        y_pred = model.predict(X_test)
        if (
            y_test.dtype == object
            or str(y_test.dtype).startswith("category")
            or (y_test.dtype != object and y_test.nunique() < 20)
        ):
            # Classification metrics
            acc = accuracy_score(y_test, y_pred)
            average = "binary" if len(set(y_test)) == 2 else "macro"
            f1 = f1_score(y_test, y_pred, average=average)
            results["model_accuracy"] = acc
            results["model_f1"] = f1
        else:
            # Regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results["model_mse"] = mse
            results["model_mae"] = mae
            results["model_r2"] = r2
        # Evaluate baseline model if given
        if baseline_model is not None:
            y_base_pred = baseline_model.predict(X_test)
            if "model_accuracy" in results:
                # classification
                base_acc = accuracy_score(y_test, y_base_pred)
                base_f1 = f1_score(
                    y_test,
                    y_base_pred,
                    average=("binary" if len(set(y_test)) == 2 else "macro"),
                )
                results["baseline_accuracy"] = base_acc
                results["baseline_f1"] = base_f1
                results["accuracy_improvement"] = results["model_accuracy"] - base_acc
                results["f1_improvement"] = results["model_f1"] - base_f1
            else:
                # regression
                base_mse = mean_squared_error(y_test, y_base_pred)
                base_mae = mean_absolute_error(y_test, y_base_pred)
                base_r2 = r2_score(y_test, y_base_pred)
                results["baseline_mse"] = base_mse
                results["baseline_mae"] = base_mae
                results["baseline_r2"] = base_r2
                results["mse_reduction"] = base_mse - results["model_mse"]
                results["mae_reduction"] = base_mae - results["model_mae"]
                results["r2_improvement"] = results["model_r2"] - base_r2
        self.results = results
        return results

    def save_model(self, model, file_path: str):
        """
        Save the trained model to disk for deployment.
        :param model: The model object to save (should be pickle-able).
        :param file_path: File path to save the model (e.g., 'model.pkl').
        """
        joblib.dump(model, file_path)

    @staticmethod
    def load_model(file_path: str):
        """
        Load a model from disk.
        :param file_path: Path to the model file.
        :return: Loaded model object.
        """
        return joblib.load(file_path)
