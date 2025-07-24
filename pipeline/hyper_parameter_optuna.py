import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from configs import global_conf
from src.Stage_8_HP_tuning.Optuna_tuning import OptunaHyperTuner
from src.Stage_8_HP_tuning.optuna_tuning_ensemble import OptunaEnsembler
from src.utils.utils import compute_metric, get_metric

warnings.filterwarnings("ignore")

# --- Define datasets ---
datasets = {
    # "binary_classification": {
    #     "X": make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)[0],
    #     "y": make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)[1],
    #     "task": "classification"
    # },
    "synthetic_regression": {
        "X": make_regression(n_samples=500, n_features=10, noise=15, random_state=42)[
            0
        ],
        "y": make_regression(n_samples=500, n_features=10, noise=15, random_state=42)[
            1
        ],
        "task": "regression",
    },
}

results = []

for dataset_name, dataset in datasets.items():
    X, y, task = dataset["X"], dataset["y"], dataset["task"]

    print(f"\n🔵 Dataset: {dataset_name} ({task})")

    stratify_main = y if (task == "classification" and len(np.unique(y)) < 20) else None
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        stratify=stratify_main,
        test_size=0.2,
        random_state=global_conf.RANDOM_STATE,
    )

    stratify_val = (
        y_train_val
        if (task == "classification" and len(np.unique(y_train_val)) < 20)
        else None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        stratify=stratify_val,
        test_size=0.2,
        random_state=global_conf.RANDOM_STATE,
    )

    tuner = OptunaHyperTuner(
        X_train,
        y_train,
        X_test,
        y_test,
        n_trials=5,
        cv_folds=global_conf.DEFAULT_CV_FOLDS,
        scoring=global_conf.DEFAULT_METRIC_REGRESSION,
        sampler=global_conf.DEFAULT_SAMPLER,
        pruner=global_conf.choose_pruner(X_train),
    )
    tuner.tune()
    best_models = tuner.get_best_models()

    metric_str = get_metric(task, global_conf.DEFAULT_SCORING)

    for model_name, (retrained_model, study) in best_models.items():
        best_params = study.best_params

        try:
            # retrained_model = retrained_model.__class__(**best_params)
            # retrained_model.fit(X_train, y_train)

            train_score = compute_metric(
                y_train, retrained_model, X_train, metric_str, task=task
            )
            test_score = compute_metric(
                y_test, retrained_model, X_test, metric_str, task=task
            )

            results.append(
                {
                    "dataset": dataset_name,
                    "task": task,
                    "model": model_name,
                    "best_cv_score": study.best_value,
                    "train_score": train_score,
                    "test_score": test_score,
                    "best_params": best_params,
                }
            )

            print(
                f"✅ {model_name}: train {metric_str}={train_score:.4f}, test {metric_str}={test_score:.4f}"
            )

        except Exception as e:
            print(f"❌ Failed for {model_name} on {dataset_name}: {e}")
            continue

    # 🔥 Build and evaluate ensemble after individual models
    ensembler = OptunaEnsembler(
        best_models=best_models,
        X=X_train,
        y=y_train,
        task=task,
        scoring=global_conf.DEFAULT_METRIC_REGRESSION,
        cv_folds=global_conf.DEFAULT_CV_FOLDS,
        top_n=3,  # or set to any number of top models you prefer
        optimize_weights=True,
        n_trials=10,
    )
    ensembler.build_ensemble()
    ensemble = ensembler.get_ensemble()
    ensemble_info = ensembler.get_ensemble_info()
    ensemble_test_score = None
    try:

        # Evaluate ensemble on test set
        ensemble_test_score = compute_metric(
            y_test, ensemble, X_test, metric_str, task=task
        )
        print(
            f"✅ Final ensemble: test {metric_str}={ensemble_test_score:.4f} | fallback_used={ensemble_info['fallback_used']}"
        )
    except Exception as e:
        print(f"❌ Failed to compute metric for ensemble: {e}")
    # Record ensemble results
    results.append(
        {
            "dataset": dataset_name,
            "task": task,
            "model": "OptimizedEnsemble",
            "best_cv_score": ensemble_info["ensemble_score"],
            "train_score": None,
            "test_score": ensemble_test_score,
            "best_params": ensemble_info["ensemble_details"],
        }
    )

# Save all results
df = pd.DataFrame(results)
df.to_csv("optuna_best_models_validation.csv", index=False)
print("\n✅ Results saved to optuna_best_models_validation.csv")
