from zenml.steps import step, Output
from typing import Any, Dict
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Optional libraries
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None


@step
def tune_model(
    train: pd.DataFrame, val: pd.DataFrame
) -> Output(best_model=Any, best_params=Dict):
    X_train = train.drop(columns=["target"])
    y_train = train["target"]
    X_val = val.drop(columns=["target"])
    y_val = val["target"]

    def objective(trial):
        model_name = trial.suggest_categorical(
            "model", ["rf", "gb", "xgb", "lgb", "cat", "svc", "lr", "knn", "dt"]
        )

        if model_name == "rf":
            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 5, 50),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical(
                    "max_features", ["auto", "sqrt", "log2"]
                ),
                bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
                random_state=42,
            )
        elif model_name == "gb":
            model = GradientBoostingClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical(
                    "max_features", ["auto", "sqrt", "log2"]
                ),
                random_state=42,
            )
        elif model_name == "xgb" and XGBClassifier:
            model = XGBClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                gamma=trial.suggest_float("gamma", 0, 5),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
                random_state=42,
            )
        elif model_name == "lgb" and LGBMClassifier:
            model = LGBMClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                n_estimators=trial.suggest_int("n_estimators", 100, 500),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                num_leaves=trial.suggest_int("num_leaves", 20, 100),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 0.0, 5.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
                random_state=42,
            )
        elif model_name == "cat" and CatBoostClassifier:
            model = CatBoostClassifier(
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
                depth=trial.suggest_int("depth", 3, 10),
                iterations=trial.suggest_int("iterations", 100, 500),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                random_state=42,
                verbose=0,
            )
        elif model_name == "svc":
            model = SVC(
                C=trial.suggest_float("C", 0.01, 10),
                kernel=trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
                gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
                degree=trial.suggest_int("degree", 2, 5),
                probability=True,
            )
        elif model_name == "lr":
            model = LogisticRegression(
                penalty=trial.suggest_categorical("penalty", ["l2"]),
                C=trial.suggest_float("C", 0.01, 10),
                solver="lbfgs",
                max_iter=1000,
            )
        elif model_name == "knn":
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", 3, 30),
                weights=trial.suggest_categorical("weights", ["uniform", "distance"]),
                algorithm=trial.suggest_categorical(
                    "algorithm", ["auto", "ball_tree", "kd_tree"]
                ),
                p=trial.suggest_int("p", 1, 2),
            )
        elif model_name == "dt":
            model = DecisionTreeClassifier(
                max_depth=trial.suggest_int("max_depth", 3, 30),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
                max_features=trial.suggest_categorical(
                    "max_features", ["auto", "sqrt", "log2"]
                ),
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        model.fit(X_train, y_train)
        return model.score(X_val, y_val)

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=80)

    best_trial = study.best_trial
    model_type = best_trial.params.pop("model")

    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    model_cls_map = {
        "rf": RandomForestClassifier,
        "gb": GradientBoostingClassifier,
        "xgb": XGBClassifier,
        "lgb": LGBMClassifier,
        "cat": CatBoostClassifier,
        "svc": SVC,
        "lr": LogisticRegression,
        "knn": KNeighborsClassifier,
        "dt": DecisionTreeClassifier,
    }

    ModelClass = model_cls_map.get(model_type)
    if ModelClass is None:
        raise RuntimeError(f"No class found for: {model_type}")

    final_model = ModelClass(**best_trial.params)
    final_model.fit(X_all, y_all)

    return final_model, best_trial.params
