# core modules
import logging
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future

# libraries
import pandas as pd
import mlflow
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import (
    BinaryEncoder, HashingEncoder, HelmertEncoder, JamesSteinEncoder,
    LeaveOneOutEncoder, TargetEncoder
)

# zenml integrations
from zenml import step
from src.utils.perfkit import perfclass, PerfMixin
from src.utils.monitor import monitor

log = logging.getLogger(__name__)
DEFAULT_REPORT_DIR = Path("reports/auto_categorical")
DEFAULT_REPORT_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_EXPERIMENT = "AutoCategoricalEncoding"


@dataclass
class EncodingConfig:
    onehot_frac: float = 0.05
    ordinal_frac: float = 0.20
    freq_frac: float = 0.50
    target_smoothing: float = 0.3
    n_jobs: int = -1
    output_format: str = "parquet"
    output_dir: Path = DEFAULT_REPORT_DIR
    forced_strategies: Dict[str, str] = field(default_factory=dict)


@perfclass
class AutoCategoricalEncoder(PerfMixin):
    def __init__(self, cfg: EncodingConfig):
        self.cfg = cfg
        self.rules = {}
        self.unique_frac = {}
        self.report = {}
        self._save_futures = []
        self.y_train = None
        self.encoders = {}
        self.output_dir = cfg.output_dir
        self.output_format = cfg.output_format
        self.forced_strategies = cfg.forced_strategies
        self._save_executor = ThreadPoolExecutor(max_workers=1)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.joinpath("encoders").mkdir(exist_ok=True)
        if not (0 <= cfg.onehot_frac <= cfg.ordinal_frac <= cfg.freq_frac <= 1):
            raise ValueError("Encoding thresholds invalid")

    def _save_pickle(self, obj, name: str):
        with open(self.output_dir / f"encoders/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)

    def _load_pickle(self, name: str):
        with open(self.output_dir / f"encoders/{name}.pkl", "rb") as f:
            return pickle.load(f)

    def fit_transform(self, df: pd.DataFrame, y: Optional[pd.Series] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        self.y_train = y
        df0 = df.reset_index(drop=True)
        self.categorical_cols = df0.select_dtypes(
            include=["object", "category"]).columns.tolist()
        n = len(df0)
        self.unique_frac = {c: df0[c].nunique(
            dropna=False) / n for c in self.categorical_cols}
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            mlflow.log_params({
                "onehot_frac": self.cfg.onehot_frac,
                "ordinal_frac": self.cfg.ordinal_frac,
                "freq_frac": self.cfg.freq_frac,
                "n_jobs": self.cfg.n_jobs,
                "n_categoricals": len(self.categorical_cols)
            })

            results = {}
            for variant in ("linear", "tree", "knn"):
                df_variant = self._build_variant(df0, variant)
                results[variant] = df_variant
                path = self._save(df_variant, variant)
                self.report.setdefault(variant, {})
                self.report[variant]["path"] = str(
                    path.relative_to(self.output_dir))
                self.report[variant]["rules"] = self.rules[variant]
                mlflow.log_metric(f"{variant}_rows", df_variant.shape[0])
                mlflow.log_metric(f"{variant}_cols", df_variant.shape[1])

            self._write_reports(results)
            self._save_executor.shutdown(wait=True)
        return results["linear"], results["tree"], results["knn"]

    def _build_variant(self, df: pd.DataFrame, variant: str) -> pd.DataFrame:
        self.rules[variant] = self._decide_rules(df, variant)
        base = df.drop(columns=self.categorical_cols, errors="ignore")
        parts = [base]

        def encode_group(rule: str, cols: List[str]) -> pd.DataFrame:
            enc_func = {
                "onehot": self._encode_onehot,
                "ordinal": self._encode_ordinal,
                "frequency": self._encode_frequency,
                "binary": self._encode_binary,
                "helmert": self._encode_helmert,
                "jstein": self._encode_jstein,
                "hashing": self._encode_hashing,
                "target": self._encode_target,
                "loo": self._encode_loo
            }.get(rule)
            return enc_func(df, cols) if enc_func else pd.DataFrame(index=df.index)

        encoded_parts = Parallel(n_jobs=self.cfg.n_jobs)(
            delayed(encode_group)(rule, self.rules[variant].get(rule, []))
            for rule in self.rules[variant]
        )
        parts.extend(encoded_parts)
        return pd.concat(parts, axis=1)

    def _decide_rules(self, df: pd.DataFrame, variant: str) -> Dict[str, List[str]]:
        rules = {k: [] for k in [
            "onehot", "ordinal", "frequency", "target", "loo", "binary", "hashing", "helmert", "backdiff"
        ]}
        for col in self.categorical_cols:
            if col in self.cfg.forced_strategies:
                bucket = self.cfg.forced_strategies[col].lower()
                if bucket in rules:
                    rules[bucket].append(col)
                continue

            f = self.unique_frac[col]
            n_unique = df[col].nunique()

            if variant == "linear":
                if f <= self.cfg.onehot_frac:
                    rules["onehot"].append(col)
                elif f <= self.cfg.freq_frac:
                    rules["frequency"].append(col)
                elif f <= 0.6:
                    rules["binary"].append(col)
                elif self.y_train is not None and f <= 0.95:
                    rules["target"].append(col)
                elif f > 0.95 and n_unique > 1000:
                    rules["hashing"].append(col)
                else:
                    rules["helmert"].append(col)

            elif variant == "tree":
                if f <= self.cfg.onehot_frac:
                    rules["onehot"].append(col)
                elif f <= self.cfg.ordinal_frac:
                    rules["ordinal"].append(col)
                elif f <= self.cfg.freq_frac:
                    rules["frequency"].append(col)
                elif f <= 0.95:
                    rules["loo"].append(col)
                else:
                    rules["hashing"].append(col)
                if "order" in col.lower():
                    rules["backdiff"].append(col)

            elif variant == "knn":
                if f > 0.5:
                    rules["hashing"].append(col)
                else:
                    rules["frequency"].append(col)
        return {k: v for k, v in rules.items() if v}

    def _write_reports(self, results: Dict[str, pd.DataFrame]):
        for variant, df_variant in results.items():
            n_rows = len(df_variant)
            mem_bytes = df_variant.memory_usage(deep=True).sum()
            card = {col: int(self.unique_frac.get(col, 0) * n_rows)
                    for col in self.categorical_cols}
            self.report[variant].update({
                "memory_bytes": int(mem_bytes),
                "n_rows": n_rows,
                "cardinality": card
            })

        with open(self.output_dir / "encoding_report.json", "w") as f:
            json.dump(self.report, f, indent=2)
        mlflow.log_artifact(str(self.output_dir / "encoding_report.json"))

    def transform(self, df: pd.DataFrame, variant: str = "linear") -> pd.DataFrame:
        path = self.output_dir / f"processed_{variant}.{self.output_format}"
        template = pd.read_parquet(
            path) if self.output_format == "parquet" else pd.read_csv(path)
        df = df.drop(columns=self.categorical_cols,
                     errors="ignore").reset_index(drop=True)
        for col in template.columns:
            if col not in df.columns:
                df[col] = 0
        missing = set(df.columns) - set(template.columns)
        if missing:
            log.warning(f"Extra columns dropped: {missing}")
        return df[template.columns]

    # Encoding methods

    def _encode_onehot(self, df, cols):
        if not cols:
            return pd.DataFrame(index=df.index)
        enc = OneHotEncoder(sparse=False, handle_unknown="ignore")
        arr = enc.fit_transform(df[cols])
        self._save_pickle(enc, "onehot")
        return pd.DataFrame(arr, columns=enc.get_feature_names_out(cols), index=df.index)

    def _encode_ordinal(self, df, cols):
        if not cols:
            return pd.DataFrame(index=df.index)
        enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1)
        arr = enc.fit_transform(df[cols])
        self._save_pickle(enc, "ordinal")
        return pd.DataFrame(arr, columns=cols, index=df.index)

    def _encode_frequency(self, df, cols):
        freq_df = pd.DataFrame(index=df.index)
        for c in cols:
            vc = df[c].value_counts(normalize=True)
            freq_df[f"{c}_freq"] = df[c].map(vc).fillna(0.0)
        return freq_df

    def _encode_target(self, df, cols):
        if not cols or self.y_train is None:
            return pd.DataFrame(index=df.index)
        enc = TargetEncoder(cols=cols, smoothing=self.cfg.target_smoothing)
        enc.fit(df[cols], self.y_train)
        self._save_pickle(enc, "target")
        return enc.transform(df[cols])

    def _encode_binary(self, df, cols):
        enc = BinaryEncoder(cols=cols)
        df_enc = enc.fit_transform(df[cols])
        self._save_pickle(enc, "binary")
        return df_enc

    def _encode_helmert(self, df, cols):
        enc = HelmertEncoder(cols=cols)
        df_enc = enc.fit_transform(df[cols])
        self._save_pickle(enc, "helmert")
        return df_enc

    def _encode_jstein(self, df, cols):
        enc = JamesSteinEncoder(cols=cols)
        df_enc = enc.fit_transform(df[cols], self.y_train)
        self._save_pickle(enc, "jstein")
        return df_enc

    def _encode_loo(self, df, cols):
        enc = LeaveOneOutEncoder(cols=cols)
        enc.fit(df[cols], self.y_train)
        self._save_pickle(enc, "loo")
        return enc.transform(df[cols])

    def _encode_hashing(self, df, cols):
        enc = HashingEncoder(cols=cols, n_components=8)
        df_enc = enc.fit_transform(df[cols])
        self._save_pickle(enc, "hashing")
        return df_enc

    def _save(self, df: pd.DataFrame, variant: str) -> Path:
        path = self.output_dir / f"processed_{variant}.{self.output_format}"

        def writer():
            try:
                if self.output_format == "parquet":
                    df.to_parquet(path, index=False)
                else:
                    df.to_csv(path, index=False)
            except Exception as e:
                log.error(f"Saving failed: {e}")

        future = self._save_executor.submit(writer)
        self._save_futures.append(future)
        return path
