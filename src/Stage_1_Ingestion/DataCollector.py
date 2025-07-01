#!/usr/bin/env python3
from __future__ import annotations
import contextlib
import hashlib
import io
import json
import logging
import pathlib as Path
import re
import time
import pandas as pd
from configs import global_conf
with contextlib.suppress(ImportError):
    import boto3
    import requests
    import kafka
    import gspread
    from sqlalchemy import create_engine
    from pymongo import MongoClient
    import paho.mqtt.client as mqtt
    import great_expectations as ge

# ─── Logging & Directories ─────────────────────────────────────────────────────
# LOG_DIR = Path.Path("reports/logs")
REPORT_DIR = Path.Path(global_conf.HEALTH_CHECK_REPORT_PATH)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
# LOG_DIR.mkdir(exist_ok=True, parents=True)

# # Configure logging to both file and console
# logging.basicConfig(
#     level=logging.INFO,
#     handlers=[logging.FileHandler(
#         LOG_DIR / "ingest.log"), logging.StreamHandler()],
#     format="%(asctime)s | %(levelname)s | %(message)s",
# )
log = logging.getLogger("collector")

# ─── PII Regex Patterns ────────────────────────────────────────────────────────
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN = re.compile(r"\b\d{10,}\b")

# ─── Supported File Suffixes ───────────────────────────────────────────────────
SUPPORTED_SUFFIXES = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
}


def _audit_checksum(df: pd.DataFrame, source: str) -> None:
    """
    Compute a short SHA256 checksum of the DataFrame (csv string),
    and log row count + checksum.
    """
    sha256 = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:12]
    log.info(f"{source:15} | rows={len(df):7,} | sha256={sha256}")


def _mask_pii(df: pd.DataFrame) -> pd.DataFrame:
    """
    Redact obvious PII (emails & 10+ digit numbers) by replacing them with [email] and [phone].
    """

    def _scrub(cell):
        if not isinstance(cell, str):
            return cell
        return PHONE_PATTERN.sub("[phone]", EMAIL_PATTERN.sub("[email]", cell))

    return df.applymap(_scrub)


def _semantic_type_profile(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """
    Simple semantic‐type profiler. Detects:
      - constant, ID-like, boolean, datetime, time-only, duration, ZIP code, currency,
      - email, URL, JSON-like, geolocation, categorical, high-cardinality categorical,
      - integer, float, text‐paragraph, or string/text.
    Saves a JSON report to REPORT_DIR.
    Returns a “cleaned” DataFrame (with conversions to bool, datetime, numeric, etc.).
    """
    report: dict[str, dict] = {}
    df_clean = df.copy()

    def is_boolean(series: pd.Series) -> bool:
        vals = set(series.dropna().astype(str).str.lower())
        return vals.issubset({"true", "false", "0", "1", "yes", "no"})

    def is_numeric(series: pd.Series) -> bool:
        """
        Check if a column is numeric or can be reliably coerced to numeric (ignoring nulls).
        """
        if pd.api.types.is_numeric_dtype(series):
            return True
        try:
            coerced = pd.to_numeric(series.dropna(), errors="coerce")
            return coerced.notna().mean() > 0.9
        except:
            return False

    def is_datetime(series: pd.Series) -> bool:
        try:
            parsed = pd.to_datetime(
                series, errors="coerce", infer_datetime_format=True)
            return parsed.notna().mean() > 0.9
        except:
            return False

    def is_time_only(series: pd.Series) -> bool:
        pattern = re.compile(r"^\d{1,2}:\d{2}(:\d{2})?(\s?[APMapm]{2})?$")
        vals = series.dropna().astype(str).str.strip()
        return vals.apply(lambda x: bool(pattern.match(x))).mean() > 0.9

    def is_duration(series: pd.Series) -> bool:
        try:
            parsed = pd.to_timedelta(series, errors="coerce")
            return parsed.notna().mean() > 0.9
        except:
            return False

    def is_zip(series: pd.Series) -> bool:
        vals = series.dropna().astype(str)
        return vals.str.match(r"^\d{5}(-\d{4})?$").mean() > 0.9

    def is_currency(series: pd.Series) -> bool:
        vals = series.dropna().astype(str)
        return vals.str.match(r"^\s*[$₹€£]?[0-9,]+(\.\d{2})?\s*$").mean() > 0.9

    def is_email(series: pd.Series) -> bool:
        vals = series.dropna().astype(str)
        return vals.str.contains(r"^[^@]+@[^@]+\.[^@]+$").mean() > 0.9

    def is_url(series: pd.Series) -> bool:
        vals = series.dropna().astype(str)
        return vals.str.contains(r"^(http://|https://|www\.)").mean() > 0.9

    def is_json_like(series: pd.Series) -> bool:
        vals = series.dropna().astype(str)
        return vals.str.strip().str.startswith("{").mean() > 0.9

    def is_geo(series: pd.Series) -> bool:
        try:
            fv = series.astype(float)
            return ((fv >= -180) & (fv <= 180)).mean() > 0.9
        except:
            return False

    def is_categorical(series: pd.Series) -> bool:
        return (series.nunique(dropna=False) / len(series)) < 0.05

    def is_constant(series: pd.Series) -> bool:
        return series.nunique(dropna=False) == 1

    def is_id_like(series: pd.Series) -> bool:
        return series.is_unique and (series.nunique() == len(series))

    def is_high_card_cat(series: pd.Series) -> bool:
        return (
            (series.dtype == "object")
            and (series.nunique(dropna=False) / len(series) > 0.5)
            and (not series.is_unique)
        )

    for col in df.columns:
        series = df[col]
        orig_dtype = str(series.dtype)
        converted = False
        semantic = "Unknown"

        # Pre-clean copy of column
        col_clean = series.copy()

        if is_constant(series):
            semantic = "Constant / Redundant"

        elif is_id_like(series):
            semantic = "ID-like Field"

        elif is_boolean(series):
            semantic = "Boolean"
            df_clean[col] = (
                series.astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                        "yes": True,
                        "no": False,
                    }
                )
            )
            converted = True

        # --- Handle Numeric BEFORE datetime to avoid false conversions ---
        elif (
            is_numeric(series)
            or series.dropna()
            .map(
                lambda x: isinstance(x, (int, float))
                or str(x).replace(".", "", 1).isdigit()
            )
            .mean()
            > 0.9
        ):
            num_series = pd.to_numeric(series, errors="coerce")

            if num_series.dropna().map(float.is_integer).mean() > 0.9:
                semantic = "Integer"
                df_clean[col] = num_series.astype("Int64")  # nullable integer
            else:
                semantic = "Float"
                df_clean[col] = num_series.astype("float")
            converted = True

        elif is_datetime(series):
            semantic = "Datetime"
            df_clean[col] = pd.to_datetime(series, errors="coerce")
            converted = True

        elif is_time_only(series):
            semantic = "Time Only"

        elif is_duration(series):
            semantic = "Duration / Timedelta"
            df_clean[col] = pd.to_timedelta(series, errors="coerce")
            converted = True

        elif is_zip(series):
            semantic = "ZIP Code"

        elif is_currency(series):
            semantic = "Currency"
            df_clean[col] = series.replace(
                r"[^\d.]", "", regex=True).astype(float)
            converted = True

        elif is_email(series):
            semantic = "Email"

        elif is_url(series):
            semantic = "URL"

        elif is_json_like(series):
            semantic = "JSON / Dict-like"

        elif is_geo(series):
            semantic = "Geolocation"

        elif is_categorical(series):
            semantic = "Categorical"
            df_clean[col] = series.astype("category")
            converted = True

        elif is_high_card_cat(series):
            semantic = "High Cardinality Categorical"

        else:
            # Last fallback
            textmask = series.dropna().map(lambda x: isinstance(x, str) and len(x) > 50)
            if textmask.mean() > 0.9:
                semantic = "Text Paragraph"
            else:
                semantic = "String / Text"

    report[col] = {
        "original_dtype": orig_dtype,
        "semantic_type": semantic,
        "converted": converted,
        "final_dtype": str(df_clean[col].dtype) if converted else orig_dtype,
    }

    # Write JSON report
    outpath = REPORT_DIR / f"dtypes.json"
    with open(outpath, "w") as f:
        json.dump(report, f, indent=2)
    log.info(f"Semantic profiling → {outpath}")

    return df_clean


class DataCollector:
    """
    Orchestrates “Phase 1” data ingestion from various sources, with PII redaction,
    checksum logging, and optional Great Expectations validation.
    """

    def __init__(
        self,
        pii_mask: bool = True,
        validate: bool = True,
        suite_name: str = "default_suite",
    ):
        self.pii_mask = pii_mask
        self.validate = validate
        self.suite_name = suite_name

    def _validate_df(self, df: pd.DataFrame, source: str) -> None:
        """
        If validate=True and Great Expectations is installed, run a suite.
        Otherwise, skip with a warning.
        """
        if not self.validate:
            return
        if "great_expectations" not in globals():
            log.warning(
                "Great Expectations not installed—skipping validation.")
            return
        try:
            ctx = ge.DataContext()
            suite = ctx.get_expectation_suite(self.suite_name)
            validator = ge.from_pandas(df)
            result = validator.validate(expectation_suite=suite)
            if not result.success:
                raise ValueError(f"GE validation failed for '{source}'")
            log.info(
                f"GE validation ✓ ({sum(m.success for m in result.results)}/{len(result.results)})"
            )
        except Exception as e:
            log.warning(f"GE validation exception: {e}")

    def _postprocess(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        1) PII mask (if requested)
        2) GE validation (if installed)
        3) Semantic profiling → JSON report & type conversions
        4) Check for emptiness
        5) Check for duplicates (log but do not drop)
        6) Audit checksum/logging
        """
        if self.pii_mask:
            df = _mask_pii(df)

        self._validate_df(df, source)
        df = _semantic_type_profile(df, source)

        if df is None or df.empty:
            raise ValueError(f"Loaded data from '{source}' is empty.")
        # Log duplicates
        dup_count = df.duplicated().sum()
        if dup_count:
            log.warning(
                f"{source:15} | duplicates={dup_count} rows (logged, not dropped)."
            )
        _audit_checksum(df, source)
        return df

    def read_file(self, path: str, **storage_opts) -> pd.DataFrame:
        """
        Read from local file or S3. Supported suffixes: csv, tsv, parquet, excel.
        """
        if path.startswith("s3://"):
            bucket_key = path.split("s3://", 1)[1]
            bucket, key = bucket_key.split("/", 1)
            obj = boto3.client("s3").get_object(
                Bucket=bucket, Key=key, **storage_opts)
            buffer = io.BytesIO(obj["Body"].read())
            suffix = Path.Path(key).suffix.lower()
        else:
            buffer = path
            suffix = Path.Path(path).suffix.lower()

        if suffix not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file extension: {suffix}")
        ftype = SUPPORTED_SUFFIXES[suffix]

        if ftype == "csv":
            df = pd.read_csv(buffer)
        elif ftype == "tsv":
            df = pd.read_csv(buffer, sep="\t")
        elif ftype == "parquet":
            df = pd.read_parquet(buffer)
        elif ftype == "excel":  # excel
            df = pd.read_excel(buffer)
        else:
            raise ValueError(f"Unsupported file type: {ftype}")
        return self._postprocess(df, f"flat:{Path.Path(path).name}")

    def read_sql(self, dsn: str, query: str) -> pd.DataFrame:
        df = pd.read_sql(query, create_engine(dsn))
        return self._postprocess(df, "sql")

    def read_mongo(
        self, uri: str, db: str, coll: str, query: dict | None = None
    ) -> pd.DataFrame:
        cursor = MongoClient(uri)[db][coll].find(query or {})
        df = pd.DataFrame(list(cursor)).drop(columns="_id", errors="ignore")
        return self._postprocess(df, "mongo")

    def read_rest(self, url: str, *, params=None, headers=None) -> pd.DataFrame:
        payload = requests.get(
            url, params=params, headers=headers, timeout=20).json()
        df = pd.json_normalize(payload)
        return self._postprocess(df, "rest")

    def read_kafka(
        self, topic: str, bootstrap: str, batch: int = 10, group_id: str = "collector"
    ) -> pd.DataFrame:
        rows = []
        consumer = kafka.KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap,
            group_id=group_id,
            auto_offset_reset="latest",
            enable_auto_commit=False,
            value_deserializer=lambda b: json.loads(b.decode()),
        )
        for _ in range(batch):
            rows.append(next(consumer).value)
        df = pd.DataFrame(rows)
        return self._postprocess(df, "kafka")

    def read_gsheet(self, sheet_key: str, creds_json: str) -> pd.DataFrame:
        sheet = (
            gspread.service_account(
                filename=creds_json).open_by_key(sheet_key).sheet1
        )
        df = pd.DataFrame(sheet.get_all_records())
        return self._postprocess(df, "gsheet")

    def read_mqtt(self, broker: str, topic: str, timeout: int = 5) -> pd.DataFrame:
        rows = []

        def on_msg(client, _, msg):
            rows.append(json.loads(msg.payload))

        client = mqtt.Client()
        client.connect(broker)
        client.subscribe(topic)
        client.on_message = on_msg
        client.loop_start()
        time.sleep(timeout)
        client.loop_stop()
        df = pd.DataFrame(rows)
        return self._postprocess(df, "mqtt")


"""
if __name__ == "__main__":
    # === Quick Self-Test ===
    # Create a tiny CSV, read it, and ensure output is reasonable
    import tempfile

    df_test = pd.DataFrame({
        "id": [1, 2, 2],
        "email": ["a@x.com", "b@y.com", None],
        "amount": ["$10.00", "$20.00", "$30.00"],
        "zip": ["10001", "99999", "12345"],
        "flag": ["Yes", "No", "Yes"],
    })
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    df_test.to_csv(tmp.name, index=False)
    collector = DataCollector(pii_mask=True, validate=False)
    df_loaded = collector.read_file(tmp.name)
    print("\nLoaded DataFrame:")
    print(df_loaded.head())
    print("\nSemantic Profiling Report JSON → see 'reports/profiling' directory.")
"""
