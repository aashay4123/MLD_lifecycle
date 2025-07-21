````markdown
## 1 — Phase 1 · Data Ingestion & Profiling <a name="1-phase-1--data-ingestion"></a>

> **Goal** — ingest tabular & semi-structured data from any source, redact obvious PII, validate schemas, profile semantic types, and audit-log snapshots. Everything is orchestrated by **[`DataCollector`](src/data_collector.py)**.

### 1·0 What happens under the hood 🛠

1. **PII Redaction** via regex for emails & 10+-digit numbers
2. **Great Expectations** suite (schema/range/null checks) → optional
3. **Semantic Profiling** → JSON report & dtype conversions
4. **Emptiness & Duplicate Checks** (log warnings, do _not_ drop duplicates)
5. **Audit Checksum & Row Count** logged to `logs/ingest.log`

---

### 1A Flat-Files & Object Storage <a name="1a-flat-files--object-storage"></a>

| Format / Suffix                | Call                                         | Notes                               |
| ------------------------------ | -------------------------------------------- | ----------------------------------- |
| **CSV** `.csv`                 | `collector.read_file("data/users.csv")`      | Pandas infer dtypes; auto-delimiter |
| **TSV** `.tsv`                 | `collector.read_file("data/users.tsv")`      | Tab-separated                       |
| **Parquet** `.parquet` / `.pq` | `collector.read_file("data/events.parquet")` | Requires `pyarrow`                  |
| **Excel** `.xlsx` / `.xls`     | same as above                                | Multi-sheet → reads first sheet     |
| **S3** `s3://…`                | `collector.read_file("s3://bucket/key")`     | IAM/KMS via `storage_opts`          |

---

### 1B Relational DBs <a name="1b-relational-databases"></a>

```python
df = collector.read_sql(
    dsn="postgresql+psycopg2://user:pwd@host/db",
    query="SELECT * FROM users WHERE updated_at >= '2025-01-01'"
)
```
````

---

### 1C NoSQL / MongoDB <a name="1c-nosql--mongo"></a>

```python
df = collector.read_mongo(
    uri="mongodb://user:pwd@host:27017",
    db="crm", coll="events", query={"active": True}
)
```

---

### 1D REST APIs <a name="1d-rest-apis"></a>

```python
df = collector.read_rest(
    url="https://api.example.com/data",
    params={"limit": 100}, headers={"Authorization": "Bearer …"}
)
```

---

### 1E Kafka Streams <a name="1e-kafka-streams"></a>

```python
df = collector.read_kafka(
    topic="tx-events", bootstrap="broker:9092", batch=500
)
```

---

### 1F Google Sheets <a name="1f-google-sheets"></a>

```python
df = collector.read_gsheet(
    sheet_key="your-sheet-id", creds_json="gcp-creds.json"
)
```

---

### 1G MQTT Streams <a name="1g-mqtt-streams"></a>

```python
df = collector.read_mqtt(
    broker="mqtt.example.com", topic="sensors/+", timeout=10
)
```

---

### 🔧 Quick-Start

```bash
# install dependencies
pip install pandas boto3 requests kafka-python gspread pymongo paho-mqtt sqlalchemy great_expectations
```

```python
from src.data_collector import DataCollector

collector = DataCollector(pii_mask=True, validate=True, suite_name="default_suite")
df = collector.read_file("data/sample.csv")
print(df.head())
```

---

### 📌 Why the extra validation & profiling?

1. **Fail-Fast** – catch bad schemas & empty feeds early
2. **Data Confidence** – semantic typing + GE checks build trust
3. **Audit & Governance** – PII redaction & checksum log → compliance

---

> **Next phase ➜ [Phase 1.2 · Data Inspection](#1.2-phase-1.2--data-inspection)**

```

```
