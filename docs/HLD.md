# High‑Level Design: Local Docker + ZenML + DVC + MLflow Pipeline

---

## 1. Design Principles

1. **Isolation & Reproducibility**

   - **DockerSequentialRunner**: each ZenML step in its own container.
   - **DVC**: `dvc pull` ensures raw data consistency.

2. **Continuous Integration**

   - **Makefile** orchestrates environment setup, linting, testing, and deployment.
   - **GitHub Actions** will invoke these targets on every PR/push.

3. **Code Quality**

   - **Black** for formatting, **Flake8** for linting, **Mypy** (via `make check`) for typing.

4. **Experiment & Model Management**

   - **MLflow Escalade** for tracking and model registry.
   - **Model serving** via MLflow REST API, consumed by FastAPI.

5. **Local‑First, Production‑Parity**

   - Host‑mounted volumes for `./data` and `./models`.
   - Mirror production Docker stack in a single `docker-compose.yml`.

---

## 2. Core Layers & Technologies

| Layer                    | Technology / File                       | Role                                                                       |
| ------------------------ | --------------------------------------- | -------------------------------------------------------------------------- |
| **Environment Setup**    | `make env`                              | Poetry → Python 3.11 environment                                           |
| **Dependencies**         | `make install`                          | Poetry install + ZenML integrations                                        |
| **Dev Stack**            | `make devup_bg` / `make devdown`        | ZenML local stack login; MLflow server startup/teardown                    |
| **Data Versioning**      | DVC (`make train` runs `dvc pull`)      | Ensures `./data/raw` is in sync                                            |
| **Orchestration**        | ZenML DockerSequentialRunner            | Defined in `pipeline/train.py`                                             |
| **Configuration**        | `global_conf.py`                        | Central path & parameter definitions                                       |
| **Training**             | `make train`                            | Calls `pipeline/train.py` → ingestion, prep, split, HPO, train, MLflow log |
| **Testing**              | `make test`                             | Calls `pipeline/test.py` (unit/integration tests)                          |
| **Formatting & Linting** | `make fmt` / `make lint` / `make check` | Black, Flake8, Mypy                                                        |
| **Serving**              | FastAPI + Nginx (Docker)                | `/predict` & `/health` endpoints                                           |
| **CI/CD**                | GitHub Actions → Makefile targets       | Automate all above on push                                                 |
| **Alerts**               | Discord webhook                         | Triggered by ZenML `on_failure` hooks and CI failures                      |

---

## 3. Architecture Diagrams

### 3.1 Training Pipeline

```mermaid
flowchart TD
  subgraph ZenML
    Z[DockerSequentialRunner]
  end

  Z -->|1. DVC Pull| DVC[DVC Pull → ./data/raw]
  Z -->|2. Ingest| Ingest[ingest_step]
  Z -->|3. Prep| Prep[prep_step → ./data/processed]
  Z -->|4. Split| Split[split_step → ./data/split]
  Z -->|5. HPO| HPO[hpo_step (Optuna)]
  Z -->|6. Train| Train[train_step → artifacts]
  Z -->|7. Log| Log[mlflow_step → remote Escalade]
  Log -->|8. Register| Reg[MLflow Model Registry]
  Reg -->|9. Export| ServeModel[./models/served]
```

- **Step names** correspond to functions in `pipeline/train.py`.
- **Makefile**: `make train` wraps ZML orchestration.

---

### 3.2 Inference Pipeline

```mermaid
flowchart LR
  U[Client (Web/App)] --> Nginx[Nginx]
  Nginx --> API[FastAPI (Docker)]
  API --> Pre[preprocess_input()]
  Pre --> MLflowServe[MLflow REST Serve]
  MLflowServe --> Predict[predict()]
  Predict --> Post[postprocess_output()]
  Post --> API --> U
  API --> Health[/health_check()/]
```

- **FastAPI** calls MLflow’s serve endpoint rather than loading files directly.
- **Makefile**: `docker-compose up -d` (invoked in CI/CD) brings up Nginx + FastAPI.

---

### 3.3 CI/CD & Quality Gates

```mermaid
flowchart TB
  Repo[GitHub] --> Actions[GitHub Actions]
  Actions -->|runs| EnvInstall[make env & install]
  EnvInstall --> Lint[make fmt; make lint]
  Lint --> TypeCheck[make check]
  TypeCheck --> Test[make test]
  Test --> Build[Build Docker Images]
  Build --> Deploy[Docker Compose Up]
  Deploy --> Smoke[/health endpoint test/]
  Smoke --fail--> Discord[Discord Webhook]
  ZMLStepError[ZenML Step Error] --on_failure hook--> Discord
```

- Every PR triggers the full Makefile pipeline.
- Failures at any stage send a message to your Discord channel.

---

## 4. `Makefile` & `global_conf.py` Deep‑Dive

- **Makefile `devup` targets** spin up your **MLflow** (SQLite backend) and **ZenML local stack**.
- **`train`** target calls your training script, which:

  - Reads paths from **`global_conf.py`** (e.g. `CSV_PATH`, `RAW_PARQUET_PATH`).
  - Executes ZenML steps in Docker containers.
  - Uses Optuna per `OPTUNA_REPORT_PATH`.
  - Persists model artifacts under `MODEL_ARTIFACTS_PATH` and `FINAL_MODEL_PATH`.

- **Lint & Format**:

  - `make fmt` → Black
  - `make lint` → Flake8 (`src/`)
  - `make check` → Mypy on `src/` & `pipeline/`

---

## 5. End‑to‑End Workflow

1. **Developer** does `git clone` & `make env install`.
2. **Data Sync**: `make train` → internally runs `dvc pull`.
3. **Training**: ZenML DockerSequentialRunner executes all steps; logs to MLflow; artifacts → `./models/`.
4. **Serve**: `docker-compose up -d` brings up Nginx & FastAPI, which delegates inference to MLflow Serve.
5. **Quality**: build passes only if `make fmt`, `make lint`, `make check`, and `make test` all succeed.
6. **Alerts**: CI or pipeline failures pushed to Discord.

---

### 6. Next Actions

- Wire in your ZenML **on_failure** hooks to call Discord.
- Ensure your `docker-compose.yml` mounts `./data` and `./models` correctly.
- Add the missing `make tune` and `make predict` targets once scripts exist.
- Document all commands in your README, referencing the HLD above.
