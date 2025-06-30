# ──────────────────────────────────────────────
# 🛠️  Makefile for ML Pipeline Project
# ──────────────────────────────────────────────

# Main entrypoint: env + install + test
all:
	@echo "\033[1;36m🏁 Running full pipeline: env → install → test\033[0m"
	make env
	make install
	make devup_bg
	make test

# ───────────────
# 🔧 Environment
# ───────────────

env:
	@echo "\033[1;34m🔹 Setting up Python environment...\033[0m"
	poetry env list || poetry env use python3.11

install:
	@echo "\033[1;34m🔹 Installing dependencies...\033[0m"
	poetry install
	poetry run zenml integration install s3 sklearn mlflow -y

# ───────────────
# 🚀 Developer stack
# ───────────────

devup:
	@echo "\033[1;34m🔹 Starting ZenML stack & MLflow server (foreground)...\033[0m"
	poetry run zenml up
	poetry run mlflow server \
		--backend-store-uri sqlite:///.zen/mlflow.db \
		--default-artifact-root ./.zen/mlruns \
		--host 127.0.0.1 \
		--port 7000

devup_bg:
	@echo "\033[1;34m🔹 Starting dev stack in background (logs in devup.log)...\033[0m"
	nohup make devup > devup.log 2>&1 &

devdown:
	@echo "\033[1;31m🛑 Stopping ZenML and MLflow services...\033[0m"
	poetry run zenml down || true
	@echo "\033[1;31m🔹 Killing MLflow server (if running)...\033[0m"
	-pkill -f "mlflow server" || true

# ───────────────
# 🏗️ Pipeline execution
# ───────────────

train:
	@echo "\033[1;32m🚀 Running training pipeline...\033[0m"
	poetry run python pipeline/train.py

test:
	@echo "\033[1;32m🚀 Running test pipeline...\033[0m"
	poetry run python pipeline/test.py

tune:
	@echo "\033[1;33m⚠️  Tune target not implemented yet. Add a tuning script or remove this target.\033[0m"
	@exit 1

predict:
	@echo "\033[1;33m⚠️  Predict target not implemented yet. Add a prediction script or remove this target.\033[0m"
	@exit 1

# ───────────────
# 🧹 Maintenance
# ───────────────

fmt:
	@echo "\033[1;34m🔹 Formatting code with black...\033[0m"
	poetry run black .

lint:
	@echo "\033[1;34m🔹 Linting code with flake8...\033[0m"
	poetry run flake8 src/

check:
	@echo "\033[1;34m🔹 Running type checks with mypy...\033[0m"
	poetry run mypy src/ pipeline/

clean:
	@echo "\033[1;34m🔹 Cleaning Python caches and artifacts...\033[0m"
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .zen .mlruns .mypy_cache .pytest_cache .coverage dist
	@echo "\033[1;31m🛑 Also stopping any running dev services (devdown)...\033[0m"
	make devdown

# ───────────────
# 📢 Utility
# ───────────────

help:
	@echo "\033[1;36mAvailable targets:\033[0m"
	@echo "  all       → Setup env, install dependencies, and run tests"
	@echo "  env       → Create or activate poetry environment"
	@echo "  install   → Install dependencies and ZenML integrations"
	@echo "  devup     → Start ZenML & MLflow in foreground"
	@echo "  devup_bg  → Start ZenML & MLflow in background (logs in devup.log)"
	@echo "  devdown   → Stop ZenML & MLflow services"
	@echo "  train     → Run the training pipeline"
	@echo "  test      → Run the test pipeline"
	@echo "  tune      → Placeholder for hyperparameter tuning"
	@echo "  predict   → Placeholder for inference scripts"
	@echo "  fmt       → Format code with black"
	@echo "  lint      → Lint code with flake8"
	@echo "  clean     → Remove caches & stop dev services (devdown)"
	@echo "  help      → Show this help message"

.PHONY: all env install devup devup_bg devdown train test tune predict fmt lint clean help
