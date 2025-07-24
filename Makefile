# ──────────────────────────────────────────────
# 🛠️  Makefile for ML Pipeline Project
# ──────────────────────────────────────────────

# Main entrypoint: env + install + test
all:
	@echo "\033[1;36m🏁 Running full pipeline: env → install → test\033[0m"
	make env
	make install
	make devup_bg
	make wait_for_devup
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
	poetry run zenml integration install sklearn xgboost lightgbm mlflow great_expectations evidently -y

# ───────────────
# 🚀 Developer stack
# ───────────────

devup:
	@echo "\033[1;34m🔹 Starting ZenML stack & MLflow server (foreground)...\033[0m"
	poetry run zenml login --local
	lsof -ti tcp:7000 | xargs -r kill -9 && echo "✅ Killed processes on port 7000" || echo "🚫 No processes found."
	@echo "\033[1;31m🔹 Stopping ZenML stack...\033[0m"
	poetry run mlflow server \
		--backend-store-uri sqlite:///dvc/mlruns/mlflow.db \
		--default-artifact-root .dvc//mlruns/artifacts \
		--host 127.0.0.1 \
		--port 7000

devup_bg:
	@echo "\033[1;34m🔹 Starting ZenML stack in background (logs in devup.log)...\033[0m"
	nohup make devup > devup.log 2>&1 &
	@echo "\033[1;34m🔹 ZenML background process started (check devup.log for logs).\033[0m"

devdown:
	@echo "\033[1;31m🛑 Stopping ZenML and MLflow services...\033[0m"
	poetry run zenml logout --local || true
	poetry run zenml disconnect

	@echo "\033[1;31m🔹 Killing MLflow server (if running)...\033[0m"
	set -o pipefail; \
	lsof -ti tcp:7000 | xargs -r kill -9 \
		&& echo "✅ Killed processes on port 7000" \
		|| echo "🚫 No processes found on port 7000"

	@echo "\033[1;31m🔹 Stopping ZenML stack...\033[0m"

wait_for_devup:
	@echo "\033[1;33m🔸 Waiting for ZenML stack to become ready...\033[0m"
	@sleep 5
	@echo "\033[1;33m🔸 Waiting for MLflow server at localhost:7000...\033[0m"
	@SECONDS=0; \
	while ! curl -s http://127.0.0.1:7000/ >/dev/null; do \
		if [ $$SECONDS -gt 30 ]; then \
			echo "❌ Timeout waiting for MLflow server!"; exit 1; \
		fi; \
		sleep 1; \
	done; \
	echo "✅ MLflow server is up!"

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

maintain:
	@echo "\033[1;34m🔹 Running maintenance tasks: fmt, lint, check...\033[0m"
	make fmt
	make lint
	make check

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
	@echo "  all       → Setup env, install dependencies, and run test pipeline"
	@echo "  env       → Create or activate poetry environment"
	@echo "  install   → Install dependencies and ZenML integrations"
	@echo "  devup     → Start ZenML & MLflow in foreground"
	@echo "  devup_bg  → Start ZenML & MLflow in background (logs in devup.log)"
	@echo "  devdown   → Stop ZenML & MLflow services"
	@echo "  wait_for_devup → Wait until dev services are ready"
	@echo "  train     → Run the training pipeline"
	@echo "  test      → Run the test pipeline"
	@echo "  tune      → hyperparameter tuning (placeholder)"
	@echo "  predict   → inference scripts (placeholder)"
	@echo "  fmt       → Format code with black"
	@echo "  lint      → Lint code with flake8"
	@echo "  check     → Type-check code with mypy"
	@echo "  clean     → Remove caches & stop dev services (devdown)"
	@echo "  help      → Show this help message"

.PHONY: all env install devup devup_bg devdown wait_for_devup train test tune predict fmt lint check clean help maintain
