# utils/monitor.py

import functools
import logging
import time
import traceback
import tracemalloc
import sys
import inspect
import json
import os
import joblib
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None

try:
    import mlflow
except ImportError:
    mlflow = None

logging.basicConfig(
    filename="pipeline_monitor.log",
    filemode="a",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("UniversalPipelineLogger")
logger.setLevel(logging.INFO)


def estimate_size(obj):
    try:
        if hasattr(obj, "memory_usage"):
            return obj.memory_usage(deep=True).sum() / 1024**2
        elif hasattr(obj, "nbytes"):
            return obj.nbytes / 1024**2
        elif isinstance(obj, list):
            return sum(sys.getsizeof(i) for i in obj) / 1024**2
        else:
            return sys.getsizeof(obj) / 1024**2
    except Exception:
        return None


def log_resource_usage(step_id: str):
    if psutil:
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        logger.info(f"[{step_id}] CPU: {cpu:.1f}% | RAM: {mem:.1f}%")


def log_report(report: dict, name: str = "report"):
    if mlflow:
        try:
            filename = f"{name}.json"
            with open(filename, "w") as f:
                json.dump(report, f, indent=2)
            mlflow.log_artifact(filename)
            os.remove(filename)
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to log report: {e}")


def log_model_artifact(obj, name: str = "model.joblib"):
    if mlflow:
        try:
            joblib.dump(obj, name)
            mlflow.log_artifact(name)
            os.remove(name)
        except Exception as e:
            logger.warning(f"[MLFLOW] Failed to log model: {e}")


def monitor(
    name: str = None,
    log_args: bool = True,
    log_result: bool = True,
    track_memory: bool = True,
    track_input_size: bool = True,
    retries: int = 0,
    enabled: bool = True,
    mlflow_report: dict = None,
    mlflow_model: Any = None,
):
    """
    Universal decorator for logging, memory/time tracking, and optional MLflow reporting.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not enabled:
                return func(*args, **kwargs)

            step_id = name or func.__name__
            start_time = time.time()
            logger.info(f"[{step_id}] STARTED")

            if track_input_size:
                try:
                    sig = inspect.signature(func)
                    bound = sig.bind(*args, **kwargs)
                    bound.apply_defaults()
                    sizes = {
                        k: f"{estimate_size(v):.2f} MB"
                        for k, v in bound.arguments.items()
                        if estimate_size(v) is not None
                    }
                    logger.info(f"[{step_id}] Input sizes: {sizes}")
                except Exception:
                    logger.warning(f"[{step_id}] Failed to estimate input sizes")

            if log_args:
                logger.info(f"[{step_id}] Args: {args}")
                logger.info(f"[{step_id}] Kwargs: {kwargs}")

            if track_memory:
                tracemalloc.start()

            last_exception = None
            for attempt in range(retries + 1):
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    logger.info(f"[{step_id}] SUCCESS in {duration:.2f}s")

                    if log_result:
                        try:
                            logger.info(f"[{step_id}] Result type: {type(result)}")
                            if hasattr(result, "shape"):
                                logger.info(f"[{step_id}] Result shape: {result.shape}")
                        except Exception:
                            logger.warning(f"[{step_id}] Result inspection failed")

                    if track_memory:
                        current, peak = tracemalloc.get_traced_memory()
                        logger.info(
                            f"[{step_id}] Peak memory: {peak / 1024 / 1024:.2f} MB"
                        )
                        tracemalloc.stop()

                    log_resource_usage(step_id)

                    # MLflow artifact logging if applicable
                    if mlflow_report:
                        log_report(mlflow_report, name=f"{step_id}_report")
                    if mlflow_model:
                        log_model_artifact(mlflow_model, name=f"{step_id}_model.joblib")

                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    logger.error(
                        f"[{step_id}] FAILED (attempt {attempt+1}) in {duration:.2f}s"
                    )
                    logger.error(f"[{step_id}] Exception: {str(e)}")
                    logger.error(traceback.format_exc())
                    last_exception = e
                    time.sleep(0.5 * attempt)

            raise last_exception

        return wrapper

    return decorator
