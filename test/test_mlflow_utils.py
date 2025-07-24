# tests/test_mlflow_utils.py

import importlib  # ← make sure this is here
import os
import subprocess
from unittest.mock import MagicMock

import mlflow
import pytest
import yaml
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from src.utils import mlflow_utils
from src.utils.mlflow_utils import *

# Top‐level stubs so imports never break
mlflow.set_experiment = lambda *a, **k: None
mlflow.set_tracking_uri = lambda *a, **k: None

# ────────────────────────────────────────────────────────────────────────────────
# Autouse fixture: mock side‐effects for smoke tests only.
# Skip for integration tests (marked @pytest.mark.integration).
# ────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _mock_smoke(monkeypatch, tmp_path, request):
    if request.node.get_closest_marker("integration"):
        return  # don’t mock in integration

    # Stub mlflow env calls
    monkeypatch.setattr(mlflow, "set_tracking_uri", lambda *a, **k: None)
    monkeypatch.setattr(mlflow, "set_experiment", lambda *a, **k: None)

    # Mock the internal client
    client = MagicMock()
    monkeypatch.setattr(mlflow_utils, "_client", client)

    # Fake download_artifacts
    def fake_download_artifacts(*args, **kwargs):
        dst = kwargs.get("dst_path") or tmp_path / "dl"
        os.makedirs(dst, exist_ok=True)
        mlfile = os.path.join(dst, "MLmodel")
        with open(mlfile, "w") as f:
            yaml.safe_dump({"flavors": {"sklearn": {}, "pyfunc": {}}}, f)
        return mlfile

    monkeypatch.setattr(mlflow_utils, "download_artifacts", fake_download_artifacts)

    # Patch list_artifacts
    monkeypatch.setattr(
        mlflow_utils.mlflow, "list_artifacts", lambda *a, **k: [], raising=False
    )

    # Stub subprocess.Popen
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: MagicMock(pid=9999))


# ────────────────────────────────────────────────────────────────────────────────
# Smoke/unit tests (all mocked)
# ────────────────────────────────────────────────────────────────────────────────


def test_log_params_and_metrics(monkeypatch):
    store = {}
    monkeypatch.setattr(mlflow_utils.mlflow, "log_params", lambda p: store.update(p))
    log_params({"foo": 42})
    assert store["foo"] == 42

    calls = []
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_metric",
        lambda k, v, step=None: calls.append((k, v, step)),
    )
    log_metrics({"acc": 0.8}, step=5)
    assert calls == [("acc", 0.8, 5)]


def test_artifacts_and_listing(tmp_path, monkeypatch):
    # single file
    f = tmp_path / "x.txt"
    f.write_text("hello")
    rec = {}
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_artifact",
        lambda p, artifact_path=None: rec.update({"p": p, "ap": artifact_path}),
    )
    log_artifacts(str(f), artifact_path="dest")
    assert rec == {"p": str(f), "ap": "dest"}

    # directory
    d = tmp_path / "d"
    d.mkdir()
    (d / "a").write_text("A")
    rec.clear()
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_artifacts",
        lambda p, artifact_path=None: rec.update({"p": p}),
    )
    log_artifacts(str(d))
    assert rec["p"] == str(d)

    # list_artifacts
    fake = [MagicMock(path="a"), MagicMock(path="b")]
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "list_artifacts",
        lambda path, run_id=None: fake,
        raising=False,
    )
    assert list_artifacts("p", "r") == ["a", "b"]


def test_download_and_flavors(tmp_path):
    out = download_artifact("runs:/1/2", dst_path=str(tmp_path / "out"))
    assert out.endswith("MLmodel")
    flavors = list_model_flavors("models:/1/2")
    assert set(flavors) == {"sklearn", "pyfunc"}


def test_start_and_end_run(monkeypatch):
    dummy = MagicMock()

    class C:
        def __enter__(self):
            return dummy

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(mlflow_utils.mlflow, "start_run", lambda **k: C())
    with start_run("r", experiment_name="smoke") as run:
        assert run is dummy
    assert callable(end_run)


def test_enable_autologging(monkeypatch):
    fake = MagicMock()
    monkeypatch.setitem(mlflow_utils._AUTOLOGGERS, "X", "fake.mod")
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: fake if name == "fake.mod" else importlib.import_module(name),
    )
    enable_autologging("X")
    fake.autolog.assert_called_once()
    with pytest.raises(ValueError):
        enable_autologging("None")


def test_sklearn_model_log_load(monkeypatch):
    import mlflow.sklearn as sk

    dummy = MagicMock()
    monkeypatch.setattr(sk, "log_model", lambda **k: None)
    monkeypatch.setattr(sk, "load_model", lambda uri: dummy)
    log_sklearn_model(dummy, "ap")
    assert load_sklearn_model("uri") is dummy


def test_registry_operations(monkeypatch):
    mv = MagicMock()
    monkeypatch.setattr(mlflow_utils.mlflow, "register_model", lambda **k: mv)
    assert register_model("u", "n") is mv
    promote_model("n", 1, stage="P")
    mlflow_utils._client.transition_model_version_stage.assert_called_once()
    archive_model("n", 2)
    assert mlflow_utils._client.transition_model_version_stage.call_count == 2


def test_conda_env_and_serve(tmp_path, monkeypatch):
    p = tmp_path / "env.yml"
    p.write_text("c")
    monkeypatch.chdir(tmp_path)
    rec = {}
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "log_artifact",
        lambda path, artifact_path=None: rec.setdefault("p", path),
    )
    log_conda_env("env.yml")
    assert rec["p"] == "env.yml"

    rec.clear()
    monkeypatch.setattr(subprocess, "check_output", lambda *a: b"pkg\n")
    log_conda_env("none.yml")
    assert os.path.exists("requirements.txt")
    assert rec["p"] == "requirements.txt"

    proc = serve_model("models:/u/v")
    assert proc.pid == 9999


def test_list_runs_logic(monkeypatch):
    mlflow_utils._client.search_runs.return_value = ["R1"]
    assert list_runs() == ["R1"]
    mlflow_utils._client.get_experiment_by_name.return_value = None
    with pytest.raises(ValueError):
        list_runs(experiment_name="smoke")
    exp = MagicMock(experiment_id="EID")
    mlflow_utils._client.get_experiment_by_name.return_value = exp
    mlflow_utils._client.search_runs.reset_mock()
    mlflow_utils._client.search_runs.return_value = ["R2"]
    out = list_runs(
        experiment_name="smoke", filter_string="f", order_by=["o"], max_results=1
    )
    mlflow_utils._client.search_runs.assert_called_once_with(
        experiment_ids=["EID"],
        filter_string="f",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["o"],
    )
    assert out == ["R2"]


# … [all the smoke/unit tests above remain unchanged] …


# ────────────────────────────────────────────────────────────────────────────────
# Integration test (no mocks): logs into real experiment `testmlflow`
# ────────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────────
# Integration test (no mocks): logs into real experiment `testmlflow`
# ────────────────────────────────────────────────────────────────────────────────
@pytest.mark.integration
def test_real_mlflow_run(tmp_path):
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:7000")
    mlflow.set_tracking_uri(uri)

    client = MlflowClient(tracking_uri=uri)

    # 1) ensure experiment exists
    exp_name = "testmlflow"
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        client.create_experiment(exp_name)
        exp = client.get_experiment_by_name(exp_name)
    assert exp is not None

    # 2) log a real run into that experiment, capture run_id
    with start_run("integration_smoke", experiment_name=exp_name) as run:
        log_params({"p": 9})
        log_metrics({"m": 0.99})
        f = tmp_path / "out.txt"
        f.write_text("ok")
        log_artifacts(str(f), artifact_path="smoke")
        run_id = run.info.run_id

    # 3) directly fetch by run_id
    fetched = client.get_run(run_id)
    assert fetched is not None
    assert fetched.info.run_id == run_id
    assert fetched.info.run_name == "integration_smoke"

    print(
        f"\n✅ Successfully logged and fetched run {run_id!r} in '{exp_name}' (ID={exp.experiment_id})"
    )
