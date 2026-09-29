"""Inert local controller checks. Docker/Popen are mocked in every control."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PATH = Path(__file__).resolve().parents[1] / "claim_conflict_v64_production_dream_host.py"
spec = importlib.util.spec_from_file_location("production_host_test", PATH)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    benchmarks = tmp_path / "benchmarks"
    benchmarks.mkdir()
    monkeypatch.setattr(host, "BENCHMARKS", benchmarks)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    files = {}
    for name in (host.WORKER, "launch-manifest.json", "meter.py", "supervisor.py"):
        raw = b"inert sealed test fixture"
        host.write(inputs / name, raw)
        files[name] = hashlib.sha256(raw).hexdigest()
    value = {"version": host.VERSION, "root_reviewed": True,
        "controller_sha256": host.sha(PATH), "host_stage": str(benchmarks / "fresh-v1"),
        "container_stage": str(host.CONTAINER_BENCHMARKS / "fresh-v1"),
        "container": "hermes-1", "container_id": "id", "image_id": "image",
        "source_host": "/source", "mounts": {
            "/home/node/HyMem": {"source": "/source", "rw": True, "type": "bind"},
            "/home/node/.hermes": {"source": str(benchmarks.parent), "rw": True, "type": "bind"}},
        "interpreter": "/home/node/hymem-env/bin/python3", "files": files,
        "manifest": "launch-manifest.json", "manifest_sha256": files["launch-manifest.json"]}
    path = inputs / "bundle.json"
    host.json_write(path, value)
    seal = host.sha(path)
    run = Mock(side_effect=AssertionError("unexpected_process"))
    popen = Mock(side_effect=AssertionError("unexpected_launch"))
    monkeypatch.setattr(host.subprocess, "run", run)
    monkeypatch.setattr(host.subprocess, "Popen", popen)
    return SimpleNamespace(path=path, seal=seal, value=value, run=run, popen=popen)


def test_stage_is_private_exact_and_exclusive(bundle):
    h = bundle
    assert host.stage(h.path, h.seal)["files"] == 4
    root = Path(h.value["host_stage"])
    assert root.stat().st_mode & 0o777 == 0o700
    for name, digest in h.value["files"].items():
        assert host.sha(root / name) == digest
        assert (root / name).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        host.stage(h.path, h.seal)
    h.run.assert_not_called()
    h.popen.assert_not_called()


def test_unreviewed_descriptor_never_stages(bundle):
    h = bundle
    h.value["root_reviewed"] = False
    path = h.path.parent / "unreviewed.json"
    host.json_write(path, h.value)
    with pytest.raises(RuntimeError, match="external_review_required"):
        host.stage(path, host.sha(path))
    assert not Path(h.value["host_stage"]).exists()


def docker_receipt(h, **changes):
    row = {"Id": "id", "Image": "image", "State": {"Running": True,
        "Health": {"Status": "healthy"}}, "Mounts": [
        {"Destination": "/home/node/HyMem", "Source": "/source", "RW": True, "Type": "bind"},
        {"Destination": "/home/node/.hermes", "Source": str(host.BENCHMARKS.parent), "RW": True, "Type": "bind"}]}
    row.update(changes)
    return SimpleNamespace(stdout=json.dumps([row]).encode(), returncode=0)


def test_launch_detaches_once_with_durable_intent(bundle):
    h = bundle
    host.stage(h.path, h.seal)
    h.run.side_effect = None
    h.run.return_value = docker_receipt(h)
    def popen(_command, **kwargs):
        assert (Path(h.value["host_stage"]) / "host-launch-v1/intent.json").exists()
        assert kwargs["start_new_session"] is True
        assert kwargs["stdin"] == host.subprocess.DEVNULL
        return SimpleNamespace(pid=123)
    h.popen.side_effect = popen
    assert host.launch(h.path, h.seal)["pid"] == 123
    with pytest.raises(FileExistsError):
        host.launch(h.path, h.seal)
    assert h.popen.call_count == 1


@pytest.mark.parametrize("field,value", [("Id", "changed"), ("Image", "changed"),
    ("State", {"Running": True, "Health": {"Status": "unhealthy"}})])
def test_drift_refuses_launch(bundle, field, value):
    h = bundle
    host.stage(h.path, h.seal)
    h.run.side_effect = None
    h.run.return_value = docker_receipt(h, **{field: value})
    with pytest.raises(RuntimeError, match="container_runtime_changed"):
        host.launch(h.path, h.seal)
    h.popen.assert_not_called()


def test_supervisor_is_one_shot_even_when_called_directly(bundle):
    h = bundle
    host.stage(h.path, h.seal)
    intent = Path(h.value["host_stage"]) / "host-launch-v1"
    intent.mkdir(mode=0o700)
    host.json_write(intent / "intent.json", {"bundle_sha256": h.seal})
    h.run.side_effect = [docker_receipt(h), SimpleNamespace(returncode=0), docker_receipt(h)]
    assert host.supervise(h.path, h.seal)["status"] == "completed"
    with pytest.raises(FileExistsError):
        host.supervise(h.path, h.seal)
    assert len([call for call in h.run.call_args_list if call.args[0][:2] == ["docker", "exec"]]) == 1


def test_status_emits_only_static_codes_numbers_and_hashes(bundle):
    h = bundle
    host.stage(h.path, h.seal)
    root = Path(h.value["host_stage"])
    host.json_write(root / "production-dream-result.json", {"status": "completed",
        "completion_calls": 59, "http_attempts": 218, "private": "never emit",
        "error_code": "private provider failure"})
    result = host.status(h.path, h.seal)
    assert result["completion_calls"] == 59 and result["http_attempts"] == 218
    assert "never emit" not in str(result) and "private provider failure" not in str(result)
    h.run.assert_not_called()
    h.popen.assert_not_called()


def test_mapping_accepts_exact_sealed_ancestor_and_prefers_nested_bind():
    mounts = {"/home/node": {"source": "/opt/stacks/hermes/instance1/home",
        "rw": True, "type": "bind"}}
    assert host.mapped_path(mounts, "/home/node/.hermes/benchmarks/fresh-v1") == Path(
        "/opt/stacks/hermes/instance1/home/.hermes/benchmarks/fresh-v1")
    mounts["/home/node/HyMem"] = {"source": "/exact-source", "rw": False, "type": "bind"}
    assert host.mapped_path(mounts, "/home/node/HyMem") == Path("/exact-source")


def test_any_unsealed_extra_mount_refuses_before_launch(bundle):
    h = bundle
    host.stage(h.path, h.seal)
    inspected = docker_receipt(h)
    rows = json.loads(inspected.stdout)
    rows[0]["Mounts"].append({"Destination": "/home/node/.hermes/benchmarks",
        "Source": "/unexpected", "RW": True, "Type": "bind"})
    h.run.side_effect = None
    h.run.return_value = SimpleNamespace(stdout=json.dumps(rows).encode())
    with pytest.raises(RuntimeError, match="sealed_mounts_changed"):
        host.launch(h.path, h.seal)
    h.popen.assert_not_called()
