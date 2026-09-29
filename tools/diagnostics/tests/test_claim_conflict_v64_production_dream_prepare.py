import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

DIRECTORY = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, DIRECTORY / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_code_upload_has_exact_four_pinned_inputs():
    upload = module("claim_conflict_v64_production_dream_upload")
    pieces = upload.local_payload()
    assert len(pieces) == 4
    assert {name: hashlib.sha256(raw).hexdigest() for name, raw in pieces.items()} == upload.FILES
    assert "subprocess" not in upload.REMOTE and "docker" not in upload.REMOTE


def test_upload_is_install_only_and_never_chains_stage(monkeypatch):
    upload = module("claim_conflict_v64_production_dream_upload")
    run = Mock(return_value=SimpleNamespace(returncode=0,
        stdout=b'{"status":"installed_not_staged_not_launched","files":4}'))
    monkeypatch.setattr(upload.subprocess, "run", run)
    assert upload.install()["files"] == 4
    run.assert_called_once()
    assert run.call_args.args[0][:2] == ["ssh", "-C"]
    assert len(run.call_args.kwargs["input"]) == sum(len(raw) for raw in upload.local_payload().values())


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    bundle = module("claim_conflict_v64_production_dream_bundle")
    inputs = tmp_path / "inputs"
    inputs.mkdir(mode=0o700)
    monkeypatch.setattr(bundle, "INPUTS", inputs)
    monkeypatch.setattr(bundle, "HOST_STAGE", tmp_path / "final-stage")
    # Synthetic exact481 contract, isolated from the accepted production pins.
    files = {"file" + str(index): "a" * 64 for index in range(481)}
    candidate = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(bundle, "CANDIDATE", candidate)
    code = {}
    for name in bundle.CODE:
        raw = b"inert-code-fixture"
        bundle.write(inputs / name, raw)
        code[name] = bundle.digest(raw)
    monkeypatch.setattr(bundle, "CODE", code)
    gates = {}
    for label in ("full_suite", "private_paid_postflight", "deployment_start", "production_preflight", "role_profiles"):
        path = tmp_path / (label + ".json")
        raw = bundle.encode({"verified": True, "candidate_sha256": candidate})
        bundle.write(path, raw)
        gates[label] = {"path": str(path), "sha256": bundle.digest(raw)}
    spec = {"candidate_files": files, "gates": gates, "runtime_env_sha256": "b" * 64,
            "container_id": "container-id", "image_id": "image-id"}
    path = tmp_path / "spec.json"
    bundle.write(path, bundle.encode(spec))
    return SimpleNamespace(module=bundle, path=path, spec=spec, inputs=inputs)


def test_bundle_templates_are_unapproved_and_exact(evidence):
    h = evidence
    result = h.module.build(h.path)
    assert result["status"] == "templates_unapproved"
    manifest = json.loads((h.inputs / "launch-manifest.json").read_bytes())
    host = json.loads((h.inputs / "reviewed-bundle-template.json").read_bytes())
    assert manifest["root_reviewed"] is False and host["root_reviewed"] is False
    assert manifest["bounds"]["completions"] == 128
    assert host["mounts"] == {"/home/node": {"source": "/opt/stacks/hermes/instance1/home", "rw": True, "type": "bind"}}
    for name, digest in host["files"].items():
        assert h.module.digest((h.inputs / name).read_bytes()) == digest
        assert (h.inputs / name).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        h.module.build(h.path)


def test_unverified_required_receipt_refuses_before_writes(evidence):
    h = evidence
    item = h.spec["gates"]["full_suite"]
    path = Path(item["path"])
    path.unlink()
    raw = h.module.encode({"verified": False, "candidate_sha256": h.module.CANDIDATE})
    h.module.write(path, raw)
    item["sha256"] = h.module.digest(raw)
    h.path.unlink()
    h.module.write(h.path, h.module.encode(h.spec))
    with pytest.raises(RuntimeError, match="bundle_evidence_invalid"):
        h.module.build(h.path)
    assert not (h.inputs / "launch-manifest.json").exists()

