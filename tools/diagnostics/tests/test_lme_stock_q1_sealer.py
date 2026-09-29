"""Synthetic-only offline package fixtures; never seal an actual candidate."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import socket
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]
PATH = ROOT / "tools/diagnostics/lme_stock_q1/seal_successor.py"


@pytest.fixture
def helper(monkeypatch):
    module = types.ModuleType("q1_offline_seal_test")
    module.__file__ = str(PATH)
    exec(compile(PATH.read_bytes(), str(PATH), "exec"), module.__dict__)
    def denied(*args, **kwargs):
        pytest.fail("offline sealer must not use network")
    monkeypatch.setattr(socket.socket, "connect", denied)
    for name in ("getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo"):
        monkeypatch.setattr(socket, name, denied)
    return module


def write(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def inputs(tmp_path, helper):
    # Copy only the existing independently pinned selector function. Everything
    # else in the fixture is invented; no real candidate is packaged here.
    text = (ROOT / "benchmarks/longmemeval_adapter.py").read_text()
    function = next(node for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "select_label_blind_questions")
    selector = (ast.get_source_segment(text, function) + "\n").encode()
    tree = tmp_path / "synthetic-tree"
    source = {
        "benchmarks/longmemeval_adapter.py": write(tree / "benchmarks/longmemeval_adapter.py", selector),
        "hymem/__init__.py": write(tree / "hymem/__init__.py", b"# synthetic fixture, not a runnable application\n"),
    }
    manifest = tmp_path / "synthetic-r5.json"
    pin = write(manifest, helper.encoded({"revision": "r5", "source_sha256": source}))
    return {"source_manifest": manifest, "approved_manifest_sha256": pin, "tree": tree,
            "layout": "source", "remote_root": helper.REMOTE_BASE + "/q1-stock-v4",
            "remote_source": helper.REMOTE_BASE + "/offline-r5/candidate"}


def test_inspection_is_read_only_and_uses_exact_pinned_helpers(helper, inputs, tmp_path):
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    manifest, sources, helpers, raw = helper.prepare(**inputs)
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before
    assert manifest["source_files"] == 2 and len(sources) == 2
    assert {name: helper.sha(value) for name, value in helpers.items()} == helper.HELPER_PINS
    assert raw == inputs["source_manifest"].read_bytes()
    assert manifest["target_runtime_expected_versions"]["python"] == "3.11.2"
    for field in ("runtime_verified_by_sealing", "target_runtime_gate_pass_claimed", "benchmark_executed", "deployment_performed"):
        assert manifest[field] is False


def test_seal_synthetic_source_then_reverify_all_bytes(helper, inputs, tmp_path):
    prepared = helper.prepare(**inputs)
    output = tmp_path / "synthetic-sealed"
    receipt = helper.seal(prepared, output=output, input_tree=inputs["tree"])
    receipt_pin = helper.sha((output / "seal-receipt.json").read_bytes())
    verified = helper.verify_sealed(output=output, expected_receipt_sha256=receipt_pin)
    assert verified["source_files"] == 2
    assert verified["package_manifest_sha256"] == receipt["package_manifest_sha256"]
    assert receipt["paid_calls"] == receipt["remote_operations"] == 0
    assert verified["benchmark_readiness_claimed"] is False
    assert (output / "bundle/manifest.json").stat().st_mode & 0o777 == 0o400
    run = helper.load_definitions(prepared[2]["q1_stock_run.py"], "q1_synthetic_sealed_runner", output / "bundle/q1_stock_run.py")
    run.PACKAGE, run.SOURCE = output / "bundle", output / "candidate"
    run.validate_package(receipt["package_manifest_sha256"])
    run.verify_source(prepared[0])
    with pytest.raises(FileExistsError):
        helper.seal(prepared, output=output, input_tree=inputs["tree"])


def test_verification_tree_checks_but_does_not_copy_test_or_aux_content(helper, inputs, tmp_path):
    data = helper.decode(inputs["source_manifest"].read_bytes())
    data["test_sha256"] = {"tests/test_invented.py": write(inputs["tree"] / "tests/test_invented.py", b"def test_invented(): pass\n")}
    data["auxiliary_sha256"] = {"README.md": write(inputs["tree"] / "README.md", b"invented auxiliary file\n")}
    inputs["approved_manifest_sha256"] = write(inputs["source_manifest"], helper.encoded(data))
    inputs["layout"] = "verification"
    prepared = helper.prepare(**inputs)
    assert len(prepared[1]) == 2
    output = tmp_path / "combined-to-source-only"
    helper.seal(prepared, output=output, input_tree=inputs["tree"])
    assert not (output / "candidate/tests").exists()
    assert not (output / "candidate/README.md").exists()


@pytest.mark.parametrize("fault", ["extra", "source_hash", "source_symlink", "directory_symlink", "fifo"])
def test_input_inventory_corruption_is_rejected_before_any_output(helper, inputs, tmp_path, fault):
    tree = inputs["tree"]
    if fault == "extra":
        (tree / ".env").write_text("synthetic-only; must not be read")
    elif fault == "source_hash":
        (tree / "hymem/__init__.py").write_text("changed source")
    elif fault == "source_symlink":
        (tree / "link.py").symlink_to(tree / "hymem/__init__.py")
    elif fault == "directory_symlink":
        (tree / "linked").symlink_to(tmp_path, target_is_directory=True)
    else:
        helper.os.mkfifo(tree / "fifo")
    with pytest.raises(ValueError, match="q1_seal_tree_"):
        helper.prepare(**inputs)
    assert not (tmp_path / "sealed").exists()


@pytest.mark.parametrize("fault", ["manifest_pin", "old_revision", "duplicate_json", "overlap", "unsafe_path", "credential_map"])
def test_manifest_rejections(helper, inputs, fault):
    data = helper.decode(inputs["source_manifest"].read_bytes())
    if fault == "manifest_pin":
        inputs["approved_manifest_sha256"] = "0" * 64
    else:
        if fault == "old_revision":
            data["revision"] = "r4"
        elif fault == "overlap":
            data["test_sha256"] = dict(data["source_sha256"])
            data["auxiliary_sha256"] = {"README.md": "a" * 64}
            inputs["layout"] = "verification"
        elif fault == "unsafe_path":
            data["source_sha256"] = {"../outside.py": "a" * 64}
        elif fault == "credential_map":
            data["source_sha256"] = {"credentials.pem": "a" * 64}
        raw = b'{"revision":"r5","revision":"r5"}' if fault == "duplicate_json" else helper.encoded(data)
        inputs["approved_manifest_sha256"] = write(inputs["source_manifest"], raw)
    with pytest.raises(ValueError, match="q1_seal_"):
        helper.prepare(**inputs)


def test_pending_helper_drift_is_not_silently_repinned(helper, inputs, tmp_path, monkeypatch):
    pending = tmp_path / "changed-helpers"
    for name in helper.HELPER_PINS:
        write(pending / name, (helper.PENDING / name).read_bytes())
    (pending / "q1_stock_run.py").write_text("raise AssertionError('must never execute this')\n")
    monkeypatch.setattr(helper, "PENDING", pending)
    with pytest.raises(ValueError, match="q1_seal_tree_hash"):
        helper.prepare(**inputs)


@pytest.mark.parametrize("fault", ["candidate", "helper", "extra", "receipt", "wrong_receipt_pin"])
def test_postseal_corruption_fails_verification(helper, inputs, tmp_path, fault):
    output = tmp_path / "synthetic-sealed"
    helper.seal(helper.prepare(**inputs), output=output, input_tree=inputs["tree"])
    pin = helper.sha((output / "seal-receipt.json").read_bytes())
    if fault == "wrong_receipt_pin":
        pin = "a" * 64
    else:
        path = {"candidate": output / "candidate/hymem/__init__.py",
                "helper": output / "bundle/q1_stock_run.py",
                "extra": output / "unexpected.bin", "receipt": output / "seal-receipt.json"}[fault]
        if path.exists():
            path.chmod(0o600)
        path.write_bytes(b"tampered synthetic fixture\n")
    with pytest.raises(ValueError, match="q1_seal_"):
        helper.verify_sealed(output=output, expected_receipt_sha256=pin)


@pytest.mark.parametrize("field,value", [("remote_root", "/opt/stacks/hermes"),
                                          ("remote_root", "PREVIOUS"), ("remote_source", "OLD_SOURCE")])
def test_remote_metadata_rejects_broad_or_previous_targets(helper, inputs, field, value):
    replacements = {"PREVIOUS": helper.REMOTE_BASE + "/q1-stock-v3", "OLD_SOURCE": helper.REMOTE_BASE + "/offline-r3/candidate"}
    inputs[field] = replacements.get(value, value)
    with pytest.raises(ValueError, match="q1_seal_remote_"):
        helper.prepare(**inputs)


def test_output_cannot_modify_input_tree_or_existing_symlink(helper, inputs, tmp_path):
    prepared = helper.prepare(**inputs)
    with pytest.raises(ValueError, match="q1_seal_output_scope"):
        helper.seal(prepared, output=inputs["tree"] / "nested-output", input_tree=inputs["tree"])
    target = tmp_path / "outside"
    target.mkdir()
    link = tmp_path / "output-link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="q1_seal_output_scope"):
        helper.seal(prepared, output=link, input_tree=inputs["tree"])
    assert list(target.iterdir()) == []


def test_default_cli_inspects_without_writes(helper, inputs, tmp_path, monkeypatch, capsys):
    argv = [str(PATH)]
    for name, value in inputs.items():
        flag = "--approved-manifest-sha256" if name == "approved_manifest_sha256" else "--" + name.replace("_", "-")
        argv += [flag, str(value)]
    before = sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*"))
    monkeypatch.setattr(sys, "argv", argv)
    helper.main()
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "inspected_only_no_writes"
    assert sorted(str(path.relative_to(tmp_path)) for path in tmp_path.rglob("*")) == before


def test_synthetic_seal_is_byte_reproducible_in_separate_fresh_outputs(helper, inputs, tmp_path):
    prepared = helper.prepare(**inputs)
    first, second = tmp_path / "first", tmp_path / "second"
    helper.seal(prepared, output=first, input_tree=inputs["tree"])
    helper.seal(prepared, output=second, input_tree=inputs["tree"])
    a = {path.relative_to(first).as_posix(): path.read_bytes() for path in first.rglob("*") if path.is_file()}
    b = {path.relative_to(second).as_posix(): path.read_bytes() for path in second.rglob("*") if path.is_file()}
    assert a == b
