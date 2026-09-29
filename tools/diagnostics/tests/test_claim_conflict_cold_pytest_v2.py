"""Offline controls for the cold-import candidate's isolated full-suite stage."""
import json
import inspect
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_cold_pytest as previous
from tools.diagnostics import claim_conflict_cold_pytest_v2 as cold


def test_v2_keeps_v1_application_and_baseline_source_logic():
    for name in ("derive_candidate_manifest", "load_base", "remote_install", "worker"):
        assert inspect.getsource(getattr(cold, name)) == inspect.getsource(
            getattr(previous, name))
    assert cold.PARENT_CANDIDATE_SHA == previous.PARENT_CANDIDATE_SHA
    assert cold.PARENT_PHASE1_SHA == previous.PARENT_PHASE1_SHA
    assert cold.NEW_PHASE1_SHA == previous.NEW_PHASE1_SHA
    assert cold.BASELINE_PROOF_RESULT_SHA == previous.BASELINE_PROOF_RESULT_SHA
    assert cold.BASE_SUITE_SHA == previous.BASE_SUITE_SHA
    assert cold.FULL_SECONDS == previous.FULL_SECONDS == 10800
    assert cold.HOST_SECONDS == previous.HOST_SECONDS == 11200


def test_eighteenth_overlay_mutation_fails_before_prepare(monkeypatch, tmp_path):
    copied = tmp_path / "tests"
    shutil.copytree(cold.TEST_ROOT, copied)
    monkeypatch.setattr(cold, "TEST_ROOT", copied)
    target = copied / "test_orphan_quarantine_application.py"
    target.write_bytes(target.read_bytes() + b"\n# altered\n")
    with pytest.raises(RuntimeError, match="cold_test_overlay_pin_drift"):
        cold.overlay_manifest(copied)
    monkeypatch.setattr(cold, "REVIEWED_FINAL_CANDIDATE", True)
    with pytest.raises(RuntimeError, match="reviewed_test_root_required"):
        cold.prepare(tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_review_gate_blocks_prepare_and_install_without_touching_stage(monkeypatch, tmp_path):
    assert type(cold.REVIEWED_FINAL_CANDIDATE) is bool
    monkeypatch.setattr(cold, "REVIEWED_FINAL_CANDIDATE", False)
    destination = tmp_path / "prepared"
    with pytest.raises(RuntimeError, match="final_cold_candidate_review_pending"):
        cold.prepare(destination)
    with pytest.raises(RuntimeError, match="final_cold_candidate_review_pending"):
        cold.remote_install(SimpleNamespace())
    assert not destination.exists()


def test_reviewed_prepare_contains_only_code_and_exact_overlay(monkeypatch, tmp_path):
    monkeypatch.setattr(cold, "REVIEWED_FINAL_CANDIDATE", True)
    output = tmp_path / "prepared"
    result = cold.prepare(output)
    assert result["status"] == "prepared_not_uploaded"
    assert result["test_overrides"] == 18
    assert result["overlay_sha256"] == cold.NEW_OVERLAY_SHA
    assert result["phase1_sha256"] == cold.NEW_PHASE1_SHA
    assert result["baseline_proof_only"] is True
    assert result["new_candidate_replay_verified"] is False
    assert cold.sha(output / "overrides/hymem/dreaming/phase1.py") == cold.NEW_PHASE1_SHA
    assert cold.digest(cold.load_base(local_prepare=True).inventory(output / "overlay")) == cold.NEW_OVERLAY_SHA
    assert (output / cold.SELF.name).is_file()
    assert not (output / "candidate").exists()
    assert not (output / "capture").exists()


def test_overlay_pins_preserve_original_seventeen_and_add_exact_orphan_fix():
    manifest = cold.overlay_manifest()
    parent = {name: manifest[name] for name in (
        "tests/" + item for item in cold.PARENT_TEST_NAMES
    )}
    assert len(parent) == 14 and len(manifest) == 18
    assert cold.digest(parent) == cold.PARENT_OVERLAY_SHA
    assert cold.digest(manifest) == cold.NEW_OVERLAY_SHA
    previous_manifest = {name: manifest[name] for name in (
        "tests/" + item for item in previous.TEST_NAMES
    )}
    assert cold.digest(previous_manifest) == previous.NEW_OVERLAY_SHA
    assert cold.sha(cold.PHASE1_SOURCE) == cold.NEW_PHASE1_SHA
    assert cold.TEST_NAMES[-4:] == (
        "test_claim_cold_import_replay.py", "test_claim_cold_import_replay_root.py",
        "test_claim_cold_replay_reviewer_retired.py",
        "test_orphan_quarantine_application.py",
    )
    assert manifest["tests/test_orphan_quarantine_application.py"] == (
        "14c63215f382f7f2b27b229fce79b9492b3794e20aa0cbdc80ef7ced61e8501f"
    )


def parent_manifest(monkeypatch):
    parent = {
        "hymem/dreaming/phase1.py": cold.PARENT_PHASE1_SHA,
        **{"source/%03d.py" % index: "a" * 64 for index in range(480)},
    }
    monkeypatch.setattr(cold, "PARENT_CANDIDATE_SHA", cold.digest(parent))
    return parent


def test_candidate_manifest_is_exact_single_phase1_delta(monkeypatch):
    parent = parent_manifest(monkeypatch)
    changed = cold.derive_candidate_manifest(parent)
    assert len(changed) == 481
    assert changed["hymem/dreaming/phase1.py"] == cold.NEW_PHASE1_SHA
    assert all(changed[name] == value for name, value in parent.items()
               if name != "hymem/dreaming/phase1.py")
    tampered = dict(parent)
    tampered["source/007.py"] = "b" * 64
    with pytest.raises(RuntimeError, match="parent_candidate_manifest_drift"):
        cold.derive_candidate_manifest(tampered)
    tampered = dict(parent)
    tampered["hymem/dreaming/phase1.py"] = cold.NEW_PHASE1_SHA
    with pytest.raises(RuntimeError, match="parent_candidate_manifest_drift"):
        cold.derive_candidate_manifest(tampered)


def test_pinned_base_is_reused_with_new_one_shot_network_none_container(monkeypatch):
    local_base = Path(__file__).resolve().parents[1] / "claim_conflict_proof_pytest.py"
    monkeypatch.setattr(cold, "BASE_SUITE", local_base)
    base = cold.configure_base()
    helper = SimpleNamespace(RUNTIME=Path("/runtime"), IMAGE="sha256:" + "a" * 64)
    command, mounts = base.configure(helper)
    assert command[command.index("--name") + 1] == "hymem-cold-replay-pytest-v2"
    assert command[command.index("--network") + 1] == "none"
    assert "--read-only" in command and "--cap-drop" in command
    assert [(dst, rw) for _, dst, rw in mounts if rw] == [("/work", True)]
    assert (str(local_base), "/diag/base_suite.py", False) in mounts
    assert not any(dst in ("/capture", "/production", "/root") for _, dst, _ in mounts)
    assert base.FULL_SECONDS == 10800 and base.HOST_SECONDS == 11200
    assert base.MIN_COLLECTED == 7752
    assert base.TEST_NAMES == cold.TEST_NAMES
    assert base.PROOF_ROOT.name == "proof-replay-v3"


def test_new_collection_floor_and_one_full_run_are_enforced(monkeypatch):
    local_base = Path(__file__).resolve().parents[1] / "claim_conflict_proof_pytest.py"
    monkeypatch.setattr(cold, "BASE_SUITE", local_base)
    base = cold.configure_base()
    counts = {
        "collected": 7752, "passed": 7752, "failed": 0,
        "skipped": 0, "errors": 0, "exit_code": 0,
    }
    raw = {
        "status": "passed", "candidate_sha256": base.CANDIDATE_SHA,
        "test_inventory_sha256": "a" * 64,
        "application_unchanged": True, "pinned_source_unchanged": True,
        "full_runs_started": 1,
        "collect": {**counts, "passed": 0},
        "full": counts.copy(),
    }
    assert base.project_worker(raw)["full_runs_started"] == 1
    raw["collect"]["collected"] = 7751
    with pytest.raises(RuntimeError, match="worker_collection_invalid"):
        base.project_worker(raw)
    raw["collect"]["collected"] = 7752
    raw["full_runs_started"] = 2
    with pytest.raises(RuntimeError, match="worker_run_count_invalid"):
        base.project_worker(raw)


def test_pins_label_old_proof_as_baseline_only(monkeypatch):
    fake = SimpleNamespace(
        configure=lambda _: (["--workdir", "/work", "--name", "old"], []),
        pins=lambda *_: {"proof_result_sha256": cold.BASELINE_PROOF_RESULT_SHA},
    )
    monkeypatch.setattr(cold, "load_base", lambda **_: fake)
    base = cold.configure_base()
    receipt = base.pins(None, None, None)
    assert receipt["baseline_proof_only"] is True
    assert receipt["new_candidate_replay_verified"] is False
    assert receipt["candidate_phase1_sha256"] == cold.NEW_PHASE1_SHA
    assert receipt["parent_candidate_sha256"] == cold.PARENT_CANDIDATE_SHA
    rejected = SimpleNamespace(
        configure=lambda _: (["--workdir", "/work", "--name", "old"], []),
        pins=lambda *_: {"proof_result_sha256": "0" * 64},
    )
    monkeypatch.setattr(cold, "load_base", lambda **_: rejected)
    rejected = cold.configure_base()
    with pytest.raises(RuntimeError, match="baseline_proof_result_drift"):
        rejected.pins(None, None, None)


def test_inherited_pins_accept_parent_proof_and_distinct_new_candidate(
    monkeypatch, tmp_path,
):
    local_base = Path(__file__).resolve().parents[1] / "claim_conflict_proof_pytest.py"
    monkeypatch.setattr(cold, "BASE_SUITE", local_base)
    root = tmp_path / "cold-suite"
    overlay = root / "overlay/tests"
    overlay.mkdir(mode=0o700, parents=True)
    root.chmod(0o700)
    (root / "candidate").mkdir()
    self_file = root / "claim_conflict_cold_pytest_v2.py"
    self_file.write_text("# sealed adapter")
    self_file.chmod(0o400)
    overlay_manifest = {}
    for name in cold.TEST_NAMES:
        path = overlay / name
        path.write_text("# reviewed " + name)
        overlay_manifest["tests/" + name] = cold.sha(path)
    (root / "test-overlay.json").write_text(json.dumps(overlay_manifest))
    monkeypatch.setattr(cold, "ROOT", root)
    monkeypatch.setattr(cold, "SELF", self_file)
    monkeypatch.setattr(cold, "OVERLAY", root / "overlay")
    monkeypatch.setattr(cold, "CANDIDATE", root / "candidate")
    monkeypatch.setattr(cold, "PARENT_CANDIDATE", tmp_path / "parent")
    parent = parent_manifest(monkeypatch)
    changed = cold.derive_candidate_manifest(parent)
    base = cold.configure_base()
    monkeypatch.setattr(base.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(base, "REVIEWED_OVERLAY_SHA", cold.digest(overlay_manifest))
    monkeypatch.setattr(base, "inventory", lambda path: (
        overlay_manifest if path == cold.OVERLAY else
        parent if path == cold.PARENT_CANDIDATE else changed
    ))
    proof_root = tmp_path / "proof-replay-v3"
    proof_root.mkdir()
    proof_result = proof_root / "result.json"
    proof_result.write_text(json.dumps({
        "status": "completed", "networked_runs_started": 0,
        "stages": {"replay": {
            "container_id": "c" * 64, "metadata": {"valid": True},
        }},
    }))
    baseline = base.baseline_suite
    monkeypatch.setattr(baseline, "PROOF_ROOT", proof_root)
    monkeypatch.setattr(baseline, "PROOF_RESULT_SHA", baseline.sha(proof_result))
    monkeypatch.setattr(baseline, "CANDIDATE_SHA", cold.PARENT_CANDIDATE_SHA)
    monkeypatch.setattr(cold, "BASELINE_PROOF_RESULT_SHA", baseline.PROOF_RESULT_SHA)
    overrides = {"hymem/dreaming/phase1.py": cold.PARENT_PHASE1_SHA}
    receipt = {
        "source_files": 481, "host_sha256": baseline.PROOF_HOST_SHA,
        "worker_sha256": "b" * 64, "phase1_sha256": cold.PARENT_PHASE1_SHA,
        "override_sha256": base.digest(overrides),
        "candidate_sha256": cold.PARENT_CANDIDATE_SHA,
        "audit_result_sha256": "a" * 64,
    }
    helper = SimpleNamespace(
        regular=lambda path: path.is_file() or pytest.fail("missing sealed file"),
        read_json=lambda path: json.loads(path.read_text()),
    )
    shared = SimpleNamespace(
        baseline_inventory=lambda _: parent.copy(), OVERRIDE_SHAS={},
    )
    proof = SimpleNamespace(
        WORKER_SHA="b" * 64, reviewed_overrides=lambda: overrides,
        dependencies=lambda: (shared, None, helper),
        installed=lambda *_: receipt,
        inspect=lambda *_: {
            "status": "exited", "pid": 0, "exit_code": 0, "oom_killed": False,
        },
        configure=lambda *_: ([], []),
        project=lambda raw: raw,
        verdict=lambda metadata, _: metadata == {"valid": True}
        or pytest.fail("invalid verdict"),
    )
    pinned = base.pins(proof, shared, helper)
    assert pinned["candidate_sha256"] == cold.digest(changed)
    assert pinned["proof_result_sha256"] == baseline.PROOF_RESULT_SHA
    assert pinned["baseline_proof_only"] is True
    assert pinned["new_candidate_replay_verified"] is False
    receipt["candidate_sha256"] = cold.digest(changed)
    with pytest.raises(RuntimeError, match="proof_v3_receipt_drift"):
        base.pins(proof, shared, helper)


def test_source_pins_new_candidate_and_rejects_mutation(monkeypatch):
    parent = parent_manifest(monkeypatch)
    changed = cold.derive_candidate_manifest(parent)
    fake = SimpleNamespace(
        configure=lambda _: (["--workdir", "/work", "--name", "old"], []),
        pins=lambda *_: {"proof_result_sha256": cold.BASELINE_PROOF_RESULT_SHA},
        inventory=lambda path: parent if path == cold.PARENT_CANDIDATE else changed,
    )
    monkeypatch.setattr(cold, "load_base", lambda **_: fake)
    base = cold.configure_base()
    shared = SimpleNamespace(baseline_inventory=lambda _: parent.copy(), OVERRIDE_SHAS={})
    proof = SimpleNamespace(reviewed_overrides=lambda: {})
    assert base.source_pins(shared, proof, object()) == changed
    assert base.CANDIDATE_SHA == cold.digest(changed)
    fake.inventory = lambda path: parent if path == cold.PARENT_CANDIDATE else {
        **changed, "unreviewed.py": "f" * 64
    }
    with pytest.raises(RuntimeError, match="cold_candidate_pin_drift"):
        base.source_pins(shared, proof, object())


def test_remote_install_copies_only_reviewed_phase1_after_parent_and_proof_checks(
    monkeypatch, tmp_path,
):
    root = tmp_path / "new-stage"
    parent_tree = tmp_path / "parent-candidate"
    overlay = root / "overlay/tests"
    uploaded = root / "overrides/hymem/dreaming/phase1.py"
    old_phase1 = parent_tree / "hymem/dreaming/phase1.py"
    for path in (overlay, uploaded.parent, old_phase1.parent):
        path.mkdir(parents=True, exist_ok=True)
    old_phase1.write_text("parent phase1")
    uploaded.write_text("new phase1")
    for index in range(480):
        target = parent_tree / ("source/%03d.py" % index)
        target.parent.mkdir(exist_ok=True)
        target.write_text("pinned %d" % index)
    overlay_file = overlay / "test_control.py"
    overlay_file.write_text("reviewed test")
    manifest = {"tests/test_control.py": cold.sha(overlay_file)}
    (root / "test-overlay.json").write_text(json.dumps(manifest))
    inventory = lambda tree: {
        file.relative_to(tree).as_posix(): cold.sha(file)
        for file in tree.rglob("*") if file.is_file()
    }
    parent = inventory(parent_tree)
    monkeypatch.setattr(cold, "ROOT", root)
    monkeypatch.setattr(cold, "WORK", root / "work")
    monkeypatch.setattr(cold, "CANDIDATE", root / "candidate")
    monkeypatch.setattr(cold, "OVERLAY", root / "overlay")
    monkeypatch.setattr(cold, "PARENT_CANDIDATE", parent_tree)
    monkeypatch.setattr(cold, "PHASE1_UPLOAD", uploaded)
    monkeypatch.setattr(cold, "PARENT_PHASE1_SHA", cold.sha(old_phase1))
    monkeypatch.setattr(cold, "NEW_PHASE1_SHA", cold.sha(uploaded))
    monkeypatch.setattr(cold, "PARENT_CANDIDATE_SHA", cold.digest(parent))
    monkeypatch.setattr(cold, "NEW_OVERLAY_SHA", cold.digest(manifest))
    monkeypatch.setattr(cold, "REVIEWED_FINAL_CANDIDATE", True)
    proof = SimpleNamespace(reviewed_overrides=lambda: {})
    shared = SimpleNamespace(baseline_inventory=lambda _: parent.copy(), OVERRIDE_SHAS={})
    helper = SimpleNamespace(read_json=lambda path: json.loads(path.read_text()))
    base = SimpleNamespace(
        dependencies=lambda: (proof, shared, helper),
        inventory=inventory,
        replay_gate=lambda *_: cold.BASELINE_PROOF_RESULT_SHA,
        remote=lambda action: {"status": "installed_not_launched", "action": action},
    )
    uploaded.write_text("unreviewed replacement")
    with pytest.raises(RuntimeError, match="cold_phase1_upload_pin_drift"):
        cold.remote_install(base)
    assert not cold.CANDIDATE.exists()
    uploaded.write_text("new phase1")
    result = cold.remote_install(base)
    assert result["action"] == "remote-install"
    expected = cold.derive_candidate_manifest(parent)
    assert inventory(cold.CANDIDATE) == expected
    assert inventory(parent_tree) == parent


def test_worker_manifest_cannot_mislabel_new_app_as_parent(monkeypatch, tmp_path):
    parent = parent_manifest(monkeypatch)
    changed = cold.derive_candidate_manifest(parent)
    source_manifest = tmp_path / "candidate-manifest.json"
    source_manifest.write_text(json.dumps(changed))
    fake = SimpleNamespace(SOURCE_MANIFEST=source_manifest,
                           CANDIDATE_SHA=None, worker=lambda: {"status": "passed"})
    assert cold.worker(fake) == {"status": "passed"}
    assert fake.CANDIDATE_SHA == cold.digest(changed)
    source_manifest.write_text(json.dumps(parent))
    with pytest.raises(RuntimeError, match="worker_cold_candidate_manifest_drift"):
        cold.worker(fake)
