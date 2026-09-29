"""Independent network-free controls for the replay-gated private dream."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_alias_dream_host as dream
from tools.diagnostics import claim_conflict_alias_replay_host as replay
from tools.diagnostics import claim_conflict_alias_replay_v2_host as replay_v2
from tools.diagnostics import claim_conflict_shared_embedding_host as shared
from tools.diagnostics.tests.test_claim_conflict_alias_replay_root import _metadata


def test_corrected_replay_and_unchanged_dream_worker_dependency_pins():
    directory = Path(__file__).resolve().parents[1]
    assert dream.ALIAS_REPLAY == replay_v2.ROOT
    assert dream.ALIAS_HOST == replay_v2.SELF
    assert dream.sha(directory / "claim_conflict_alias_replay_v2_host.py") == dream.ALIAS_HOST_SHA
    assert dream.sha(directory / "claim_conflict_alias_replay_host.py") == replay_v2.V1_HOST_SHA
    assert dream.sha(directory / "claim_conflict_instrumented_replay.py") == replay_v2.WORKER_SHA
    assert dream.sha(directory / "claim_conflict_instrumented_dream.py") == dream.WORKER_SHA


def _replay_gate_fixture(monkeypatch):
    receipt = {
        "canonicalize_sha256": dream.CANONICALIZE_SHA,
        "phase1_sha256": dream.PHASE1_SHA, "source_files": 480,
        "capture_sha256": "d" * 64, "snapshot_sha256": "e" * 64,
    }
    result = {"status": "completed", "networked_runs_started": 0, "stages": {}}
    inspected = []
    for mode, char in (("baseline", "a"), ("fixed", "b")):
        result["stages"][mode] = {
            "container_id": char * 64, "status": "exited", "pid": 0,
            "exit_code": 0, "oom_killed": False, "configuration_verified": True,
            "metadata": replay.project(_metadata(mode)),
        }

    def inspect(_h, cid, mode, mounts):
        inspected.append(mode)
        assert cid == result["stages"][mode]["container_id"]
        return {"status": "exited", "pid": 0, "exit_code": 0, "oom_killed": False}

    alias = SimpleNamespace(
        installed=lambda *_: receipt, configure=lambda *_: ([], []),
        inspect=inspect, verdict=replay.verdict,
    )
    helper = SimpleNamespace(read_json=lambda *_: result)
    monkeypatch.setattr(dream, "sha", lambda *_: "f" * 64)
    return alias, helper, result, receipt, inspected


def test_replay_gate_rechecks_both_exact_verdicts_and_live_terminal_states(monkeypatch):
    alias, helper, _result, _receipt, inspected = _replay_gate_fixture(monkeypatch)
    assert dream.replay_gate(shared, alias, helper) == "f" * 64
    assert inspected == ["baseline", "fixed"]


@pytest.mark.parametrize("mutation", [
    "failed_campaign", "networked_replay", "wrong_override", "wrong_phase1", "wrong_inventory",
    "running_stage", "nonzero_exit", "oom", "unchecked_configuration",
    "wrong_baseline_reason", "fixed_no_publication", "changed_fixed_repeat",
])
def test_replay_gate_rejects_every_unproven_prerequisite(monkeypatch, mutation):
    alias, helper, result, receipt, _inspected = _replay_gate_fixture(monkeypatch)
    stage = result["stages"]["fixed"]
    if mutation == "failed_campaign":
        result["status"] = "failed"
    elif mutation == "networked_replay":
        result["networked_runs_started"] = 1
    elif mutation == "wrong_override":
        receipt["canonicalize_sha256"] = "0" * 64
    elif mutation == "wrong_phase1":
        receipt["phase1_sha256"] = "0" * 64
    elif mutation == "wrong_inventory":
        receipt["source_files"] = 479
    elif mutation == "running_stage":
        stage["status"] = "running"
    elif mutation == "nonzero_exit":
        stage["exit_code"] = 1
    elif mutation == "oom":
        stage["oom_killed"] = True
    elif mutation == "unchecked_configuration":
        stage["configuration_verified"] = False
    elif mutation == "wrong_baseline_reason":
        result["stages"]["baseline"]["metadata"]["dedup_on"]["first"]["reason_code"] = "sqlite_integrity_rejection"
    elif mutation == "fixed_no_publication":
        stage["metadata"]["dedup_off"]["published_after_first"] = 0
    else:
        stage["metadata"]["dedup_on"]["exact_repeat"]["logical_digest_after"] = "0" * 64
    with pytest.raises(RuntimeError):
        dream.replay_gate(shared, alias, helper)


def test_failed_replay_gate_cannot_create_candidate_or_work(monkeypatch, tmp_path):
    alias, helper, result, _receipt, _inspected = _replay_gate_fixture(monkeypatch)
    result["status"] = "failed"
    work, candidate = tmp_path / "work", tmp_path / "candidate"
    monkeypatch.setattr(dream, "dependencies", lambda: (shared, alias, helper))
    monkeypatch.setattr(dream, "WORK", work)
    monkeypatch.setattr(dream, "CANDIDATE", candidate)
    with pytest.raises(RuntimeError, match="alias_replay_not_completed"):
        dream.remote("remote-install")
    assert not work.exists() and not candidate.exists()


@pytest.mark.parametrize("mode", ["offline", "live"])
def test_container_configuration_keeps_exact_budget_and_mount_limits(mode):
    helper = SimpleNamespace(RUNTIME=Path("/approved/runtime"),
                             RUNTIME_ENV=Path("/approved/runtime-env.json"),
                             IMAGE="sha256:" + "0" * 64)
    command, mounts = dream.configure(helper, mode)
    assert command[command.index("--network") + 1] == ("hermes-net" if mode == "live" else "none")
    assert [dst for _src, dst, rw in mounts if rw] == ["/work"]
    assert (str(dream.REFERENCE), "/reference/source.sqlite", False) in mounts
    assert (str(dream.CANDIDATE), "/candidate", False) in mounts
    assert command[-13:] == [
        mode, "--env", "/run/runtime-env.json", "--phase1-sha256", dream.PHASE1_SHA,
        "--max-http-attempts", "704", "--max-llm-http-attempts", "192",
        "--max-embedding-http-attempts", "512", "--deadline-seconds", "1800",
    ]
    assert "--read-only" in command


@pytest.mark.parametrize("failure", [False, True])
def test_shared_inspector_adapter_always_restores_pinned_dependency(failure):
    original = lambda *_: None
    dependency = SimpleNamespace(configure=original)

    def inspect(*_args):
        assert dependency.configure is not original
        if failure:
            raise RuntimeError("synthetic inspector rejection")
        return {"configuration_verified": True}

    dependency.inspect = inspect
    if failure:
        with pytest.raises(RuntimeError):
            dream.inspect(dependency, None, "a" * 64, "live", [])
    else:
        assert dream.inspect(dependency, None, "a" * 64, "live", [])["configuration_verified"]
    assert dependency.configure is original


def test_supervisor_timeout_stops_single_live_attempt_without_retry(monkeypatch):
    records, starts, stopped = {}, [], []
    helper = SimpleNamespace(RUNTIME=Path("/approved/runtime"),
                             RUNTIME_ENV=Path("/approved/env.json"),
                             IMAGE="sha256:" + "0" * 64)

    def put_json(path, value):
        if path in records:
            raise FileExistsError("intent already exists")
        records[path] = deepcopy(value)

    def run(command, timeout, _code):
        action = command[1]
        if action == "create":
            mode = command[command.index("--network") + 1]
            return (("a" if mode == "none" else "b") * 64).encode()
        cid = command[2]
        if action == "start":
            starts.append(cid)
            return cid.encode()
        if action == "wait":
            if cid == "b" * 64:
                assert timeout == 1860
                raise RuntimeError("synthetic wait timeout")
            return b"0\n"
        assert action == "logs"
        return b"{}"

    helper.put_json, helper.run = put_json, run
    dependency = SimpleNamespace(project=lambda *_: {
        "status": "ready", "cleanup_ok": True, "source_unchanged": True,
        "runtime_generation_verified": True, "source_sha256": dream.REFERENCE_SHA,
        "phase1_sha256": dream.PHASE1_SHA, "completion_calls": 0, "http_attempts": 0,
    })
    monkeypatch.setattr(dream, "installed", lambda *_: {})
    monkeypatch.setattr(dream, "inspect", lambda _s, _h, cid, *a: {
        "status": "exited" if cid in starts else "created", "pid": 0,
        "exit_code": 0, "oom_killed": False,
    })
    monkeypatch.setattr(dream, "stop", lambda _s, _h, cid, *a: stopped.append(cid))
    dream.supervise(dependency, None, helper)
    assert starts == ["a" * 64, "b" * 64]
    assert stopped == ["b" * 64]
    assert records[dream.ROOT / "result.json"]["status"] == "failed"
    assert records[dream.ROOT / "result.json"]["paid_live_runs_started"] == 1
    with pytest.raises(FileExistsError):
        dream.supervise(dependency, None, helper)
    assert starts == ["a" * 64, "b" * 64]
