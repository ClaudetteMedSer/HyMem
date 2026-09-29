"""Offline controls of new caps and independent aggregation exception capture."""
import importlib.util
from pathlib import Path
import sys
import types
import pytest

DIAG = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, DIAG / (name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


worker = load("claim_conflict_episode_shadow_dream_instrumented")
host = load("claim_conflict_episode_shadow_dream_host")

# Run the exact inherited privacy, payload, accounting and budget controls
# against this new module; no prior test file or worker byte is modified.
spec = importlib.util.spec_from_file_location(
    "episode_shadow_inherited_tests", DIAG / "tests/test_claim_conflict_instrumented_dream.py")
inherited = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inherited)
inherited.worker = worker


class TestInherited(inherited.InstrumentedDreamTests):
    pass


class Journal:
    def __init__(self, fail=False):
        self.events = []
        self.fail = fail

    def append(self, name, value):
        if self.fail:
            raise OSError("private write failure")
        self.events.append((name, value))


def probe(journal):
    code = probe.__code__
    return worker.Probe(journal, Path("/unused"), completion_code=code,
                        attempt_code=code, embedding_code=code,
                        extraction_code=code, persist_code=code)


def frame(line=2897):
    return types.SimpleNamespace(
        f_code=types.SimpleNamespace(co_filename="/candidate/hymem/dreaming/runner.py"),
        f_lineno=line)


def test_default_caps_and_hashes():
    value = probe(Journal())
    assert (value.max_completions, value.max_llm_attempts,
            value.max_embedding_attempts, value.max_attempts) == (128, 384, 512, 896)
    assert worker.DEFAULT_DEADLINE_SECONDS == 2700
    import hashlib
    for name, pin in (
        ("claim_conflict_episode_shadow_dream.py", host.WORKER_SHA),
        ("claim_conflict_episode_shadow_dream_instrumented.py", host.INSTRUMENTED_SHA),
        ("claim_conflict_episode_shadow_dream_supervisor.py", host.SUPERVISOR_SHA),
    ):
        assert hashlib.sha256((DIAG / name).read_bytes()).hexdigest() == pin


def test_caught_failure_survives_ordinary_saturation():
    journal = Journal()
    value = probe(journal)
    value.exception_events = 256
    value._trace_local(frame(), "exception", (RuntimeError, RuntimeError("private"), None))
    assert value.exception_capture_truncated
    assert value.exception_events == 256
    assert len(journal.events) == 1
    assert journal.events[0][0] == "aggregation-exceptions.jsonl"
    assert journal.events[0][1]["type"] == "RuntimeError"


def test_critical_lane_bounded_and_scoped():
    journal = Journal()
    value = probe(journal)
    value.exception_events = 256
    for _ in range(40):
        value._trace_local(frame(), "exception", (RuntimeError, RuntimeError("private"), None))
    assert len(journal.events) == 32
    value._trace_local(frame(200), "exception", (RuntimeError, RuntimeError("private"), None))
    assert len(journal.events) == 32


def test_critical_capture_write_failure_stops_admission():
    value = probe(Journal(fail=True))
    value.exception_events = 256
    value._trace_local(frame(), "exception", (RuntimeError, RuntimeError("private"), None))
    assert value.capture_error
    attempted = types.SimpleNamespace(f_code=value.completion_code, f_locals={})
    with pytest.raises(worker.InstrumentationStop):
        value._profile(attempted, "call", None)


def test_expanded_completion_budget_blocks_129th_call():
    value = probe(Journal())
    value.completions = 128
    attempted = types.SimpleNamespace(f_code=value.completion_code, f_locals={})
    with pytest.raises(worker.BudgetStop):
        value._profile(attempted, "call", None)
    assert value.completions == 128


def test_critical_exception_payload_stays_private(tmp_path, capsys):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    journal = worker.PrivateJournal(root)
    value = probe(journal)
    value.exception_events = 256
    value._trace_local(frame(), "exception",
                       (RuntimeError, RuntimeError("secret-private-marker"), None))
    journal.close()
    assert "secret-private-marker" in (root / "aggregation-exceptions.jsonl").read_text()
    assert "secret-private-marker" not in capsys.readouterr().out


def test_real_private_journal_has_critical_stream(tmp_path):
    root = tmp_path / "journal"
    root.mkdir(mode=0o700)
    journal = worker.PrivateJournal(root)
    value = probe(journal)
    value.exception_events = 256
    value._trace_local(frame(), "exception", (RuntimeError, RuntimeError("private"), None))
    journal.close()
    path = root / "aggregation-exceptions.jsonl"
    assert path.stat().st_mode & 0o777 == 0o600
    import json
    assert json.loads(path.read_text())["type"] == "RuntimeError"


def test_configured_mounts_and_supervisor_caps(monkeypatch):
    monkeypatch.setattr(host, "PARENT", DIAG)
    monkeypatch.setattr(host, "ROOT", DIAG)
    parent = host.configure_parent()
    controller = parent.controller()
    helper = types.SimpleNamespace(RUNTIME=Path("/runtime"),
                                   RUNTIME_ENV=Path("/runtime-env.json"), IMAGE="pinned")
    command, mounts = controller.configure(helper, "live")
    for option, expected in (("--max-http-attempts", "896"),
                             ("--max-llm-http-attempts", "384"),
                             ("--max-embedding-http-attempts", "512"),
                             ("--deadline-seconds", "2700")):
        assert command[command.index(option) + 1] == expected
    assert command[command.index("--network") + 1] == "hermes-net"
    assert "/diag/claim_conflict_instrumented_dream_v1.py" not in str(command)
    assert any(dst == "/diag/claim_conflict_episode_shadow_dream_instrumented.py"
               and rw is False for _, dst, rw in mounts)
    assert all(not rw for _, dst, rw in mounts if dst != "/work")
    assert controller.SELF == DIAG / "claim_conflict_episode_shadow_dream_host.py"


def test_replay_gate_refuses_missing_receipt(monkeypatch, tmp_path):
    monkeypatch.setattr(host, "PARENT", DIAG)
    monkeypatch.setattr(host, "ROOT", DIAG)
    monkeypatch.setattr(host, "REPLAY", tmp_path)
    parent = host.configure_parent()
    with pytest.raises(RuntimeError, match="accepted_replay_pin_drift"):
        parent.proof_gate(None)
    assert parent.suite_gate() is None


def test_projection_accepts_expanded_caps_without_private_payload():
    raw = {"status": "completed", "completion_calls": 128,
           "llm_http_attempts": 384, "embedding_http_attempts": 512,
           "http_attempts": 896, "response": "private", "environment": "private"}
    value = host.project(raw)
    assert value["http_attempts"] == 896
    assert "private" not in str(value)


@pytest.mark.parametrize("key,value", [("completion_calls", 129),
                                      ("llm_http_attempts", 385),
                                      ("embedding_http_attempts", 513),
                                      ("http_attempts", 897)])
def test_projection_refuses_exceeded_caps(key, value):
    with pytest.raises(RuntimeError, match="summary_budget_exceeded"):
        host.project({"status": "completed", key: value})


def test_projection_refuses_accounting_disagreement():
    with pytest.raises(RuntimeError, match="summary_attempt_disagreement"):
        host.project({"status": "completed", "llm_http_attempts": 2,
                      "embedding_http_attempts": 3, "http_attempts": 4})
