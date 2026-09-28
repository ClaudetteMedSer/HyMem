"""Offline warning-classifier checks; no Codex process or model call."""
import importlib.util
import json
from pathlib import Path
import queue
import stat


SOURCE = Path(__file__).resolve().parents[1] / "luna_subscription_warning_probe.py"
SPEC = importlib.util.spec_from_file_location("luna_warning_subject", SOURCE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def test_exact_service_warning_and_near_misses():
    message = "Configured service tier `default` is not advertised as supported for model `gpt-6-luna` and will be omitted from requests."
    assert probe.classify(message, "gpt-6-luna") == ("service_tier_omitted", True)
    assert probe.classify(message.replace("`default`", "`priority`"), "gpt-6-luna") == ("unknown_warning", False)
    assert probe.classify(message.replace("`gpt-6-luna`", "`gpt-5.6-luna`"), "gpt-6-luna") == ("unknown_warning", False)
    assert probe.classify(message + " extra", "gpt-6-luna") == ("unknown_warning", False)


def test_exact_unstable_warning_and_near_misses():
    message = ("Under-development features enabled: skip_host_skill_discovery. "
        "Under-development features are incomplete and may behave unpredictably. "
        "To suppress this warning, set `suppress_unstable_features_warning = true` "
        "in /home/atta/.codex/config.toml.")
    assert probe.classify(message, "gpt-6-luna") == ("unstable_feature_notice", True)
    assert probe.classify(message.replace("skip_host_skill_discovery", "plugins"), "gpt-6-luna") == ("unknown_warning", False)
    assert probe.classify(message.replace("/home/atta/", "../"), "gpt-6-luna") == ("unknown_warning", False)
    assert probe.classify(message.replace("config.toml", "auth.json"), "gpt-6-luna") == ("unknown_warning", False)


def test_preflight_capture_and_drain_private_without_turn(tmp_path):
    warning = "private unknown warning text"

    class Process:
        def __init__(self):
            self.closed = False

        def poll(self):
            return 0 if self.closed else None

    class Session:
        def __init__(self, binary, cwd):
            self.process = Process()
            self.events = queue.Queue()
            self.events.put({"method": "warning", "params": {"threadId": "thr", "message": warning}})

        def receive(self):
            raise AssertionError("no RPC read expected")

        def close(self):
            self.process.closed = True

    class Transport:
        MODEL = "gpt-6-luna"
        StdioSession = Session
        SubscriptionTransportError = RuntimeError

        @staticmethod
        def inspect_preflight(session, **kwargs):
            assert session.process.poll() is None
            return {"config_isolation_admitted": True}

    evidence_dir = tmp_path / "evidence"
    report = probe.run_preflight(Transport, "/unused", evidence_dir, drain_seconds=0.01)
    assert report["ok"] and report["observed_turns"] == 0 and report["cleanup"]
    assert report["warnings"][0]["category"] == "unknown_warning"
    assert warning not in str(report)
    evidence = evidence_dir / "warning.json"
    assert json.loads(evidence.read_text())[0]["message"] == warning
    assert stat.S_IMODE(evidence.stat().st_mode) == 0o600
    assert stat.S_IMODE(evidence_dir.stat().st_mode) == 0o700


def test_explicit_turn_mode_makes_one_call_and_keeps_warning_private(tmp_path):
    warning = "private second warning about sandbox network"
    calls = []

    class Process:
        closed = False

        def poll(self):
            return 0 if self.closed else None

    class Session:
        def __init__(self, binary, cwd):
            self.process = Process()
            self.events = queue.Queue()

        def receive(self):
            return {"method": "warning", "params": {"threadId": "thr", "message": warning}}

        def close(self):
            self.process.closed = True

    class Transport:
        MODEL = "gpt-6-luna"
        StdioSession = Session
        SubscriptionTransportError = RuntimeError

        @staticmethod
        def inspect_preflight(session, **kwargs):
            return {"config_isolation_admitted": True, "_thread_id": "thr"}

        @staticmethod
        def _run_turn(session, thread_id, user):
            calls.append((thread_id, user))
            session.receive()
            raise RuntimeError("warning_unapproved")

    evidence_dir = tmp_path / "evidence"
    report = probe.run_preflight(Transport, "/unused", evidence_dir,
                                 drain_seconds=0, capture_turn=True)
    assert len(calls) == 1
    assert calls[0] == ("thr", 'Return exactly {"ok":true}.')
    assert report["observed_turns"] == 1 and report["known_tokens"] is None
    assert report["usage_complete"] is False and report["cleanup"] is True
    assert report["warnings"][0]["keywords"]["sandbox"] is True
    assert report["warnings"][0]["keywords"]["network"] is True
    assert report["warnings"][0]["keywords"]["model"] is False
    assert warning not in str(report)
    assert json.loads((evidence_dir / "warning.json").read_text())[0]["message"] == warning
