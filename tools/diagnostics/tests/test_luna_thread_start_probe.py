"""Offline controls for the one-shot, model-free thread-start probe."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from tools.diagnostics import luna_thread_start_probe as probe


def test_error_publication_is_bounded_and_text_free():
    event = {"id": 7, "error": {"code": -32000,
        "message": "secret-token Invalid params secret-thread",
        "data": {"private": "secret-prompt"}, "trace": "secret-trace"}}
    public = probe._public_error(event)
    assert public == {"response_shape": "error", "error_keys": ["code", "data", "message"],
                      "rpc_error_code": -32000,
                      "message_bytes": len(event["error"]["message"].encode()),
                      "data_bytes": len(json.dumps(event["error"]["data"]).encode()),
                      "message_class": "invalid_request"}
    assert "secret" not in json.dumps(public)
    private = probe._bounded_error(event)
    assert private["message"].startswith("secret-token")
    assert "secret-prompt" in private["data"]
    assert "trace" not in private
    assert probe._public_error({"id": 1, "result": {}}) == {"response_shape": "result"}


def test_limits_never_exceed_campaign_caps():
    fake = SimpleNamespace()
    with pytest.raises(ValueError, match="limits_invalid"):
        probe.run(fake, {"minimal": "x"}, "binary", workers=5)
    with pytest.raises(ValueError, match="limits_invalid"):
        probe.run(fake, {"minimal": "x"}, "binary", starts=129)
    with pytest.raises(ValueError, match="limits_invalid"):
        probe.run(fake, {"minimal": "x"}, "binary", wall_seconds=301)


def test_no_turn_start_and_successful_threads_unsubscribe(monkeypatch):
    calls = []

    class Process:
        def poll(self):
            return 0

    class Session:
        def __init__(self, binary, cwd, timeout):
            self.created_at = time.monotonic()
            self.active_thread = None
            self.retired_threads = set()
            self.stage = "startup"
            self.process = Process()

        def set_deadline(self, value):
            assert value <= time.monotonic() + 121

        def send(self, method, params, **kwargs):
            calls.append(method)
            return 1

        def receive(self):
            return {"id": 1, "result": {}}

        def unsubscribe(self, thread_id):
            assert thread_id == self.active_thread
            self.retired_threads.add(thread_id)
            self.active_thread = None
            calls.append("thread/unsubscribe")

        def close(self):
            calls.append("close")

    def preflight(session, *, base_instructions):
        session.send("initialize", {})
        session.send("thread/start", {"baseInstructions": base_instructions})
        thread_id = f"thread-{len(calls)}"
        session.active_thread = thread_id
        return {"_thread_id": thread_id}

    fake = SimpleNamespace(WarmSession=Session,
        base=SimpleNamespace(inspect_preflight=preflight),
        _safe_code=lambda exc: "fixed_other")
    monkeypatch.setattr(probe, "_cgroup_pids", lambda: None)
    public, private = probe.run(fake, {"minimal": "public"}, "binary",
                                workers=1, starts=4, wall_seconds=20)
    assert public["ok"] is True
    assert public["turn_starts"] == 0
    assert "turn/start" not in calls
    assert calls.count("thread/unsubscribe") == 4
    assert private["errors"] == []


def test_turn_start_guard_is_hard_failure(monkeypatch):
    class Session:
        created_at = time.monotonic()
        active_thread = None
        retired_threads = set()
        stage = "thread/start"
        process = SimpleNamespace(poll=lambda: 0)

        def __init__(self, *args, **kwargs):
            pass

        def set_deadline(self, value):
            pass

        def send(self, method, params, **kwargs):
            return 1

        def close(self):
            pass

    def preflight(session, *, base_instructions):
        session.send("turn/start", {})

    fake = SimpleNamespace(WarmSession=Session,
        base=SimpleNamespace(inspect_preflight=preflight),
        _safe_code=lambda exc: "fixed_other")
    monkeypatch.setattr(probe, "_cgroup_pids", lambda: None)
    public, _ = probe.run(fake, {"minimal": "public"}, "binary",
                          workers=1, starts=1, wall_seconds=20)
    assert public["ok"] is False
    assert public["attempts"][0]["code"] == "turn_start_forbidden"


def test_one_shot_private_remote_root(monkeypatch, tmp_path):
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    # The root validator is covered through a real fixed remote path at launch;
    # the writer contract is independently checked here.
    probe._write_once(root, probe.PRIVATE_NAME, {"errors": []})
    assert (root / probe.PRIVATE_NAME).stat().st_mode & 0o777 == 0o600
    with pytest.raises(FileExistsError):
        probe._write_once(root, probe.PRIVATE_NAME, {"errors": []})


def test_remote_root_accepts_capitalized_hostname_and_rejects_foreign(monkeypatch):
    class Child:
        def exists(self):
            return False

        def is_symlink(self):
            return False

    class Root:
        parent = Path("/home/atta")
        name = ".hymem-luna-thread-start-test"

        def is_absolute(self):
            return True

        def resolve(self):
            return self.parent / self.name

        def is_symlink(self):
            return False

        def is_dir(self):
            return True

        def stat(self):
            return SimpleNamespace(st_mode=0o700, st_uid=1000)

        def __truediv__(self, name):
            return Child()

    monkeypatch.setattr(probe.socket, "gethostname", lambda: "Afrodite.local")
    probe._private_root(Root())
    monkeypatch.setattr(probe.socket, "gethostname", lambda: "otherhost.local")
    with pytest.raises(ValueError, match="private_root_invalid"):
        probe._private_root(Root())


def test_setup_code_never_exposes_arbitrary_exception_text():
    assert probe._setup_code(ValueError("private_root_invalid")) == "private_root_invalid"
    assert probe._setup_code(ValueError("secret provider text")) == "probe_setup_or_output_failure"
    assert probe._setup_code(RuntimeError("secret provider text")) == "probe_setup_or_output_failure"
    assert probe._setup_code(FileExistsError("/private/secret")) == "output_exists"
