"""Offline tests for the one-shot, private subscription canary runner."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_subscription_canary as runner


def request(number=1):
    return SimpleNamespace(system="system", user=f"user {number}",
                           response_format="json", max_tokens=123, temperature=0)


class FakeDelegate:
    def __init__(self, output: Path, *, fail=False):
        self.output = output
        self.fail = fail
        self.calls = 0
        self.observed_turns = 0
        self.observed_tokens = 0
        self.usage_complete = True

    def complete(self, value):
        progress = json.loads((self.output / "private-progress.json").read_text())
        assert progress["in_flight"] is True
        assert progress["attempted_calls"] == self.calls + 1
        entries = [json.loads(line) for line in
                   (self.output / "private-invocations.jsonl").read_text().splitlines()]
        assert entries[-1]["event"] == "request"
        assert entries[-1]["request"]["user"] == value.user
        self.calls += 1
        self.observed_turns += 1
        if self.fail:
            self.observed_tokens = None
            self.usage_complete = False
            raise RuntimeError("private model failure")
        self.observed_tokens += 5
        return '{"ok":true}'


def test_request_is_durable_before_invocation_and_response_after(tmp_path):
    delegate = FakeDelegate(tmp_path)
    client = runner.BoundedClient(delegate, tmp_path, float("inf"),
                                  lambda phase, in_flight, client: runner._private_json(
                                      tmp_path / "private-progress.json",
                                      {"phase": phase, "in_flight": in_flight,
                                       "attempted_calls": client.attempts}))
    assert client.complete(request()) == '{"ok":true}'
    assert client.in_flight is False
    entries = [json.loads(line) for line in
               (tmp_path / "private-invocations.jsonl").read_text().splitlines()]
    assert [entry["event"] for entry in entries] == ["request", "response"]
    assert entries[1]["text"] == '{"ok":true}'
    assert json.loads((tmp_path / "private-progress.json").read_text())["in_flight"] is False
    assert (tmp_path / "private-invocations.jsonl").stat().st_mode & 0o777 == 0o600


def test_failed_invocation_preserves_in_flight_and_unknown_usage(tmp_path):
    delegate = FakeDelegate(tmp_path, fail=True)
    client = runner.BoundedClient(delegate, tmp_path, float("inf"),
                                  lambda phase, in_flight, client: runner._private_json(
                                      tmp_path / "private-progress.json",
                                      {"in_flight": in_flight,
                                       "attempted_calls": client.attempts,
                                       "usage_complete": delegate.usage_complete}))
    with pytest.raises(RuntimeError, match="private model failure"):
        client.complete(request())
    assert client.in_flight is True
    progress = json.loads((tmp_path / "private-progress.json").read_text())
    assert progress["in_flight"] is True
    assert progress["usage_complete"] is False
    assert [json.loads(line)["event"] for line in
            (tmp_path / "private-invocations.jsonl").read_text().splitlines()] == ["request"]


def test_twenty_fifth_call_rejected_without_delegate_invocation(tmp_path):
    delegate = FakeDelegate(tmp_path)
    client = runner.BoundedClient(delegate, tmp_path, float("inf"),
                                  lambda phase, in_flight, client: runner._private_json(
                                      tmp_path / "private-progress.json",
                                      {"in_flight": in_flight,
                                       "attempted_calls": client.attempts}))
    for n in range(24):
        client.complete(request(n))
    with pytest.raises(runner.CanaryStop, match="call_cap"):
        client.complete(request(25))
    assert delegate.calls == client.attempts == 24


def test_session_tracker_closes_inner_when_start_journal_fails(tmp_path, monkeypatch):
    class Process:
        pid = 12345
        ended = False

        def poll(self):
            return 0 if self.ended else None

    class Session:
        process = Process()

        def close(self):
            self.process.ended = True

    session = Session()
    monkeypatch.setattr(runner, "_journal", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    monkeypatch.setattr(runner, "_group_absent", lambda pid: session.process.ended)
    with pytest.raises(OSError, match="disk"):
        runner.SessionTracker(session, tmp_path / "sessions", [])
    assert session.process.ended is True


def test_private_snapshot_replaces_atomically_with_restricted_mode(tmp_path):
    path = tmp_path / "private-progress.json"
    runner._private_json(path, {"in_flight": True})
    runner._private_json(path, {"in_flight": False})
    assert json.loads(path.read_text()) == {"in_flight": False}
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.glob("*.tmp")) == []


def test_wall_budget_is_inside_supervised_service_limit():
    assert runner.WALL_SECONDS == 590
    assert runner.MAX_CALLS == 24
    assert len(runner.PILOT_SHA256) == len(runner.TRANSPORT_SHA256) == 64
