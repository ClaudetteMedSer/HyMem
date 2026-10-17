"""Call bounds and privacy for the stopped six-unit fact capacity probe."""

from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace
import sys

from hymem.config import HyMemConfig
from tools.diagnostics import fact_capacity_6unit_probe as probe


class _Client:
    def __init__(self):
        self.request_attempts = 0


def _rows():
    return [{"session_id": "private-session", "slice_key": f"key-{n}"}
            for n in range(1, 7)]


def test_probe_checks_all_six_units_at_reviewed_cap(monkeypatch):
    client = _Client()
    calls = []
    rows = _rows()
    monkeypatch.setattr(
        probe.facts, "next_fact_outcome_for_replay",
        lambda *_args: rows[0]["slice_key"],
    )

    def extraction(_conn, key, llm, cfg, **_kwargs):
        llm.request_attempts += 1
        calls.append((key, cfg.dream_max_facts_per_session,
                      cfg.facts_extraction_max_tokens,
                      probe.facts.build_facts_request("synthetic test", cfg).max_tokens,
                      cfg.dream_digest_max_tokens))
        return SimpleNamespace(
            parse_failed=False, failure_reason=None,
            items=[{}] * (40 if key == "key-4" else 8),
        )

    monkeypatch.setattr(probe.facts, "reextract_fact_outcome", extraction)
    results, stopped = probe._probe_units(
        object(), rows, client, HyMemConfig(root=Path("/tmp")),
        target32="current",
    )
    assert stopped is None
    assert len(results) == 6
    assert client.request_attempts == 6
    assert calls == [(f"key-{n}", 64, 6144, 6144, 3072) for n in range(1, 7)]
    assert results[3]["items"] == 40


def test_probe_stops_on_first_capacity_rejection(monkeypatch):
    client = _Client()
    rows = _rows()
    monkeypatch.setattr(
        probe.facts, "next_fact_outcome_for_replay",
        lambda *_args: rows[0]["slice_key"],
    )

    def overflow(_conn, _key, llm, _cfg, **_kwargs):
        llm.request_attempts += 1
        return SimpleNamespace(
            parse_failed=True, failure_reason="output_capacity_exceeded",
            items=[],
        )

    monkeypatch.setattr(probe.facts, "reextract_fact_outcome", overflow)
    results, stopped = probe._probe_units(
        object(), rows, client, HyMemConfig(root=Path("/tmp")),
        target32="current",
    )
    assert stopped == "cap 64 did not validate the complete unit"
    assert [(row["unit"], row["cap"]) for row in results] == [(1, 64)]
    assert client.request_attempts == 1


def test_provider_exception_is_closed_and_stops_without_private_text(monkeypatch):
    client = _Client()
    rows = _rows()
    monkeypatch.setattr(
        probe.facts, "next_fact_outcome_for_replay",
        lambda *_args: rows[0]["slice_key"],
    )

    def private_failure(_conn, _key, llm, _cfg, **_kwargs):
        llm.request_attempts += 1
        raise RuntimeError("PRIVATE_SOURCE_AND_PROVIDER_RESPONSE")

    monkeypatch.setattr(probe.facts, "reextract_fact_outcome", private_failure)
    results, stopped = probe._probe_units(
        object(), rows, client, HyMemConfig(root=Path("/tmp")),
        target32="current",
    )
    assert stopped == "cap 64 did not validate the complete unit"
    assert len(results) == 1
    assert results[0]["reason"] == "provider_or_source_error"
    assert "PRIVATE" not in str((results, stopped))


def test_wrong_fact_request_token_cap_stops_before_provider(monkeypatch):
    client = _Client()
    monkeypatch.setattr(
        probe.facts, "build_facts_request",
        lambda *_args: SimpleNamespace(max_tokens=3072),
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider called under the wrong token cap")

    monkeypatch.setattr(probe.facts, "reextract_fact_outcome", forbidden)
    try:
        probe._probe_units(
            object(), _rows(), client, HyMemConfig(root=Path("/tmp")),
            target32=probe.REVIEWED_CAP32_TARGET,
        )
    except probe.replay.RepairStopped as exc:
        assert str(exc) == "fact-specific request token cap is not 6144"
    else:
        raise AssertionError("wrong fact request token cap was accepted")
    assert client.request_attempts == 0


def test_mismatched_state_guard_prevents_all_provider_calls(monkeypatch, tmp_path):
    db_path = tmp_path / "facts.sqlite"
    db_path.touch()
    client = _Client()
    client.close = lambda: None

    class _Cursor:
        def fetchone(self):
            return None

    class _Connection:
        def execute(self, _sql):
            return _Cursor()

    @contextmanager
    def snapshot(_path):
        yield _Connection()

    monkeypatch.setattr(probe.replay, "_is_live_store", lambda _path: False)
    monkeypatch.setattr(probe.replay, "_client_from_env", lambda: client)
    monkeypatch.setattr(probe.replay, "_configuration", lambda *_args: HyMemConfig(root=tmp_path))
    monkeypatch.setattr(probe.replay, "_state", lambda *_args: {
        "schema": 64, "state_sha256": "actual-state", "stale": 77,
        "outcomes": 128, "sessions": 49, "pending_fresh": 1,
        "quarantined": 0,
    })
    monkeypatch.setattr(probe.core_db, "read_snapshot", snapshot)
    monkeypatch.setattr(
        probe.facts, "facts_config_version",
        lambda *_a, **_kw: probe.REVIEWED_CAP64_TARGET,
    )
    monkeypatch.setattr(probe, "_units", lambda *_args: (
        "session-key", _rows(), "unit-seal", [{}] * 6,
    ))

    def forbidden(*_args, **_kwargs):
        raise AssertionError("provider work occurred before the exact state guard")

    monkeypatch.setattr(probe, "_probe_units", forbidden)
    monkeypatch.setattr(sys, "argv", [
        "fact_capacity_6unit_probe.py", "--db", str(db_path),
        "--probe", "--allow-provider-calls",
        "--expect-state-sha256", "wrong-state",
        "--expect-target32", probe.REVIEWED_CAP32_TARGET,
        "--expect-target64", probe.REVIEWED_CAP64_TARGET,
        "--expect-session-key", "session-key",
        "--expect-units-sha256", "unit-seal",
        "--expect-stale", "77",
    ])
    output = io.StringIO()
    with redirect_stdout(output):
        assert probe.main() == 2
    assert "guarded fact state changed" in output.getvalue()
    assert client.request_attempts == 0
