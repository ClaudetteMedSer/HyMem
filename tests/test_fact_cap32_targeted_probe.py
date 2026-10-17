"""Privacy regression for the read-only targeted fact capacity probe."""

from contextlib import contextmanager

from tools.diagnostics import fact_cap32_targeted_probe as probe


def test_close_failure_cannot_emit_private_provider_text(
    tmp_path, monkeypatch, capsys,
):
    db_path = tmp_path / "facts.sqlite"
    db_path.touch()

    class Client:
        def close(self):
            raise RuntimeError("PRIVATE_PROVIDER_RESPONSE")

    class Cursor:
        def fetchone(self):
            return None

    class Snapshot:
        def execute(self, _sql):
            return Cursor()

    @contextmanager
    def read_snapshot(_path):
        yield Snapshot()

    monkeypatch.setattr(probe.replay, "_is_live_store", lambda _path: False)
    monkeypatch.setattr(probe.replay, "_client_from_env", Client)
    monkeypatch.setattr(probe.fact_mod, "facts_config_version", lambda *_a, **_kw: "current")
    monkeypatch.setattr(probe.core_db, "read_snapshot", read_snapshot)
    monkeypatch.setattr(probe.replay, "_state", lambda *_a: {"schema": 64})
    monkeypatch.setattr(
        probe, "_first_stale",
        lambda *_a: (_ for _ in ()).throw(probe.replay.RepairStopped("guard failed")),
    )
    monkeypatch.setattr(
        "sys.argv", ["fact_cap32_targeted_probe.py", "--db", str(db_path)],
    )

    assert probe.main() == 2
    captured = capsys.readouterr()
    assert "guard failed" in captured.out
    assert "PRIVATE_PROVIDER_RESPONSE" not in captured.out + captured.err
    assert captured.err == ""
