"""Independent ownership regression: duplicates must not poison an active run."""
import importlib.util
from pathlib import Path

import pytest


def test_duplicate_active_invocation_cannot_write_the_terminal_receipt(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "lme_r8_headless_continuation.py"
    spec = importlib.util.spec_from_file_location("root_continuation", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.save(tmp_path / "intent.json", {"owner": "the already-running invocation"})
    before = (tmp_path / "intent.json").read_bytes()
    monkeypatch.setattr(helper, "checked_config", lambda cfg: (tmp_path, tmp_path / "target"))

    def unexpected(*args, **kwargs):
        pytest.fail("duplicate invocation reached a dependency or dispatch")

    with pytest.raises(FileExistsError):
        helper.pipeline({}, object(), object(),
                        lambda name, value: helper.save(tmp_path / name, value),
                        waiting=unexpected, sending=unexpected)
    assert (tmp_path / "intent.json").read_bytes() == before
    assert not (tmp_path / "result.json").exists()
    assert {p.name for p in tmp_path.iterdir()} == {"intent.json"}
