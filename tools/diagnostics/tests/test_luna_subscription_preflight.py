import json
import sys

import pytest

from tools.diagnostics import luna_subscription_preflight as diagnostic


@pytest.mark.parametrize("metadata,expected", [
    ({"config_isolation_admitted": True, "runtime_probe_verified": False,
      "inference_enabled": False, "observed_turns": 0}, 0),
    ({"config_isolation_admitted": False, "runtime_probe_verified": False,
      "inference_enabled": False, "observed_turns": 0}, 1),
    ({"config_isolation_admitted": True, "runtime_probe_verified": False,
      "inference_enabled": True, "observed_turns": 0}, 1),
    ({"config_isolation_admitted": True, "runtime_probe_verified": False,
      "inference_enabled": False, "observed_turns": 1}, 1),
])
def test_cli_admits_only_no_inference_config_preflight(monkeypatch, capsys, metadata, expected):
    monkeypatch.setattr(sys, "argv", ["preflight", "--binary", "/official/codex"])
    monkeypatch.setattr(diagnostic, "CodexSubscriptionClient",
                        lambda binary: type("Client", (), {"preflight": lambda self: metadata})())
    assert diagnostic.main() == expected
    assert json.loads(capsys.readouterr().out) == metadata
