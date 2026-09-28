"""The final-cycle error controls have stable pytest identities and ordering."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import lme_protocol as protocol


_TARGET = (
    "tests/test_lme_protocol_hardening.py::"
    "test_lme_v3_success_rejects_every_positive_final_cycle_error"
)
_HASH_SEEDS = ("0", "1", "42")
_COLLECTION_MARKER = "LME_COLLECTED_NODE_IDS="
_CHILD = r'''
import json
import sys

socket_attempts = 0

def reject_socket_traffic(event, args):
    global socket_attempts
    if event in {"socket.connect", "socket.sendto"}:
        socket_attempts += 1
        raise PermissionError("Collection-only regression forbids socket traffic")

sys.addaudithook(reject_socket_traffic)
import pytest

class CaptureCollection:
    def pytest_collection_finish(self, session):
        print("LME_COLLECTED_NODE_IDS=" + json.dumps([
            item.nodeid for item in session.items
        ]))

result = pytest.main(sys.argv[1:], plugins=[CaptureCollection()])
print(f"COLLECTION_SOCKET_ATTEMPTS={socket_attempts}")
raise SystemExit(result if socket_attempts == 0 else 1)
'''


@pytest.fixture(scope="module")
def collected_by_seed():
    root = Path(__file__).resolve().parents[1]
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("HYMEM_", "OPENAI_", "DEEPSEEK_"))
        and name not in {"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONPATH"}
    }
    collected = {}
    for seed in _HASH_SEEDS:
        completed = subprocess.run(
            [
                sys.executable, "-c", _CHILD,
                "--collect-only", "-q", "-o", "addopts=", _TARGET,
                "-W", "error::pytest.PytestUnraisableExceptionWarning",
                "-W", "error::pytest.PytestUnhandledThreadExceptionWarning",
            ],
            cwd=root,
            env={**environment, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, (completed.stdout, completed.stderr)
        assert "COLLECTION_SOCKET_ATTEMPTS=0" in completed.stdout
        records = [
            line.removeprefix(_COLLECTION_MARKER)
            for line in completed.stdout.splitlines()
            if line.startswith(_COLLECTION_MARKER)
        ]
        assert len(records) == 1, completed.stdout
        collected[seed] = json.loads(records[0])
    return collected


@pytest.mark.parametrize("seed", _HASH_SEEDS)
def test_final_cycle_failure_collection_keeps_every_field(collected_by_seed, seed):
    expected = {
        f"{_TARGET}[{name}]" for name in protocol._INDEXING_CYCLE_FAILURE_FIELDS
    }
    collected = collected_by_seed[seed]
    assert len(collected) == len(set(collected)) == len(expected)
    assert set(collected) == expected


def test_final_cycle_failure_collection_order_is_hash_seed_independent(collected_by_seed):
    expected = [
        f"{_TARGET}[{name}]" for name in sorted(protocol._INDEXING_CYCLE_FAILURE_FIELDS)
    ]
    assert collected_by_seed == {seed: expected for seed in _HASH_SEEDS}
