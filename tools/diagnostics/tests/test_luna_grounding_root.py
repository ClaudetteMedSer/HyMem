"""Independent controls: prompt changes must never filter unsupported output."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(not os.environ.get("HYMEM_FROZEN_CANDIDATE"),
                    reason="requires explicit frozen candidate")
def test_actual_canary_still_rejects_preference_to_use_expansion():
    candidate = Path(os.environ["HYMEM_FROZEN_CANDIDATE"]).resolve()
    project = Path(__file__).resolve().parents[3]
    script = r'''
import json, socket, sys
sys.path.insert(0, sys.argv[1])
sys.path.append(sys.argv[2])
from benchmarks import extraction_canary as c
from hymem.extraction import chunk
from tools.diagnostics.luna_subscription_pilot import experimental_canary

def no_network(*args, **kwargs):
    raise AssertionError("network forbidden in offline control")
socket.socket = no_network

class Client:
    def __init__(self, surplus):
        self.surplus = surplus
        self.observed_turns = 0
        self.observed_tokens = 0
        self.usage_complete = True
    def complete(self, request):
        self.observed_turns += 1
        self.observed_tokens += 1
        triples = []
        if "OMISSION VERIFICATION PASS" not in request.system:
            payloads, failures = c._request_source_payloads(request)
            assert not failures
            for payload in payloads:
                content = payload.get("content", "")
                index = None
                if (c._TABLE_CLAIM_ROW in content and
                    payload.get("source_fragment_context") == c._EXPECTED_TABLE_CONTEXT):
                    index = 0
                if (c._PROSE_BOUNDARY_RIGHT in content and
                    payload.get("source_boundary_context") == c._EXPECTED_PROSE_BOUNDARY_CONTEXT):
                    index = 1
                if index is not None:
                    subject, _, predicate, obj, _, polarity, mid = c._CANARY_EXPECTED_CLAIMS[index]
                    item = dict(subject=subject, predicate=predicate, object=obj,
                                polarity=polarity, source_message_id=mid)
                    triples.append(item)
                    if index == 1 and self.surplus:
                        triples.append(dict(item, predicate="uses"))
        return json.dumps(dict(triples=triples, markers=[], complete=True))

good = experimental_canary(c, chunk, Client(False))
bad = experimental_canary(c, chunk, Client(True))
assert good["passed"] is True
assert bad["passed"] is False
for value in (good, bad):
    assert value["matched_core_claims"] == 2
    assert value["core_execution_path_exact"] is True
    assert value["completion_calls"] == 8
print("positive accepted; unsupported additional use rejected; no model calls")
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(candidate), str(project)],
        capture_output=True, text=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
