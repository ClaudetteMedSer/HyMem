"""Offline synthetic replay against the pinned 510-file candidate."""
import json
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from benchmarks import luna_semantic_canary as semantic_report

REPO = Path(__file__).resolve().parents[3]
CANDIDATE = Path("/private/tmp/hymem-semantic-step2-v2-candidate-20260929")

SCRIPT = r'''
import json, socket, sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
sys.path.insert(0, sys.argv[1])
def deny(*args, **kwargs): raise AssertionError('network forbidden')
socket.socket.connect = deny
socket.create_connection = deny
from benchmarks import extraction_canary as canary
from benchmarks import luna_semantic_canary as semantic
from benchmarks import luna_semantic_stage_accounting as stages
from hymem.extraction import chunk
scenario = sys.argv[3]
class Client:
    def __init__(self):
        self.observed_turns = 0
        self.observed_tokens = 0
        self.usage_complete = True
        self.request_attempts = 0
        self.calls = []
    def complete(self, request):
        self.observed_turns += 1
        self.observed_tokens += 5
        self.request_attempts += 1
        self.calls.append(stages.classify_stack(Path(sys.argv[1])))
        try: wire = json.loads(request.user)
        except ValueError: wire = None
        if isinstance(wire, dict) and 'batch' in wire:
            candidate = wire['batch']['candidates'][0]
            source = wire['batch']['sources'][0]
            index = 0 if candidate['source_message_id'] == canary.EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID else 1
            corrections = {'one': {0}, 'two': {0,1}, 'wrong_recheck': {0}, 'lost_usage': set()}.get(scenario, set())
            recheck = self.calls[-1] == 'grounding_recheck'
            status = 'supported'
            predicate = candidate['predicate']
            if not recheck and index in corrections:
                status = 'replace_predicate'
                predicate = canary._CANARY_EXPECTED_CLAIMS[index][2]
            if scenario == 'unsupported' and index == 0:
                status, predicate = 'unsupported', None
            if scenario == 'wrong_recheck' and recheck:
                status, predicate = 'unsupported', None
            if scenario == 'bad_binding':
                binding = '0' * 64
            else:
                binding = wire['batch_sha256']
            if scenario == 'lost_usage' and index == 1:
                self.usage_complete = False
            quote = canary._TABLE_CLAIM_ROW if index == 0 else canary._PROSE_BOUNDARY_RIGHT
            evidence = [] if status == 'unsupported' else [dict(source_message_id=candidate['source_message_id'],
                region='owned', quote=quote)]
            return json.dumps(dict(schema='source-grounding-v1', batch_sha256=binding, complete=True,
                verdicts=[dict(index=0, status=status, predicate=predicate, evidence=evidence)]))
        payloads, failures = canary._request_source_payloads(request)
        assert not failures
        claims = []
        if 'OMISSION VERIFICATION PASS' not in request.system:
            for i, expected in enumerate(canary._CANARY_EXPECTED_CLAIMS):
                payload = next((p for p in payloads if p['source_message_id'] == expected[6]), None)
                trigger = canary._TABLE_CLAIM_ROW if i == 0 else canary._PROSE_BOUNDARY_RIGHT
                if payload and trigger in payload['content']:
                    predicate = expected[2]
                    if scenario in {'one', 'two', 'wrong_recheck'} and (i == 0 or scenario == 'two'):
                        predicate = 'uses' if i == 0 else 'uses'
                    claim = dict(subject=expected[0], predicate=predicate, object=expected[3],
                        polarity=expected[5], source_message_id=expected[6],
                        subject_type=expected[1], object_type=expected[4])
                    if scenario == 'wrong_type' and i == 0:
                        claim['subject_type'] = 'person'
                    if scenario == 'wrong_subject' and i == 0:
                        claim['subject'] = 'Invented Subject'
                    if scenario == 'wrong_object' and i == 0:
                        claim['object'] = 'Invented Object'
                    if scenario == 'qualifier' and i == 0:
                        claim['value_text'] = 'unwarranted qualifier'
                    if scenario == 'wrong_source' and i == 0:
                        claim['source_message_id'] = canary.EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID
                    claims.append(claim)
                    if scenario == 'raw_duplicate' and i == 0:
                        claims.append(dict(claim))
                    if scenario == 'extra' and i == 0:
                        claims.append(dict(subject='Invented Entity', predicate='uses', object='Redis',
                            polarity=1, source_message_id=expected[6]))
        markers = [dict(kind='preference', statement='Invented marker')] if scenario == 'marker' and claims else []
        return json.dumps(dict(complete=True, triples=claims, markers=markers))
client = Client()
ledger = stages.StageLedger(Path(sys.argv[1]))
profiled = ledger.wrap(client, 'canary')
report = semantic.run_canary(canary, chunk, profiled, candidate=Path(sys.argv[1]))
print(json.dumps(dict(report=report, stages=client.calls,
    stage_reconciled=ledger.reconcile_canary(report),
    budget_reconciled=ledger.reconcile(dict(turns=client.observed_turns,
        known_tokens=client.observed_tokens, questions={'canary': dict(
            turns=client.observed_turns, known_tokens=client.observed_tokens)})))))
'''


def replay(scenario):
    result = subprocess.run([sys.executable, "-I", "-B", "-c", SCRIPT,
                             str(CANDIDATE), str(REPO), scenario],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("scenario,calls,rechecks", [
    ("healthy", 10, 0), ("one", 11, 1), ("two", 12, 2),
])
def test_pass_counts_and_stage_classification(scenario, calls, rechecks):
    output = replay(scenario)
    report = output["report"]
    assert report["passed"]
    assert report["completion_calls"] == calls
    assert report["grounding_initial_calls"] == 2
    assert report["grounding_recheck_calls"] == rechecks
    assert output["stage_reconciled"] and output["budget_reconciled"]
    assert output["stages"].count("grounding_initial") == 2
    assert output["stages"].count("grounding_recheck") == rechecks
    assert output["stages"].count("extraction_primary_or_other") == 4
    assert output["stages"].count("extraction_empty_verifier") == 2
    assert output["stages"].count("extraction_omission_verifier") == 2
    assert "unclassified" not in output["stages"]


@pytest.mark.parametrize("scenario", ["unsupported", "wrong_recheck", "bad_binding", "lost_usage",
                                      "extra", "wrong_type", "qualifier", "wrong_source", "marker",
                                      "raw_duplicate", "wrong_subject", "wrong_object"])
def test_rejected_grounding_and_usage(scenario):
    assert not replay(scenario)["report"]["passed"]


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(completion_calls=10),
    lambda r: r.update(grounding_initial_calls=1),
    lambda r: r.update(grounding_recheck_calls=0),
    lambda r: r.update(observed_turn_delta=10),
    lambda r: r.update(usage_complete=False),
    lambda r: r.update(corrected_claim_indexes=[]),
    lambda r: r["grounding_trace"][1].update(status="replace_predicate"),
    lambda r: r["grounding_trace"][1].update(batch_sha256=r["grounding_trace"][0]["batch_sha256"]),
    lambda r: r["grounding_trace"][1].update(recheck=False),
    lambda r: r.update(corrected_claim_indexes=[[0]]),
    lambda r: r["grounding_trace"][0].update(status=[]),
])
def test_forged_or_malformed_pass_is_rejected(mutation):
    report = copy.deepcopy(replay("one")["report"])
    mutation(report)
    with pytest.raises(ValueError):
        semantic_report.validate_report(report, fixture_sha256=report["fixture_sha256"])
