"""Root controls against the physical accepted runtime, without inference."""
from pathlib import Path
import subprocess
import sys


def test_frozen_runtime_accepts_wire_for_all_independently_reviewed_labels():
    repo = Path(__file__).resolve().parents[3]
    candidate = Path('/private/tmp/hymem-semantic-step2-v2-candidate-20260929')
    script = r'''
import sys, socket, json
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, sys.argv[1])
sys.path.append(sys.argv[2])
from hymem.extraction import grounding as g
from tools.diagnostics.luna_semantic_cases import cases, suite_sha256
def deny(*a, **k): raise AssertionError('no network')
socket.socket.connect = deny
assert Path(g.__file__).is_relative_to(Path(sys.argv[1]))
assert suite_sha256() == '511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925'
counts = {'supported': 0, 'correction': 0, 'reject': 0}
checked = 0
for case in cases():
    counts[case.category] += 1
    _, batch = g.build_grounding_request(case.triples, case.sources)
    for negative in ('unsupported', 'uncertain'):
        items = []
        for i, label in enumerate(case.expected):
            status = negative if case.category == 'reject' else next(iter(label.statuses))
            assert status in label.statuses
            predicate = (case.triples[i].predicate if status == 'supported' else
                         label.predicate if status == 'replace_predicate' else None)
            items.append(dict(index=i, status=status, predicate=predicate,
                evidence=[dict(source_message_id=case.triples[i].source_message_id, region=r, quote=q)
                          for r,q in label.evidence]))
        response = dict(schema='source-grounding-v1', batch_sha256=batch.batch_sha256,
                        complete=True, verdicts=items)
        review = g.parse_grounding_response(json.dumps(response), batch)
        assert review.all_supported == (case.category == 'supported')
        checked += 1
    if case.category == 'correction':
        corrected = tuple(replace(t, predicate=e.predicate) for t,e in zip(case.triples,case.expected))
        _, fresh = g.build_grounding_request(corrected,case.sources)
        assert fresh.batch_sha256 != batch.batch_sha256
        # A negative verdict is safe rejection, but does NOT count as recovery.
        assert all(e.statuses == frozenset({'replace_predicate'}) for e in case.expected)
assert counts == {'supported':12, 'correction':2, 'reject':10} and checked == 48
table = next(c for c in cases() if c.case_id == 'owned_table_header')
assert table.sources[0].content == '| Juniper | RidgeHost |'
assert 'Currently deployed to' in table.sources[0].contexts[0].content
print('48 wire verdicts checked; no model judgments measured')
'''
    proc = subprocess.run([sys.executable, '-I', '-B', '-c', script, str(candidate), str(repo)],
                          text=True, capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == '48 wire verdicts checked; no model judgments measured'
