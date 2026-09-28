"""Root-owned terminal proof tests; no remote access or model calls."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import luna_subscription_lme_progress as reader


@pytest.mark.parametrize('bad_field', [None, 'cleanup', 'tokens', 'correct', 'summary'])
def test_scored_wrong_answer_is_valid_but_bad_evidence_is_not(monkeypatch, tmp_path, bad_field):
    pilot, transport, dataset = [tmp_path / name for name in ('pilot.py','transport.py','data.json')]
    hashes = {pilot: reader.PILOT_SHA256, transport: reader.TRANSPORT_SHA256,
              dataset: reader.DATASET_SHA256}
    monkeypatch.setattr(reader, 'digest', lambda path: hashes[path])
    module = SimpleNamespace(verify_inventory=lambda *args: 508)
    monkeypatch.setattr(reader.importlib.util, 'spec_from_file_location',
        lambda *args: SimpleNamespace(loader=SimpleNamespace(exec_module=lambda m: None)))
    monkeypatch.setattr(reader.importlib.util, 'module_from_spec', lambda spec: module)
    protocol = SimpleNamespace(_validate_versioned_indexing=lambda *a, **kw: True)
    monkeypatch.setitem(reader.sys.modules, 'benchmarks', SimpleNamespace(lme_protocol=protocol))
    monkeypatch.setattr(reader.sys, 'path', list(reader.sys.path))
    indexing = dict(outcome='success', healthy=True, summary_healthy=True)
    row = dict(indexing=indexing, benchmark_failure=None, judge_parse_valid=True,
               judge_error=False, correct=False, private_text='must never export')
    summary = dict(question_completed=True, cleanup_ok=True, usage_complete=True,
        invocation_in_flight=False, observed_turns=17, known_tokens=100000,
        correct=False, benchmark_failure=None, canary=dict(passed=True))
    safe = dict(ok=True, question_completed=True, canary_passed=True,
        observed_turns=17, known_tokens=100000, usage_complete=True,
        invocation_in_flight=False, correct=False)
    if bad_field == 'cleanup':
        summary['cleanup_ok'] = False
    elif bad_field == 'tokens':
        safe['known_tokens'] += 1
    elif bad_field == 'correct':
        safe['correct'] = True
    elif bad_field == 'summary':
        indexing['summary_healthy'] = False
    verdict = reader.terminal_check(safe=safe, result=dict(summary=summary,row=row),row=row,
        pilot_path=pilot,transport_path=transport,candidate=tmp_path,
        inventory_stamp=tmp_path/'stamp',inventory_sha256='0'*64,dataset=dataset)
    assert verdict['validated'] is (bad_field is None)
    assert verdict['correct'] is (False if bad_field is None else None)
    assert 'must never export' not in str(verdict)


def test_token_stop_threshold_is_not_an_accounting_ceiling():
    result = reader.summarize_progress(dict(known_tokens=4000123, observed_turns=600,
        usage_complete=True,invocation_in_flight=False))
    assert result['known_tokens'] == 4000123
