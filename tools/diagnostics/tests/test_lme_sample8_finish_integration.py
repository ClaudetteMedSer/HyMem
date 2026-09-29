"""Real checkpoint -> stock validator -> detached finish projection controls.

HYMEM_Q1_VERIFIER_SOURCE must point at the independently frozen R5 tree.
All fixture text is invented; no containers or provider calls are involved.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[3]


def load(path, name):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
    return module


# Reuse the genuine frozen-R5 AtomicCheckpoint/record/finalize/reconcile fixture,
# its before/after source pins, and its network prohibition. Do not invent equal
# raw/archive rows or duplicate the real writer's projection in this test.
fixtures = load(ROOT / 'tools/diagnostics/tests/test_lme_sample8_validator.py',
                'sample8_finish_integration_writer_fixtures')
modules = fixtures.modules
make_case = fixtures.make_case
no_network = fixtures.no_network


@pytest.mark.parametrize('case_options,passed', [
    ({}, True),
    ({'wrong_index': 3}, True),
    ({'degraded_index': 4}, True),
    ({'failed_index': 2}, False),
], ids=['correct', 'ordinary-wrong-answer', 'summary-degradation', 'failed-row'])
def test_real_stock_verdict_survives_finish_projection(modules, make_case, case_options, passed):
    finish = load(ROOT / 'tools/diagnostics/lme_sample8_v1/finish_validation.py',
                  'sample8_finish_integration_current_helper')
    case = make_case(**case_options)
    case['manifest_sha'] = finish.MANIFEST_SHA
    case['supervisor']['manifest_sha256'] = finish.MANIFEST_SHA
    report = modules.v.summarize(**case)
    projected = finish.project_validator(json.dumps(report), 0 if passed else 1)
    assert report['benchmark_completed_without_faults'] is passed
    assert projected['benchmark_completed_without_faults'] is passed
    assert projected['status'] == report['status']
    assert projected['question_ids'] == report['question_ids']
    assert projected['per_question'] == report['per_question']
    assert projected['counts'] == report['counts']
    assert projected['aggregate_paid_usage'] == report['aggregate_paid_usage']
    assert projected['new_provider_calls'] == 0
    assert projected['full_500_readiness_verified'] is False
    assert 'scores' not in projected
    assert 'Invented answer' not in json.dumps(projected)
