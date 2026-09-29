"""Bounded offline controls for independent recovery receipt/SQLite auditing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import types

import pytest

ROOT=Path(__file__).resolve().parents[3]
PATH=ROOT/'tools/diagnostics/lme_summary_recovery_v1/audit.py'
BOUNDS=dict(max_calls=100,max_attempts=3,max_chars=8000,max_tokens=3072,timeout_seconds=1800)


@pytest.fixture
def audit():
    module=types.ModuleType('summary_recovery_readonly_audit_test')
    module.__file__=str(PATH)
    exec(compile(PATH.read_bytes(),str(PATH),'exec'),module.__dict__)
    return module


@pytest.fixture
def receipt():
    health=dict(summary_degraded_sessions=10,summary_missing_sessions=10,
                malformed_summaries=0,summary_healthy=False)
    return dict(status='honestly_degraded',stock_invocations=1,bounds=deepcopy(BOUNDS),
        exception_type=None,client_close_exception_type=None,
        health_before=deepcopy(health),health_after=deepcopy(health),
        changes=dict(published_sessions=0,recovered_sessions=0,private_partial_sessions=0,held_sessions=10),
        recovery=dict(calls=10,provider_attempts=10,provider_attempts_exact=True,
                      advanced=0,published=0,held=10,exhausted=0,remaining=10),
        usage=dict(calls=10,calls_available=True,request_attempts=10,request_attempts_available=True,
                   successful_responses=10,successful_responses_available=True,
                   prompt_tokens=25000,completion_tokens=1164,total_tokens=26164,
                   token_usage_available=True,latency_s=1.0,latency_available=True,
                   cost_usd=None,cost_available=False))


def test_live_result_shape_stays_degraded_and_unknown_cost_stays_unknown(audit,receipt):
    before=json.dumps(receipt,sort_keys=True)
    projected=audit.checked_projection(receipt,BOUNDS)
    assert projected['status']=='honestly_degraded'
    assert projected['recovery']['published']==0 and projected['usage']['total_tokens']==26164
    assert projected['usage']['cost_usd'] is None and projected['usage']['cost_available'] is False
    assert set(projected)=={'status','health_before','health_after','changes','recovery','usage'}
    assert json.dumps(receipt,sort_keys=True)==before


def test_recovered_result_and_zero_usage_are_valid(audit,receipt):
    receipt['status']='recovered_all'
    receipt['health_after'].update(summary_degraded_sessions=0,summary_missing_sessions=0,summary_healthy=True)
    receipt['recovery'].update(advanced=10,published=10,held=0,remaining=0)
    receipt['changes'].update(published_sessions=10,recovered_sessions=10,held_sessions=0)
    assert audit.checked_projection(receipt,BOUNDS)['status']=='recovered_all'


@pytest.mark.parametrize('field',['calls','provider_attempts','advanced','published','held','exhausted','remaining'])
@pytest.mark.parametrize('bad',[-1,True,1.0,2**63])
def test_every_recovery_count_has_strict_nonnegative_integer_type(audit,receipt,field,bad):
    receipt['recovery'][field]=bad
    with pytest.raises(RuntimeError,match='audit_recovery_schema'):
        audit.checked_projection(receipt,BOUNDS)


@pytest.mark.parametrize('mutation',['negative_matched_attempts','disagree_attempts','too_many_attempts',
    'too_many_calls','outcome_sum','published_without_advance','admitted_disagree','admitted_excess',
    'attempts_below_admitted','inexact','report_extra','usage_extra','usage_missing','bool_flag',
    'required_unknown','tokens_sum','hidden_cost','invocation_bool','bounds_bool','exception'])
def test_malformed_accounting_cannot_be_independently_blessed(audit,receipt,mutation):
    report,usage=receipt['recovery'],receipt['usage']
    if mutation=='negative_matched_attempts': report['provider_attempts']=usage['request_attempts']=-1
    elif mutation=='disagree_attempts': report['provider_attempts']=11
    elif mutation=='too_many_attempts': report['provider_attempts']=usage['request_attempts']=301
    elif mutation=='too_many_calls': report.update(calls=101,held=101)
    elif mutation=='outcome_sum': report['advanced']=1
    elif mutation=='published_without_advance': report['published']=1
    elif mutation=='admitted_disagree': usage['successful_responses']=9
    elif mutation=='admitted_excess': usage['calls']=usage['successful_responses']=11
    elif mutation=='attempts_below_admitted': report['provider_attempts']=usage['request_attempts']=9
    elif mutation=='inexact': report['provider_attempts_exact']=False
    elif mutation=='report_extra': report['private_text']='MUST NOT ESCAPE'
    elif mutation=='usage_extra': usage['private_text']='MUST NOT ESCAPE'
    elif mutation=='usage_missing': del usage['cost_available']
    elif mutation=='bool_flag': usage['token_usage_available']=1
    elif mutation=='required_unknown': usage.update(calls=None,calls_available=False)
    elif mutation=='tokens_sum': usage['total_tokens']+=1
    elif mutation=='hidden_cost': usage['cost_usd']=1
    elif mutation=='invocation_bool': receipt['stock_invocations']=True
    elif mutation=='bounds_bool': receipt['bounds']['max_attempts']=3.0
    elif mutation=='exception': receipt['client_close_exception_type']='RuntimeError'
    with pytest.raises(RuntimeError,match='audit_'):
        audit.checked_projection(receipt,BOUNDS)


@pytest.mark.parametrize('field',['calls','request_attempts','successful_responses','prompt_tokens','completion_tokens','total_tokens'])
@pytest.mark.parametrize('bad',[-1,True,1.0,float('nan'),float('inf'),'PRIVATE'])
def test_usage_counts_never_accept_text_coercion_or_nonfinite_values(audit,receipt,field,bad):
    receipt['usage'][field]=bad
    with pytest.raises(RuntimeError,match='audit_usage_value'):
        audit.checked_projection(receipt,BOUNDS)


@pytest.mark.parametrize('field,flag',[('latency_s','latency_available'),('cost_usd','cost_available')])
@pytest.mark.parametrize('bad',[-1,True,float('nan'),float('inf'),'PRIVATE'])
def test_optional_numeric_usage_is_finite_nonnegative_and_typed(audit,receipt,field,flag,bad):
    receipt['usage'].update({field:bad,flag:True})
    with pytest.raises(RuntimeError,match='audit_usage_value'):
        audit.checked_projection(receipt,BOUNDS)


def test_unavailable_tokens_and_latency_stay_null(audit,receipt):
    receipt['usage'].update(prompt_tokens=None,completion_tokens=None,total_tokens=None,
                            token_usage_available=False,latency_s=None,latency_available=False)
    projected=audit.checked_projection(receipt,BOUNDS)
    assert projected['usage']['total_tokens'] is None and projected['usage']['latency_s'] is None


def test_typed_truncation_can_hold_ten_paid_calls_without_admitted_responses(audit,receipt):
    receipt['usage'].update(calls=0,successful_responses=0)
    projected=audit.checked_projection(receipt,BOUNDS)
    assert projected['recovery']['calls']==projected['usage']['request_attempts']==10
    assert projected['recovery']['held']==10 and projected['recovery']['advanced']==0


@pytest.mark.parametrize('mutation',['unattempted_calls','unadmitted_advancement'])
def test_exact_accounting_requires_attempted_calls_and_admitted_advancement(audit,receipt,mutation):
    if mutation=='unattempted_calls':
        receipt['recovery']['provider_attempts']=9
        receipt['usage'].update(calls=9,successful_responses=9,request_attempts=9)
    else:
        receipt['recovery'].update(advanced=10,held=0)
        receipt['usage'].update(calls=0,successful_responses=0)
    with pytest.raises(RuntimeError,match='audit_provider_accounting'):
        audit.checked_projection(receipt,BOUNDS)


@pytest.mark.parametrize('mutation',['health_text','health_bool','health_missing_excess','health_claim',
    'changes_extra','changes_bool','changes_negative'])
def test_closed_health_and_changes_projection_rejects_tampering(audit,receipt,mutation):
    if mutation=='health_text': receipt['health_after']['summary_degraded_sessions']='PRIVATE'
    elif mutation=='health_bool': receipt['health_before']['malformed_summaries']=False
    elif mutation=='health_missing_excess': receipt['health_after']['summary_missing_sessions']=11
    elif mutation=='health_claim': receipt['health_after']['summary_healthy']=True
    elif mutation=='changes_extra': receipt['changes']['PRIVATE']='SECRET'
    elif mutation=='changes_bool': receipt['changes']['held_sessions']=True
    elif mutation=='changes_negative': receipt['changes']['recovered_sessions']=-1
    with pytest.raises(RuntimeError,match='audit_'):
        audit.checked_projection(receipt,BOUNDS)


def test_grouped_failure_census_has_only_known_reasons_and_bounded_attempt_counts(audit):
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE summary_recovery (failure_reason, attempts)')
        conn.executemany('INSERT INTO summary_recovery VALUES (?,?)',[('summary_output_cap',1)]*10)
        conn.execute('INSERT INTO summary_recovery VALUES (NULL,0)')
        assert audit.failure_census(conn,10,3)==[dict(failure_reason='summary_output_cap',sessions=10,
                                                    minimum_attempts=1,maximum_attempts=1)]
        with pytest.raises(RuntimeError,match='audit_private_failure_count'):
            audit.failure_census(conn,9,3)


@pytest.mark.parametrize('reason,attempts',[('PRIVATE',1),('summary_output_cap',0),('summary_output_cap',4),
                                         ('summary_output_cap',1.5),('summary_output_cap','PRIVATE')])
def test_census_cannot_print_unknown_reasons_or_invalid_attempts(audit,reason,attempts):
    with sqlite3.connect(':memory:') as conn:
        conn.execute('CREATE TABLE summary_recovery (failure_reason, attempts)')
        conn.execute('INSERT INTO summary_recovery VALUES (?,?)',(reason,attempts))
        with pytest.raises(RuntimeError,match='audit_private_failure_census'):
            audit.failure_census(conn,1,3)


def mapped_paths(audit,monkeypatch,tmp_path):
    def mapped(value):
        text=str(value)
        for name in ('work','reference'):
            prefix='/'+name+'/'
            if text.startswith(prefix):return tmp_path/name/text[len(prefix):]
        return Path(value)
    monkeypatch.setattr(audit,'Path',mapped)


def test_cold_clone_is_immutable_readonly_and_does_not_create_sidecars(audit,monkeypatch,tmp_path):
    root=tmp_path/'work';root.mkdir();file=root/'hymem.sqlite'
    with sqlite3.connect(file) as conn:
        conn.execute('PRAGMA journal_mode=WAL');conn.execute('CREATE TABLE t (v)');conn.execute('INSERT INTO t VALUES (1)')
    # Ensure the authoring connection is closed, rather than only committed.
    conn.close()
    assert not any(Path(str(file)+suffix).exists() for suffix in ('-wal','-shm'))
    before=hashlib.sha256(file.read_bytes()).hexdigest()
    mapped_paths(audit,monkeypatch,tmp_path)
    reader=audit.connect('/work/hymem.sqlite')
    try:
        assert reader.execute('SELECT v FROM t').fetchone()[0]==1
        with pytest.raises(sqlite3.OperationalError):reader.execute('INSERT INTO t VALUES (2)')
    finally:reader.close()
    assert hashlib.sha256(file.read_bytes()).hexdigest()==before
    assert not any(Path(str(file)+suffix).exists() for suffix in ('-wal','-shm'))


def test_reference_reader_sees_live_wal_instead_of_immutable_stale_base(audit,monkeypatch,tmp_path):
    root=tmp_path/'reference';root.mkdir();file=root/'hymem.sqlite'
    writer=sqlite3.connect(file,isolation_level=None)
    try:
        writer.execute('PRAGMA journal_mode=WAL');writer.execute('PRAGMA wal_autocheckpoint=0')
        writer.execute('CREATE TABLE t (v)');writer.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        writer.execute('INSERT INTO t VALUES (7)')
        assert Path(str(file)+'-wal').stat().st_size>0
        mapped_paths(audit,monkeypatch,tmp_path)
        reader=audit.connect('/reference/hymem.sqlite')
        try:
            assert reader.execute('SELECT v FROM t').fetchone()[0]==7
            with pytest.raises(sqlite3.OperationalError):reader.execute('DELETE FROM t')
        finally:reader.close()
    finally:writer.close()


def test_symlink_or_nonregular_receipts_rejected(audit,tmp_path):
    actual=tmp_path/'receipt.json';actual.write_text('{}')
    link=tmp_path/'linked.json';link.symlink_to(actual)
    with pytest.raises(AssertionError):audit.js(link)
    with pytest.raises(AssertionError):audit.js(tmp_path)
