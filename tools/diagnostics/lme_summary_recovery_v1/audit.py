"""Credential-free, read-only independent audit of the saved recovery result.

Run with pinned package/source, original, clone and results mounted read-only;
network none. Only bounded counters, hashes and usage are printed.
"""
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import stat
import sys
import types

PIN='6113dd9f513521f789392df934cb6735f0cd5b05f0432b4bad9d641e6b73d6e9'
WORKER_PIN='bd8cc72c9bec26e391632af0f133d226e00f367807120b7a9a395e4dc55bf7a5'
PROTOCOL_PIN='ef2d9057ba50103ef15278342eae88c88307acdf973ba684a58cbc27d2103df9'

def need(value,code):
    if not value: raise RuntimeError(code)

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)

def count(value):
    return type(value) is int and 0<=value<=2**63-1

def checked_accounting(result,bounds):
    """Validate saved accounting independently, without trusting worker output."""
    report=result['recovery']; usage=result['usage']
    counts={'calls','provider_attempts','advanced','published','held','exhausted','remaining'}
    need(type(report) is dict and set(report)==counts|{'provider_attempts_exact'}
         and all(count(report[key]) for key in counts)
         and report['provider_attempts_exact'] is True,'audit_recovery_schema')
    meter_counts={'calls':'calls_available','request_attempts':'request_attempts_available',
                  'successful_responses':'successful_responses_available'}
    scalars={'latency_s':'latency_available','cost_usd':'cost_available'}
    tokens={'prompt_tokens','completion_tokens','total_tokens'}
    fields=set(meter_counts)|set(meter_counts.values())|set(scalars)|set(scalars.values())|tokens|{'token_usage_available'}
    need(type(usage) is dict and set(usage)==fields,'audit_usage_schema')
    for field,flag in {**meter_counts,**scalars,**{key:'token_usage_available' for key in tokens}}.items():
        need(type(usage[flag]) is bool,'audit_usage_availability')
        if usage[flag] is False:
            need(usage[field] is None,'audit_unavailable_usage_precision')
        else:
            value=usage[field]
            valid=(count(value) if field in meter_counts or field in tokens else
                   type(value) in (int,float) and 0<=value<=2**63-1 and math.isfinite(value))
            need(valid,'audit_usage_value')
    need(all(usage[flag] is True for flag in meter_counts.values()),'audit_required_usage_unavailable')
    need(not usage['token_usage_available'] or usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens'],
         'audit_token_reconciliation')
    need(report['calls']<=bounds['max_calls']
         and report['calls']==report['advanced']+report['held']
         and report['published']<=report['advanced'],'audit_recovery_accounting')
    need(report['provider_attempts']==usage['request_attempts']<=bounds['max_calls']*3
         and usage['calls']==usage['successful_responses']<=report['calls']
         and report['calls']<=usage['request_attempts']
         and report['advanced']<=usage['successful_responses'],'audit_provider_accounting')
    return dict(report),{key:usage[key] for key in sorted(fields)}

def checked_projection(result,bounds):
    report,usage=checked_accounting(result,bounds)
    need(type(result['stock_invocations']) is int and result['stock_invocations']==1,'audit_invocation_count')
    need(canonical(result['bounds'])==canonical(bounds),'audit_bounds')
    for field in ('exception_type','client_close_exception_type','cleanup_exception_type','receipt_exception_type'):
        need(result.get(field) is None,'audit_worker_exception')
    health_fields={'summary_degraded_sessions','summary_missing_sessions','malformed_summaries'}
    health={}
    for key in ('health_before','health_after'):
        value=result[key]
        need(type(value) is dict and set(value)==health_fields|{'summary_healthy'}
             and all(count(value[field]) for field in health_fields)
             and type(value['summary_healthy']) is bool
             and value['summary_missing_sessions']<=value['summary_degraded_sessions']
             and value['summary_healthy'] is (value['summary_degraded_sessions']==value['malformed_summaries']==0),
             'audit_summary_health')
        health[key]={field:value[field] for field in sorted(value)}
    changes=result['changes']; fields={'published_sessions','recovered_sessions','private_partial_sessions','held_sessions'}
    need(type(changes) is dict and set(changes)==fields and all(count(changes[key]) for key in fields),
         'audit_changes_schema')
    need(result['status'] in ('recovered_all','honestly_degraded'),'audit_status')
    return {'status':result['status'],**health,'changes':{key:changes[key] for key in sorted(fields)},
            'recovery':report,'usage':usage}

def failure_census(conn,expected_held,max_attempts):
    allowed={'parse_failure','output_truncated','shape_failure',
             'summary_shape_failure','summary_validation_failure','summary_output_cap'}
    result=[]
    for reason,sessions,minimum,maximum in conn.execute(
        'SELECT failure_reason,COUNT(*),MIN(attempts),MAX(attempts) FROM summary_recovery '
        'WHERE failure_reason IS NOT NULL GROUP BY failure_reason ORDER BY failure_reason'):
        need(type(reason) is str and reason in allowed and count(sessions) and sessions>0
             and count(minimum) and count(maximum) and 1<=minimum<=maximum<=max_attempts,
             'audit_private_failure_census')
        result.append({'failure_reason':reason,'sessions':sessions,
                       'minimum_attempts':minimum,'maximum_attempts':maximum})
    need(sum(row['sessions'] for row in result)==expected_held,'audit_private_failure_count')
    return result

def read(path):
    path=Path(path)
    assert path.resolve()==path and stat.S_ISREG(path.lstat().st_mode)
    return path.read_bytes()

def js(path):
    raw=read(path); assert len(raw)<=65536
    return json.loads(raw)

def connect(path):
    file=Path(path)
    # The paid process has exited and closed this checkpointed clone. SQLite
    # removes its empty WAL/SHM on last close; a read-only directory cannot
    # recreate them. Immutable mode is permitted ONLY for this cold clone with
    # neither sidecar present, never for the original WAL-bearing reference.
    cold=(path=='/work/hymem.sqlite' and not Path(path+'-wal').exists()
          and not Path(path+'-shm').exists())
    conn=sqlite3.connect(file.as_uri()+'?mode=ro'+('&immutable=1' if cold else ''),
                         uri=True,isolation_level=None)
    conn.row_factory=sqlite3.Row; conn.execute('PRAGMA query_only=ON')
    conn.execute('PRAGMA temp_store=MEMORY')
    return conn

def main():
    need(__debug__,'audit_requires_assertions')
    raw=read('/diag/protocol.py'); assert hashlib.sha256(raw).hexdigest()==PROTOCOL_PIN
    protocol=types.ModuleType('recovery_readonly_protocol'); protocol.__file__='/diag/protocol.py'
    exec(compile(raw,protocol.__file__,'exec'),protocol.__dict__)
    manifest,worker=protocol.verify(PIN)
    assert manifest['helper_sha256']['worker.py']==WORKER_PIN
    result=js('/results/recovery.json'); preflight=js('/preflight/preflight.json')
    supervisor=js('/results/supervisor.json'); terminal=js('/results/invocation/terminal.json')
    assert supervisor['manifest_sha256']==PIN and supervisor['package_and_source_unchanged'] is True
    assert supervisor['exception_type'] is None and supervisor['outcome']==terminal
    assert terminal['status']=='completed' and type(terminal['returncode']) is int and terminal['returncode']==0
    assert all(terminal[field] is True for field in (
        'safe_to_continue','cleanup_complete','child_reaped','group_absent_after_reap','terminal_receipt_written'))
    assert terminal['errors']==[] and terminal['cleanup_warnings']==[]
    assert result['status'] in ('recovered_all','honestly_degraded') and result['stock_invocations']==1
    assert result['client_closed'] is True and result['lease_and_threads_clean'] is True
    assert result['reference_store_unchanged_verified'] is True and result['source_unchanged_verified'] is True
    assert result['all_non_summary_state_unchanged'] is True and result['bounds']==worker.BOUNDS
    assert result['producer_identity_sha256']==preflight['producer_identity_sha256']
    projection=checked_projection(result,worker.BOUNDS)
    api=worker.import_api(Path('/candidate'))
    cold_clone=not any(Path('/work/hymem.sqlite'+suffix).exists() for suffix in ('-wal','-shm'))
    clone_file_before=hashlib.sha256(read('/work/hymem.sqlite')).hexdigest()
    original=connect('/reference/hymem.sqlite'); clone=connect('/work/hymem.sqlite')
    try:
        worker.integrity(original); worker.integrity(clone)
        before=worker.snapshot(original); after=worker.snapshot(clone)
        assert before['full_sha256']==preflight['clone_before_sha256']==result['clone_before_sha256']
        assert after['full_sha256']==result['clone_after_sha256']
        worker.assert_unchanged(before,after)
        initial={sid:api.classify_summary_state(original,sid) for sid in before['sessions']}
        changes=worker.audit_summary_changes(clone,before,after,initial,result['recovery'],api)
        assert canonical(changes)==canonical(result['changes'])
        assert canonical(api.durable_summary_status(original))==canonical(result['health_before'])==canonical(preflight['health_before'])
        assert canonical(api.durable_summary_status(clone))==canonical(result['health_after'])
        assert result['recovery']['remaining']==result['health_after']['summary_degraded_sessions']
        assert result['recovery']['calls']<=100 and result['usage']['request_attempts']<=300
        assert result['recovery']['provider_attempts_exact'] is True
        assert result['recovery']['provider_attempts']==result['usage']['request_attempts']
        assert result['recovery']['published']==changes['recovered_sessions']
        assert result['status']==('recovered_all' if result['health_after']['summary_healthy'] else 'honestly_degraded')
        private_failures=failure_census(clone,changes['held_sessions'],worker.BOUNDS['max_attempts'])
    finally:
        clone.close(); original.close()
    assert hashlib.sha256(read('/work/hymem.sqlite')).hexdigest()==clone_file_before
    if cold_clone:
        assert not any(Path('/work/hymem.sqlite'+suffix).exists() for suffix in ('-wal','-shm'))
    protocol.verify(PIN)
    print(json.dumps({'schema':'summary-recovery-readonly-audit-v1',
        'manifest_sha256':PIN,'audit_passed':True,'new_provider_calls':0,
        'non_summary_state_unchanged':True,'original_unchanged':True,'process_cleanup_verified':True,
        'result_sha256':hashlib.sha256(read('/results/recovery.json')).hexdigest(),
        'supervisor_sha256':hashlib.sha256(read('/results/supervisor.json')).hexdigest(),
        'private_failure_census':private_failures,
        'semantic_quality_guaranteed':False,**projection},sort_keys=True,allow_nan=False))

if __name__=='__main__': main()
