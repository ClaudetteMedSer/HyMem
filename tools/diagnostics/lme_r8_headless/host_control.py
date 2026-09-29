"""One isolated R8 sample-eight candidate regression; never production.

Uses the already-reviewed byte-pinned plan builder. Create and start are separate;
Live start requires reviewed final-source evidence and genuine offline preflight.
No credential contents, raw logs or benchmark text are read or printed here.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

if sys.flags.optimize:
    raise RuntimeError('optimized_execution_forbidden')

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky')
ROOT = None
SOURCE = None
PIN = None
GATE_PIN = None
SUMMARY_PREVIOUS_PINS = {
    'manifest_sha256': 'cf341386e467e7879cbabaa8380b761c3e905ec674757d3990842dbf0f81e67c',
    'capture_inventory_sha256': 'fd645967f1596857eeb1621d73bc82d4dd3349991c94a20fd6e35e6a3779b325',
    'request_file': '035-request.json',
    'request_sha256': 'ad07d82fac2f0b65c8fd2415f2bf656dfedb54177396e9ae1536d73b267afb78',
    'request_body_sha256': '807dc2b13fbcfd1fd6940149f4472144926549288a5c3172e62d8916fdc35a36',
    'returned_chars': 505,
}

def read(path):
    assert path.resolve() == path and stat.S_ISREG(path.lstat().st_mode)
    assert path.stat().st_size <= 32 * 1024 * 1024
    return path.read_bytes()

def js(path):
    return json.loads(read(path))

def save(name, value):
    with (ROOT / name).open('x') as out:
        json.dump(value, out, sort_keys=True)

def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and len(result.stdout) <= 1024 * 1024
    return result.stdout.strip()

def definitions():
    raw = read(ROOT / 'bundle/manifest.json')
    assert hashlib.sha256(raw).hexdigest() == PIN
    manifest = json.loads(raw)
    assert hashlib.sha256(read(Path(__file__).resolve())).hexdigest() == manifest['host_controller_sha256']
    expected = manifest['source_sha256']
    source_pin = hashlib.sha256(json.dumps(expected, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert len(expected) == manifest['source_files']
    verify_runtime(manifest)
    assert manifest['approved_source_manifest_sha256'] == source_pin
    assert manifest['source_sha256'] == expected
    assert manifest['remote_run_root'] == str(ROOT) and manifest['remote_source'] == str(SOURCE)
    path = ROOT / 'bundle/q1_stock_run.py'
    code = read(path)
    assert hashlib.sha256(code).hexdigest() == manifest['helper_sha256'][path.name]
    run = types.ModuleType('q1_verified_host_runner')
    run.__file__ = str(path)
    exec(compile(code, str(path), 'exec'), run.__dict__)
    run.PACKAGE, run.SOURCE = ROOT / 'bundle', SOURCE
    run.validate_package(PIN)
    run.verify_source(manifest)
    host = run.load_verified_module(ROOT / 'bundle/q1_stock_host.py',
        manifest['helper_sha256']['q1_stock_host.py'], 'q1_verified_plan')
    return manifest, host

def inspect(cid, plan, expected_state=None):
    assert re.fullmatch('[0-9a-f]{64}', cid)
    obj = json.loads(command(['docker', 'inspect', cid]))[0]
    cfg, hc, state = obj['Config'], obj['HostConfig'], obj['State']
    assert obj['Id'] == cid and obj['Image'] == cfg['Image'] == plan['image']
    assert obj['Name'] == '/' + plan['name'] and cfg['User'] == '1000:1000'
    assert cfg['WorkingDir'] == '/candidate'
    assert cfg['Entrypoint'] == ['/home/node/hymem-env/bin/python3']
    expected_args = plan['command'][plan['command'].index(plan['image']) + 1:]
    assert cfg['Cmd'] == obj['Args'] == expected_args
    assert obj['Path'] == '/home/node/hymem-env/bin/python3'
    assert {m['Destination']: (m['Source'], m['RW'], m['Type']) for m in obj['Mounts']} == {
        target: (origin, writable, 'bind') for target, (origin, writable) in plan['mounts'].items()}
    assert hc['NetworkMode'] == plan['network'] and hc['ReadonlyRootfs'] is True
    assert hc['Privileged'] is False and hc['CapDrop'] == ['ALL']
    assert hc['SecurityOpt'] == ['no-new-privileges'] and hc['Init'] is True
    assert hc['PidsLimit'] == 128 and hc['Memory'] == 2147483648 and hc['NanoCpus'] == 2000000000
    assert hc['Tmpfs'] == {'/tmp': 'rw,noexec,nosuid,size=64m'}
    assert hc['RestartPolicy']['Name'] == 'no'
    assert not any(v.startswith(('DEEPSEEK_', 'OPENAI_', 'HYMEM_')) for v in cfg['Env'])
    if expected_state is not None:
        assert state['Status'] == expected_state
    return {'container_id': cid, 'status': state['Status'], 'exit_code': state['ExitCode'],
        'oom_killed': state['OOMKilled'], 'pid': state['Pid'], 'configuration_verified': True,
        'network': plan['network'], 'manifest_sha256': PIN,
        'credential_mount': '/run/deepseek.env' in plan['mounts']}

def verify_runtime(manifest):
    runtime = manifest['runtime_seal']
    raw = (json.dumps(runtime, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    assert hashlib.sha256(raw).hexdigest() == manifest['runtime_seal_sha256']
    assert runtime['schema'] == 'lme-v64-runtime-seal-v1' and runtime['root_reviewed'] is True
    runtime_root = Path(runtime['runtime_root'])
    assert str(runtime_root) == '/opt/stacks/hermes/instance1/home/hymem-env'
    assert runtime_root.resolve() == runtime_root
    assert runtime['image'] == 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
    checked_runtime_entries(runtime['entries'])
    assert runtime_inventory(runtime_root) == runtime['entries']

def checked_runtime_entries(entries):
    assert type(entries) is dict and entries
    for name, value in entries.items():
        assert type(name) is str and name
        rel = Path(name)
        assert not rel.is_absolute() and '..' not in rel.parts and rel.as_posix() == name
        assert type(value) is dict
        assert all(type(value[k]) is int and value[k] >= 0 for k in ('mode', 'uid', 'gid'))
        assert value['mode'] <= 0o7777
        kinds = set(value) - {'mode', 'uid', 'gid'}
        assert kinds in ({'sha256'}, {'link'}, {'directory'})
        if kinds == {'sha256'}:
            assert type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256'])
        elif kinds == {'link'}:
            assert type(value['link']) is str and value['link'] and '\x00' not in value['link']
        else:
            assert value['directory'] is True

def stable_file_identity(info):
    """Fields a read must preserve; access time can change during hashing."""
    return tuple(getattr(info, field) for field in (
        'st_dev', 'st_ino', 'st_mode', 'st_nlink', 'st_uid', 'st_gid',
        'st_size', 'st_mtime_ns', 'st_ctime_ns'))

def runtime_inventory(root):
    """Exact lstat inventory, preserving link identity without traversing links."""
    assert root.resolve() == root and root.is_dir() and not root.is_symlink()
    result = {}
    def failed(error):
        raise RuntimeError('runtime_traversal_failed')
    for directory, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in sorted(directories + files):
            path = Path(directory) / name
            info = path.lstat()
            value = {'mode': stat.S_IMODE(info.st_mode), 'uid': info.st_uid, 'gid': info.st_gid}
            if stat.S_ISLNK(info.st_mode):
                value['link'] = os.readlink(path)
            elif stat.S_ISDIR(info.st_mode):
                value['directory'] = True
            else:
                assert stat.S_ISREG(info.st_mode)
                digest = hashlib.sha256()
                # Never follow a replacement symlink. Check the opened object
                # against the path snapshot before and after reading, then
                # check that the path still names that same unchanged object.
                fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
                try:
                    assert stable_file_identity(os.fstat(fd)) == stable_file_identity(info)
                    for chunk in iter(lambda: os.read(fd, 1024 * 1024), b''):
                        digest.update(chunk)
                    assert stable_file_identity(os.fstat(fd)) == stable_file_identity(info)
                    assert stable_file_identity(path.lstat()) == stable_file_identity(info)
                finally:
                    os.close(fd)
                value['sha256'] = digest.hexdigest()
            result[path.relative_to(root).as_posix()] = value
    return result

def evidence_record(binding):
    assert set(binding) == {'path', 'sha256'}
    path = Path(binding['path'])
    assert path.is_relative_to(BASE) and re.fullmatch('[0-9a-f]{64}', binding['sha256'])
    raw = read(path)
    assert hashlib.sha256(raw).hexdigest() == binding['sha256']
    return json.loads(raw)

def candidate_evidence(gate, manifest):
    assert set(gate['evidence']) == {'suite', 'retained_extraction', 'summary'}
    for name, binding in gate['evidence'].items():
        assert set(binding) == ({'receipt', 'manifest', 'supervisor', 'audit'}
                               if name == 'retained_extraction' else {'receipt', 'manifest', 'supervisor'})
        receipt = evidence_record(binding['receipt'])
        source = evidence_record(binding['manifest'])
        supervisor = evidence_record(binding['supervisor'])
        assert source['schema'] == {'suite': 'r8-offline-full-suite-v1',
            'retained_extraction': 'r8-retained-chunks-v1',
            'summary': 'r8-summary-repair-verification-package-v2'}[name]
        assert receipt['manifest_sha256'] == binding['manifest']['sha256']
        assert supervisor['manifest_sha256'] == binding['manifest']['sha256']
        outcome = supervisor['outcome']
        assert outcome['returncode'] == 0 and outcome['safe_to_continue'] is True
        assert outcome['child_reaped'] is True and outcome['group_absent_after_reap'] is True
        assert outcome['cleanup_complete'] is True and not outcome['errors']
        assert source['source_sha256'] == manifest['source_sha256']
        if name == 'summary':
            assert receipt['schema'] == 'r8-summary-repair-verification-v2'
            assert receipt['status'] == 'effectiveness_passed_review_pending'
            assert gate['summary_control_review_passed'] is True
            assert gate['repair_control_review_passed'] is True
            assert source['previous_pins'] == receipt['previous_pins_verified'] == SUMMARY_PREVIOUS_PINS
            assert source['repair_control_targets'] == 4
            assert receipt['normal_retained']['complete_sessions'] == 14
            assert receipt['normal_controls']['complete_sessions'] == 4
            assert receipt['explicit_recovery']['recovered_all'] is True
            repair_controls = receipt['repair_contract_controls']
            assert repair_controls['verification_kind'] == 'direct_repair_prompt_contract_not_stock_invocation'
            assert repair_controls['complete_sessions'] == 4 and len(repair_controls['sessions']) == 4
            assert type(repair_controls['synthetic_primary_calls']) is int and repair_controls['synthetic_primary_calls'] == 0
            assert repair_controls['database_unchanged'] is True
            assert all(item['complete'] is True and item['windows']
                       and all(window['failure_reason'] is None for window in item['windows'])
                       for item in repair_controls['sessions'])
            exact_case = receipt['retained_repair_case']
            assert exact_case['verification_kind'] == 'direct_repair_prompt_on_exact_recorded_primary_input'
            assert exact_case['passed'] is True and exact_case['failure_reason'] is None
            assert type(exact_case['completion_calls']) is int and exact_case['completion_calls'] == 1
            assert exact_case['raw_content_exported'] is False
            assert type(exact_case['database_writes']) is int and exact_case['database_writes'] == 0
            assert exact_case['previous_pins'] == SUMMARY_PREVIOUS_PINS
            assert receipt['mode'] == 'live'
            assert all(receipt[k] is True for k in ('clients_closed', 'connections_closed', 'reference_unchanged', 'source_unchanged', 'threads_clean'))
            assert receipt['cleanup_errors'] == []
            assert source['bounds'] == {'completion_calls':192, 'http_attempts':576, 'timeout_seconds':2700}
            paid_accounting(receipt['usage'], receipt['completion_calls'], source['bounds'])
            assert receipt['provider_calls'] == receipt['usage']['calls']
            assert receipt['http_attempts'] == receipt['usage']['request_attempts']
        else:
            assert receipt['source_unchanged'] is True
            if name == 'suite':
                assert receipt['status'] == 'passed' and receipt['tested_source_unchanged'] is True
                assert receipt['full_runs_started'] == 1
                assert receipt['collect']['collected'] == receipt['full']['collected'] == source['expected_collected']
                assert receipt['full']['exit_code'] == receipt['full']['failed'] == receipt['full']['errors'] == 0
                assert type(receipt['full']['passed']) is int and receipt['full']['passed'] >= 0
                assert type(receipt['full']['skipped']) is int and 0 <= receipt['full']['skipped'] <= 4
                assert receipt['full']['passed'] + receipt['full']['skipped'] == source['expected_collected']
                assert receipt['provider_calls'] == 0
            else:
                assert receipt['status'] == 'live_completed' and receipt['extraction_success'] is True
                assert receipt['mode'] == 'live'
                assert receipt['extraction_invocations'] == 2 and receipt['reference_unchanged'] is True
                assert receipt['client_closed'] is True and receipt['threads_clean'] is True
                assert len(receipt['chunks']) == 2 and all(c['extraction']['failed'] is False for c in receipt['chunks'])
                assert [c['chunk_id'] for c in receipt['chunks']] == [
                    'chk_aace7a800d186c4351ce09307040592724ff7239',
                    'chk_57346bc06395cd17a1f1bfd38b6701a3de5e5b8e']
                assert source['bounds'] == {'completion_calls':96, 'http_attempts':288, 'timeout_seconds':1200}
                calls = sum(c['extraction']['completion_calls'] for c in receipt['chunks'])
                attempts = sum(c['extraction']['provider_attempts'] for c in receipt['chunks'])
                assert all(type(c['extraction'][k]) is int and c['extraction'][k] >= 0
                           for c in receipt['chunks'] for k in ('completion_calls', 'provider_attempts'))
                paid_accounting(receipt['usage'], calls, source['bounds'])
                assert attempts == receipt['usage']['request_attempts']
                audit = evidence_record(binding['audit'])
                assert audit['status'] == 'audit_passed' and audit['manifest_sha256'] == binding['manifest']['sha256']
                assert audit['capture_inventory_sha256'] == gate['retained_capture_inventory_sha256']
                assert re.fullmatch('[0-9a-f]{64}', audit['capture_inventory_sha256'])
                assert all(audit[k] is True for k in ('exact_requests_verified', 'private_results_verified',
                    'capture_unchanged', 'reference_unchanged', 'source_unchanged', 'container_removed'))
                assert audit['provider_calls'] == 0 and audit['completion_calls'] == calls
                assert audit['recorded_http_attempts'] == attempts
                assert [{k:c[k] for k in ('chunk_id', 'failed', 'completion_calls', 'provider_attempts')}
                        for c in audit['chunks']] == [dict(chunk_id=c['chunk_id'], **{
                            k:c['extraction'][k] for k in ('failed', 'completion_calls', 'provider_attempts')})
                        for c in receipt['chunks']]

def paid_accounting(usage, calls, bounds):
    assert type(calls) is int and 0 <= calls <= bounds['completion_calls']
    assert all(usage[k] is True for k in ('calls_available', 'request_attempts_available', 'successful_responses_available'))
    assert all(type(usage[k]) is int for k in ('calls', 'request_attempts', 'successful_responses'))
    assert 0 <= usage['calls'] == usage['successful_responses'] <= calls
    assert usage['successful_responses'] <= usage['request_attempts'] <= min(bounds['http_attempts'], calls*3)

def live_gates(manifest, host):
    raw = read(ROOT / 'root-reviewed-candidate-gate.json')
    assert hashlib.sha256(raw).hexdigest() == GATE_PIN
    gate = json.loads(raw)
    assert gate['schema'] == 'lme-r8-root-reviewed-candidate-gate-v1'
    assert gate['root_reviewed'] is True and gate['status'] == 'passed'
    assert gate['source_manifest_sha256'] == manifest['approved_source_manifest_sha256'] and gate['schema_version'] == 64
    assert gate['preparation_manifest_sha256'] == PIN
    assert gate['runtime_seal_sha256'] == manifest['runtime_seal_sha256']
    assert gate['isolated_candidate_regression'] is True and gate['deployment_performed'] is False
    assert gate['production_source_verified'] is False
    candidate_evidence(gate, manifest)
    preflight_id = js(ROOT / 'preflight-container-id.json')['container_id']
    preflight_plan = host.command(str(ROOT), str(SOURCE), PIN, live=False)
    preflight_state = inspect(preflight_id, preflight_plan, 'exited')
    assert preflight_state['exit_code'] == 0 and preflight_state['oom_killed'] is False
    assert preflight_state['pid'] == 0
    pre = js(ROOT / 'preflight-results/preflight.json')
    assert pre['status'] == 'preflight_only' and type(pre['api_calls']) is int and pre['api_calls'] == 0
    probe = pre['startup_probe']
    assert probe['schema'] == 'stock-lme-sample8-pre-provider-startup-v1'
    assert probe['status'] == 'passed' and probe['phase'] == 'complete'
    assert probe['cleanup_failed'] is False and probe['real_credentials_loaded'] is False
    assert probe['expected_question_ids'] == manifest['question_ids']
    assert type(probe['expected_question_count']) is int and probe['expected_question_count'] == 8
    assert all(probe[field] is True for field in (
        'standalone_producer_identity_exact','runtime_transport_identity_exact',
        'stock_cli_pre_provider_boundary_reached','checkpoint_runtime_producer_matches',
        'cli_checkpoint_handles_closed','private_home_created','canonical_extra_body_environment_absent'))
    assert all(type(probe[field]) is int and probe[field] == 0 for field in (
        'provider_completions','provider_http_attempts','provider_successful_responses',
        'outbound_operations_blocked','completed_questions'))
    assert all(type(probe[field]) is int and probe[field] == 1 for field in (
        'runtime_clients_constructed','runtime_clients_closed','cli_checkpoints_observed'))
    assert pre['selection_predeclared'] is True
    assert pre['per_question_census'] == [{field: row[field] for field in (
        'source_index','question_id','sessions','messages')} for row in manifest['questions']]
    assert pre['source_files_verified'] == manifest['source_files']
    for key in ('source_mapping_sha256', 'dataset_sha256', 'source_question_count',
                'source_indices', 'question_ids', 'sample', 'seed', 'sessions', 'messages'):
        assert pre[key] == manifest[key]
    assert pre['raw_data_exported'] is False and pre['benchmark_executed'] is False

def main():
    global PIN, GATE_PIN, ROOT, SOURCE
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('create', 'start', 'status'))
    parser.add_argument('mode', choices=('preflight', 'live', 'validation'))
    parser.add_argument('container_id', nargs='?')
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--candidate-gate-sha256')
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    action, mode = args.action, args.mode
    PIN, GATE_PIN = args.manifest_sha256, args.candidate_gate_sha256
    ROOT = args.root
    assert ROOT.is_absolute() and ROOT.resolve() == ROOT and ROOT.is_relative_to(BASE)
    raw = read(ROOT / 'bundle/manifest.json')
    assert hashlib.sha256(raw).hexdigest() == PIN
    SOURCE = Path(json.loads(raw)['remote_source'])
    assert re.fullmatch('[0-9a-f]{64}', PIN)
    assert (action == 'create') == (args.container_id is None)
    if action == 'start' and mode == 'live':
        assert GATE_PIN is not None and re.fullmatch('[0-9a-f]{64}', GATE_PIN)
    manifest, host = definitions()
    plan = (host.validation_command(str(ROOT), str(SOURCE), PIN) if mode == 'validation'
            else host.command(str(ROOT), str(SOURCE), PIN, live=mode == 'live'))
    if action == 'create':
        save(mode + '-create-intent.json', {'manifest_sha256': PIN, 'plan': plan})
        cid = command(plan['command'])
        save(mode + '-container-id.json', {'container_id': cid})
        result = inspect(cid, plan, 'created')
        save(mode + '-created.json', result)
    else:
        cid = args.container_id
        assert js(ROOT / (mode + '-container-id.json'))['container_id'] == cid
        if action == 'start':
            inspect(cid, plan, 'created')
            if mode == 'validation':
                live_id = js(ROOT / 'live-container-id.json')['container_id']
                live_state = inspect(live_id, host.command(str(ROOT), str(SOURCE), PIN, live=True), 'exited')
                # A failed paid run still requires genuine offline evidence
                # validation; do not hide it because its exit status is nonzero.
                assert live_state['pid'] == 0
            if mode == 'live':
                live_gates(manifest, host)
                key = Path(host.KEY)
                info = key.lstat()
                assert key.resolve() == key and stat.S_ISREG(info.st_mode)
                assert info.st_uid == 1000 and stat.S_IMODE(info.st_mode) == 0o600
            save(mode + '-start-intent.json', {'container_id': cid, 'manifest_sha256': PIN})
            assert command(['docker', 'start', cid]) == cid
        result = inspect(cid, plan)
    print(json.dumps(result, sort_keys=True))

if __name__ == '__main__':
    main()
