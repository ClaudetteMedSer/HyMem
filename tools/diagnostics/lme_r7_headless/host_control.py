"""Fixed R7 sample-eight host controller; SSH stdin, never production.

Uses the already-reviewed byte-pinned plan builder. Create and start are separate;
live start additionally requires successful target tests and offline preflight.
No credential contents, raw logs or benchmark text are read or printed here.
"""
import hashlib
import json
from pathlib import Path
import re
import stat
import subprocess
import sys
import types

if sys.flags.optimize:
    raise RuntimeError('optimized_execution_forbidden')

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ')
ROOT = BASE / 'r7-sample8-headless-v1'
SOURCE = BASE / 'offline-r7/verification'
PIN = 'c7ad2eb704f25a4c4914bc910869c07d8931643a16b2f7dda5a8ed285d0bfc47'
R7_PIN = '1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8'

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
    r7_raw = read(BASE / 'offline-r7/diag/manifest.json')
    assert hashlib.sha256(r7_raw).hexdigest() == R7_PIN
    r7 = json.loads(r7_raw)
    expected = {}
    for field in ('source_sha256', 'test_sha256', 'auxiliary_sha256'):
        assert not set(expected) & set(r7[field])
        expected.update(r7[field])
    assert len(expected) == 479
    assert manifest['approved_source_manifest_sha256'] == R7_PIN
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

def live_gates(manifest, host):
    target = BASE / 'offline-r7'
    raw = read(target / 'diag/manifest.json')
    assert hashlib.sha256(raw).hexdigest() == R7_PIN
    expected = json.loads(raw)
    tests, supervisor = js(target / 'results/receipt.json'), js(target / 'results/supervisor.json')
    assert tests['manifest_sha256'] == R7_PIN and tests['gate_passed'] is True
    assert tests['selected'] == 1823 and tests['junit_totals'] == {'tests': 1823, 'failures': 0, 'errors': 0, 'skipped': 4}
    assert tests['observed_skip_nodeids'] == sorted(expected['expected_skip_nodeids'])
    assert hashlib.sha256(read(target / 'results/junit.xml')).hexdigest() == tests['junit_sha256']
    assert supervisor['status'] == 'passed' and supervisor['provider_calls'] == 0
    assert supervisor['source_verified_before_and_after'] is True
    assert supervisor['producer_identity_parity'] is True
    assert supervisor['extraction_contract_parity'] is True
    assert set(supervisor['stages']) == {'tests', 'startup-r6', 'startup-r7', 'contract-r6', 'contract-r7'}
    assert all(stage['status'] == 'completed' and stage['returncode'] == 0
               and stage['safe_to_continue'] is True and stage['child_reaped'] is True
               and stage['group_absent_after_reap'] is True and stage['errors'] == []
               for stage in supervisor['stages'].values())
    startup = supervisor['startup_reports']['r7']
    assert startup['status'] == 'passed' and startup['provider_completions'] == 0
    assert startup['provider_http_attempts'] == 0 and startup['cli_checkpoint_handles_closed'] is True
    target_id = js(target / 'results/container-id.json')['container_id']
    assert target_id == 'c44e1402b16054736da47b94c19eaf413cc53328b2b8d21031f7639690327998'
    target_state = json.loads(command(['docker', 'inspect', target_id]))[0]['State']
    assert target_state['Status'] == 'exited' and target_state['ExitCode'] == 0
    assert target_state['OOMKilled'] is False and target_state['Pid'] == 0
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
    assert pre['source_files_verified'] == 479
    for key in ('source_mapping_sha256', 'dataset_sha256', 'source_question_count',
                'source_indices', 'question_ids', 'sample', 'seed', 'sessions', 'messages'):
        assert pre[key] == manifest[key]
    assert pre['raw_data_exported'] is False and pre['benchmark_executed'] is False

def main():
    assert len(sys.argv) in (3, 4)
    action, mode = sys.argv[1:3]
    assert action in ('create', 'start', 'status') and mode in ('preflight', 'live', 'validation')
    assert (action == 'create') == (len(sys.argv) == 3)
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
        cid = sys.argv[3]
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
