"""Pinned R6 regression create/start/status; only the parent invokes live start."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import types

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ')
ROOT = BASE / 'r6-lme-failed-question-v1'
SOURCE = ROOT / 'candidate'
SOURCE_PIN = 'bd8d0f3a8fb40bd6b77e7ca6579c8e5e8ee78733bea2bb22df71bc4b2c12eaa2'


def need(value, code):
    if not value:
        raise RuntimeError('r6_host_' + code)


def read(path):
    need(path.resolve() == path and stat.S_ISREG(path.lstat().st_mode), 'regular_file')
    need(path.stat().st_size <= 32 * 1024 * 1024, 'file_size')
    return path.read_bytes()


def js(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, 'duplicate_json_key')
            result[key] = value
        return result
    return json.loads(read(path), object_pairs_hook=unique,
                      parse_constant=lambda _: need(False, 'nonfinite_json'))


def save(name, value):
    fd = os.open(ROOT / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def command(argv):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=30)
    need(result.returncode == 0, 'docker_command')
    need(len(result.stdout) <= 1024 * 1024, 'docker_output_size')
    return result.stdout.strip()


def definitions(pin):
    need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'manifest_pin_shape')
    raw = read(ROOT / 'bundle/manifest.json')
    need(hashlib.sha256(raw).hexdigest() == pin, 'manifest_pin')
    manifest = js(ROOT / 'bundle/manifest.json')
    raw = read(ROOT / 'input-manifest.json')
    need(hashlib.sha256(raw).hexdigest() == SOURCE_PIN, 'source_manifest_pin')
    source = js(ROOT / 'input-manifest.json')
    need(manifest['approved_source_manifest_sha256'] == SOURCE_PIN
         and manifest['source_sha256'] == source['source_sha256']
         and manifest['remote_run_root'] == str(ROOT)
         and manifest['remote_source'] == str(SOURCE), 'source_binding')
    path = ROOT / 'bundle/q1_stock_run.py'
    code = read(path)
    need(hashlib.sha256(code).hexdigest() == manifest['helper_sha256'][path.name], 'runner_pin')
    run = types.ModuleType('r6_verified_host_runner')
    run.__file__ = str(path)
    exec(compile(code, str(path), 'exec'), run.__dict__)
    run.PACKAGE, run.SOURCE = ROOT / 'bundle', SOURCE
    run.validate_package(pin)
    run.verify_source(manifest)
    host = run.load_verified_module(ROOT / 'bundle/q1_stock_host.py',
                                    manifest['helper_sha256']['q1_stock_host.py'], 'r6_verified_plan')
    return manifest, host


def inspect(cid, plan, pin, expected_state=None):
    need(type(cid) is str and re.fullmatch('[0-9a-f]{64}', cid), 'container_id')
    obj = json.loads(command(['docker', 'inspect', cid]))[0]
    cfg, hc, state = obj['Config'], obj['HostConfig'], obj['State']
    need(obj['Id'] == cid and obj['Image'] == cfg['Image'] == plan['image'], 'image')
    need(obj['Name'] == '/' + plan['name'] and cfg['User'] == '1000:1000', 'name_user')
    need(cfg['WorkingDir'] == '/candidate'
         and cfg['Entrypoint'] == ['/home/node/hymem-env/bin/python3']
         and obj['Path'] == '/home/node/hymem-env/bin/python3', 'entrypoint')
    need(cfg['Cmd'] == obj['Args'] == plan['command'][plan['command'].index(plan['image']) + 1:], 'command')
    need({m['Destination']: (m['Source'], m['RW'], m['Type']) for m in obj['Mounts']} == {
        target: (origin, writable, 'bind') for target, (origin, writable) in plan['mounts'].items()}, 'mounts')
    need(hc['NetworkMode'] == plan['network'] and hc['ReadonlyRootfs'] is True
         and hc['Privileged'] is False and hc['CapDrop'] == ['ALL']
         and hc['SecurityOpt'] == ['no-new-privileges'] and hc['Init'] is True, 'isolation')
    need(hc['PidsLimit'] == 128 and hc['Memory'] == 2147483648 and hc['NanoCpus'] == 2000000000
         and hc['Tmpfs'] == {'/tmp': 'rw,noexec,nosuid,size=64m'}
         and hc['RestartPolicy']['Name'] == 'no', 'limits')
    need(not any(v.startswith(('DEEPSEEK_', 'OPENAI_', 'HYMEM_')) for v in cfg['Env']), 'ambient_credentials')
    if expected_state is not None:
        need(state['Status'] == expected_state, 'container_state')
    return {'container_id': cid, 'status': state['Status'], 'exit_code': state['ExitCode'],
            'oom_killed': state['OOMKilled'], 'pid': state['Pid'], 'configuration_verified': True,
            'network': plan['network'], 'manifest_sha256': pin,
            'credential_mount': '/run/deepseek.env' in plan['mounts']}


def admission(pin):
    need(type(pin) is str and re.fullmatch('[0-9a-f]{64}', pin), 'admission_pin_shape')
    need(hashlib.sha256(read(ROOT / 'admission.json')).hexdigest() == pin, 'admission_pin')
    value = js(ROOT / 'admission.json')
    need(type(value) is dict and set(value) == {
        'schema', 'source_manifest_sha256', 'all_three_fixes_parent_accepted',
        'full_offline_gate_passed', 'semantic_controls_parent_accepted', 'production_changes', 'receipts'},
        'admission_fields')
    need(value['schema'] == 'r6-parent-regression-admission-v1'
         and value['source_manifest_sha256'] == SOURCE_PIN
         and value['production_changes'] is False
         and all(value[field] is True for field in ('all_three_fixes_parent_accepted',
             'full_offline_gate_passed', 'semantic_controls_parent_accepted')), 'admission_verdict')
    receipts = value['receipts']
    need(type(receipts) is dict and 4 <= len(receipts) <= 32, 'admission_receipts')
    for name, digest in receipts.items():
        need(type(name) is str, 'receipt_name')
        path = PurePosixPath(name)
        need(path.as_posix() == name and not path.is_absolute() and '..' not in path.parts
             and len(path.parts) >= 2 and path.parts[0] == 'gates'
             and all(not part.startswith('.') for part in path.parts)
             and type(digest) is str and re.fullmatch('[0-9a-f]{64}', digest), 'receipt_path')
        need(hashlib.sha256(read(ROOT / name)).hexdigest() == digest, 'receipt_pin')
    return value


def live_gates(manifest, host, pin, admission_pin):
    admission(admission_pin)
    cid = js(ROOT / 'preflight-container-id.json')['container_id']
    state = inspect(cid, host.command(str(ROOT), str(SOURCE), pin), pin, 'exited')
    need(state['exit_code'] == 0 and state['oom_killed'] is False and state['pid'] == 0, 'preflight_exit')
    pre = js(ROOT / 'preflight-results/preflight.json')
    need(pre['status'] == 'preflight_only' and type(pre['api_calls']) is int and pre['api_calls'] == 0
         and pre['raw_data_exported'] is False and pre['benchmark_executed'] is False
         and pre['selection_predeclared'] is True, 'preflight_status')
    probe = pre['startup_probe']
    need(probe['schema'] == 'r6-lme-failed-question-pre-provider-startup-v1'
         and probe['status'] == 'passed' and probe['phase'] == 'complete'
         and probe['cleanup_failed'] is False and probe['real_credentials_loaded'] is False
         and probe['expected_question_ids'] == manifest['question_ids']
         and type(probe['expected_question_count']) is int and probe['expected_question_count'] == 1,
         'startup_status')
    need(all(probe[field] is True for field in (
        'standalone_producer_identity_exact', 'runtime_transport_identity_exact',
        'stock_cli_pre_provider_boundary_reached', 'checkpoint_runtime_producer_matches',
        'cli_checkpoint_handles_closed', 'private_home_created', 'canonical_extra_body_environment_absent')),
        'startup_proofs')
    need(all(type(probe[field]) is int and probe[field] == 0 for field in (
        'provider_completions', 'provider_http_attempts', 'provider_successful_responses',
        'outbound_operations_blocked', 'completed_questions')), 'startup_calls')
    need(all(type(probe[field]) is int and probe[field] == 1 for field in (
        'runtime_clients_constructed', 'runtime_clients_closed', 'cli_checkpoints_observed')), 'startup_handles')
    need(pre['per_question_census'] == [{field: row[field] for field in (
        'source_index', 'question_id', 'sessions', 'messages')} for row in manifest['questions']], 'census')
    need(pre['source_files_verified'] == 231, 'source_count')
    for field in ('source_mapping_sha256', 'dataset_sha256', 'source_question_count',
                  'source_indices', 'question_ids', 'sample', 'seed', 'sessions', 'messages'):
        need(pre[field] == manifest[field], 'preflight_binding')
    key = Path(host.KEY)
    info = key.lstat()
    need(key.resolve() == key and stat.S_ISREG(info.st_mode)
         and info.st_uid == 1000 and stat.S_IMODE(info.st_mode) == 0o600, 'key_metadata')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('create', 'start', 'status'))
    parser.add_argument('mode', choices=('preflight', 'live', 'validation'))
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--container-id')
    parser.add_argument('--admission-sha256')
    args = parser.parse_args()
    need((args.action == 'create') == (args.container_id is None), 'container_argument')
    need((args.action == 'start' and args.mode == 'live') == (args.admission_sha256 is not None),
         'admission_argument')
    pin = args.manifest_sha256
    manifest, host = definitions(pin)
    plan = (host.validation_command(str(ROOT), str(SOURCE), pin) if args.mode == 'validation'
            else host.command(str(ROOT), str(SOURCE), pin, live=args.mode == 'live'))
    if args.action == 'create':
        save(args.mode + '-create-intent.json', {'manifest_sha256': pin, 'plan': plan})
        cid = command(plan['command'])
        save(args.mode + '-container-id.json', {'container_id': cid})
        result = inspect(cid, plan, pin, 'created')
        save(args.mode + '-created.json', result)
    else:
        cid = args.container_id
        need(js(ROOT / (args.mode + '-container-id.json')) == {'container_id': cid}, 'owned_container')
        if args.action == 'start':
            inspect(cid, plan, pin, 'created')
            if args.mode == 'validation':
                live_id = js(ROOT / 'live-container-id.json')['container_id']
                state = inspect(live_id, host.command(str(ROOT), str(SOURCE), pin, live=True), pin, 'exited')
                # Failed executions must still receive genuine offline archive validation.
                need(state['pid'] == 0, 'live_process_remaining')
            elif args.mode == 'live':
                live_gates(manifest, host, pin, args.admission_sha256)
            save(args.mode + '-start-intent.json', {'container_id': cid, 'manifest_sha256': pin,
                 'admission_sha256': args.admission_sha256})
            need(command(['docker', 'start', cid]) == cid, 'start_identity')
        result = inspect(cid, plan, pin)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
