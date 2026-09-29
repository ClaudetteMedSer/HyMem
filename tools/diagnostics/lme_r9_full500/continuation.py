"""Pinned one-shot host pipeline. No install/launch, retry, resume or deployment API.

Run only after root review, as uid1000 on Afrodite, detached by the operator.
Timeout/ambiguous dispatch stops with a private operator-pending receipt; it
never kills or removes containers, reads credentials, or exports raw output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import types

if sys.flags.optimize:
    raise RuntimeError('optimized_execution_forbidden')
BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky')
EVIDENCE_PINS = {
    'suite': 'c8f2cae80ac082b65c619402cdead920912073b4f64fcc734752316612a05842',
    'sample8': '2ba18ff05ec1b679ea5f74cb8f3153558e042bc070811a9cc7a7476b71649b80',
}
SEMANTIC_REVIEW = {
    'admission': 'full500_development_integration_test',
    'same_source_sample8_validated': True,
    'full_suite_passed': True,
    'semantic_quality_guaranteed': False,
    'official_comparable': False,
}

HEX = re.compile('[0-9a-f]{64}\\Z')

def need(value, label):
    if not value:
        raise RuntimeError(label)

def read(path):
    need(path.is_absolute() and path.resolve() == path and not path.is_symlink()
         and path.is_file() and path.stat().st_size <= 32*1024*1024, 'input_scope')
    return path.read_bytes()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def save(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    fd = os.open(path, os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(raw); out.flush(); os.fsync(out.fileno())

def load(path, pin, name):
    raw = read(path); need(sha(raw) == pin, 'code_pin')
    obj = types.ModuleType(name); obj.__file__ = str(path)
    sys.modules[name] = obj
    exec(compile(raw, str(path), 'exec'), obj.__dict__)
    return obj

def checked_config(cfg):
    need(set(cfg) == {'schema','root','target','target_pin','controller_pin','source_pin',
         'source_files','expected_collected','suite_cid','preflight_cid','evidence',
         'semantic_review'}, 'config_fields')
    need(cfg['schema'] == 'r9-headless-continuation-v1', 'config_schema')
    root, target = Path(cfg['root']), Path(cfg['target'])
    need(root == BASE/'lme-r9-full500-continuation-v1'
         and target == BASE/'lme-r9-full500-headless-v1', 'root_scope')
    need(root.resolve() == root and target.resolve() == target, 'canonical_scope')
    need(type(cfg['source_files']) is int and cfg['source_files'] == 508
         and type(cfg['expected_collected']) is int and cfg['expected_collected'] == 7940, 'source_contract')
    need(cfg['source_pin'] == '35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51',
         'source_pin')
    need(json.dumps(cfg['semantic_review'], sort_keys=True) == json.dumps(SEMANTIC_REVIEW, sort_keys=True), 'semantic_review')
    for key in ('target_pin','controller_pin','source_pin','suite_cid','preflight_cid'):
        need(type(cfg[key]) is str and HEX.fullmatch(cfg[key]), 'pin_format')
    need(set(cfg['evidence']) == {'suite','sample8'}, 'evidence_fields')
    for name, value in cfg['evidence'].items():
        need(set(value) == {'root','manifest_sha256'}, 'binding_fields')
        need(value['root'] == str(BASE/('lme-r9-full-suite-v1' if name == 'suite'
                                      else 'lme-r9-sample8-headless-v1'))
             and value['manifest_sha256'] == EVIDENCE_PINS[name], 'evidence_scope')
    return root, target

def wait(cid, seconds):
    need(HEX.fullmatch(cid) and seconds in (10860,2592060,3600), 'wait_scope')
    reply = subprocess.run(['docker','wait',cid], capture_output=True, timeout=seconds)
    need(reply.returncode == 0 and re.fullmatch(rb'[0-9]{1,3}\n?', reply.stdout), 'wait_failed')
    code = int(reply.stdout); need(0 <= code <= 255, 'exit_range')
    return code

def terminal(state, code, *, allow_oom=False):
    need(state['status'] == 'exited' and state['pid'] == 0
         and state['exit_code'] == code and type(state['oom_killed']) is bool
         and (allow_oom or state['oom_killed'] is False), 'terminal_state')

def bound(path):
    raw = read(path)
    return {'path':str(path),'sha256':sha(raw)}

def gate(cfg, manifest):
    evidence = {}
    for name, binding in cfg['evidence'].items():
        root = Path(binding['root'])
        mp = root/('diag/manifest.json' if name == 'suite' else 'bundle/manifest.json')
        need(sha(read(mp)) == binding['manifest_sha256'], 'evidence_manifest_pin')
        data = json.loads(read(mp))
        need(data['source_sha256'] == manifest['source_sha256'], 'exact_source')
        if name == 'suite':
            need(data['expected_collected'] == cfg['expected_collected'], 'suite_collection')
        results = root/('work' if name == 'suite' else 'live-results')
        receipt = results/'result.json' if name == 'suite' else Path(cfg['target'])/'prior-sample8-validation.json'
        supervisor = results/('supervisor.json' if name == 'suite' else 'supervisor-summary.json')
        evidence[name] = {'manifest':bound(mp), 'receipt':bound(receipt), 'supervisor':bound(supervisor)}
    return {'schema':'lme-r9-root-reviewed-candidate-gate-v1','root_reviewed':True,
        'status':'passed','schema_version':64,'source_manifest_sha256':cfg['source_pin'],
        'preparation_manifest_sha256':cfg['target_pin'],
        'runtime_seal_sha256':manifest['runtime_seal_sha256'],
        'isolated_candidate_regression':True,'deployment_performed':False,
        'production_source_verified':False, 'semantic_review':cfg['semantic_review'],
        'evidence':evidence}

def dispatch(host_path, cfg, action, mode, cid=None, gate_pin=None):
    need((action,mode) in {('create','live'),('start','live'),('status','live'),
         ('create','validation'),('start','validation'),('status','validation')}, 'action_scope')
    argv = [sys.executable,'-I','-B',str(host_path),action,mode]
    if cid is not None: need(HEX.fullmatch(cid), 'cid'); argv.append(cid)
    argv += ['--root',cfg['target'],'--manifest-sha256',cfg['target_pin']]
    if gate_pin: argv += ['--candidate-gate-sha256',gate_pin]
    reply = subprocess.run(argv,capture_output=True,timeout=600)
    need(reply.returncode == 0 and len(reply.stdout) <= 1024*1024, 'dispatch_ambiguous')
    state = json.loads(reply.stdout)
    need(state['configuration_verified'] is True and HEX.fullmatch(state['container_id']), 'dispatch_state')
    if cid: need(state['container_id'] == cid, 'dispatch_identity')
    return state

def pipeline(cfg, controller, suite, record, waiting=wait, sending=dispatch):
    """Dependencies are injectable for zero-network failure/ordering tests."""
    root, target = checked_config(cfg)
    outcome = {'status':'operator_pending','paid_runs_started':0,'validation_runs_started':0}
    # Only the exclusive intent owner may write progress or terminal receipts.
    record('intent.json', {'config':cfg,'live_runs_allowed':1,'validation_runs_allowed':1})
    try:
        manifest, plan_host = controller.definitions()
        need(len(manifest['source_sha256']) == cfg['source_files']
             and manifest['approved_source_manifest_sha256'] == cfg['source_pin'], 'source_pin')
        need(controller.js(target/'preflight-container-id.json')['container_id'] == cfg['preflight_cid'], 'preflight_identity')
        suite_root = Path(cfg['evidence']['suite']['root'])
        need(json.loads(read(suite_root/'launch.json')) ==
             {'manifest_sha256':cfg['evidence']['suite']['manifest_sha256'],'container_id':cfg['suite_cid']}, 'suite_owner')
        suite.verify(suite_root,cfg['evidence']['suite']['manifest_sha256'])
        code = waiting(cfg['suite_cid'],10860)
        terminal(suite.inspect(cfg['suite_cid'],suite_root,cfg['evidence']['suite']['manifest_sha256']), code)
        need(code == 0, 'suite_failed')
        candidate_gate = gate(cfg,manifest)
        controller.candidate_evidence(candidate_gate,manifest)
        save(target/'root-reviewed-candidate-gate.json',candidate_gate)
        gate_pin = sha(read(target/'root-reviewed-candidate-gate.json'))
        controller.GATE_PIN = gate_pin
        controller.live_gates(manifest,plan_host)
        record('admitted.json', {'gate_sha256':gate_pin,'suite_exit_code':code})
        host_path = target/'host_control.py'
        created = sending(host_path,cfg,'create','live')
        cid = created['container_id']; need(created['status'] == 'created', 'live_create')
        record('live-owned.json', {'container_id':cid})
        record('live-start-requested.json', {'container_id':cid})
        outcome.update(paid_runs_started=None, live_start_attempted=True)
        started = sending(host_path,cfg,'start','live',cid,gate_pin)
        need(started['status'] in ('running','exited'), 'live_start')
        outcome['paid_runs_started'] = 1
        code = waiting(cid,2592060)
        live_state = sending(host_path,cfg,'status','live',cid)
        terminal(live_state,code,allow_oom=True)
        outcome['live_exit_code'] = code
        created = sending(host_path,cfg,'create','validation')
        vid = created['container_id']; need(created['status'] == 'created', 'validation_create')
        record('validation-owned.json', {'container_id':vid})
        record('validation-start-requested.json', {'container_id':vid})
        outcome.update(validation_runs_started=None, validation_start_attempted=True)
        started = sending(host_path,cfg,'start','validation',vid)
        need(started['status'] in ('running','exited'), 'validation_start')
        outcome['validation_runs_started'] = 1
        vcode = waiting(vid,3600)
        terminal(sending(host_path,cfg,'status','validation',vid),vcode)
        outcome.update(status='offline_validation_finished',validation_exit_code=vcode,
                       live_success=code == 0 and live_state['oom_killed'] is False,
                       validation_success=vcode == 0)
    except BaseException as exc:
        outcome.update(error_type=type(exc).__name__, automatic_retry=False,
                       operator_inspection_required=True)
    record('result.json',outcome)
    return outcome

def exit_status(outcome):
    return 0 if (outcome['status'] == 'offline_validation_finished'
                 and outcome['live_success'] is True
                 and outcome['validation_success'] is True) else 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--self-sha256',required=True)
    args = parser.parse_args()
    need(os.geteuid() == 1000, 'host_uid')
    need(sha(read(Path(__file__).resolve())) == args.self_sha256, 'self_pin')
    raw = read(args.config); need(sha(raw) == args.config_sha256, 'config_pin')
    cfg = json.loads(raw); root,target = checked_config(cfg)
    need(args.config.parent == root and root.is_dir() and root.stat().st_uid == 1000
         and root.stat().st_mode & 0o077 == 0, 'private_root')
    controller = load(target/'host_control.py',cfg['controller_pin'],'continuation_controller')
    controller.ROOT, controller.PIN = target,cfg['target_pin']
    mp = json.loads(read(target/'bundle/manifest.json'))
    need(sha(read(target/'bundle/manifest.json')) == cfg['target_pin'], 'target_pin')
    need(mp['continuation_sha256'] == args.self_sha256, 'sealed_continuation_pin')
    controller.SOURCE = Path(mp['remote_source'])
    sr = Path(cfg['evidence']['suite']['root'])
    smraw = read(sr/'diag/manifest.json')
    need(sha(smraw) == cfg['evidence']['suite']['manifest_sha256'], 'suite_pin')
    suite = load(sr/'diag/runner.py',json.loads(smraw)['runner_sha256'],'continuation_suite')
    return exit_status(pipeline(cfg,controller,suite,lambda name,value:save(root/name,value)))

if __name__ == '__main__': sys.exit(main())
