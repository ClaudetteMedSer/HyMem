"""Seal then explicitly launch one offline full suite. No provider credentials.

Seal only after candidate owner reports frozen. Supply reviewed source inventory
digest and expected collection. Transport sealed package intact to Afrodite;
verify its manifest pin before explicit launch. Status never restarts a run.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import types

IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
SECONDS = 10800
ENV = {'PATH':'/home/node/hymem-env/bin:/usr/local/bin:/usr/bin:/bin','HOME':'/work',
       'TMPDIR':'/work','PYTHONDONTWRITEBYTECODE':'1','PYTEST_DISABLE_PLUGIN_AUTOLOAD':'1'}


def need(value, code):
    if not value:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def inventory(root, clean=False):
    need(root.is_absolute() and root.resolve() == root and root.is_dir(), 'source_path')
    result = {}
    for path in root.rglob('*'):
        relative = path.relative_to(root)
        if clean and (any(p in ('.git','.pytest_cache','__pycache__') for p in relative.parts)
                      or path.suffix in ('.pyc','.pyo')):
            continue
        need(not path.is_symlink() and (path.is_dir() or path.is_file()), 'source_special_file')
        if path.is_file():
            result[relative.as_posix()] = sha(path)
    return result


def save(path, value):
    with path.open('x') as stream:
        json.dump(value,stream,sort_keys=True,indent=2)
        stream.write('\n')


def seal(tree, root, source_pin, count):
    need(type(count) is int and count >= 7764, 'collection_admission')
    need(root.is_absolute() and root.resolve() == root and not root.exists()
         and not root.is_relative_to(tree), 'fresh_package_required')
    pins = inventory(tree,clean=True)
    need(digest(pins) == source_pin, 'source_inventory_pin')
    need('hymem/__init__.py' in pins and 'pyproject.toml' in pins
         and 'tests/test_summary_overview_policy.py' in pins, 'suite_scope')
    here = Path(__file__).resolve()
    supervisor = here.parent/'lme_sample8_v1/bundle/supervised_invocation.py'
    manifest = {'schema':'r8-offline-full-suite-v1','source_sha256':pins,
                'source_inventory_sha256':source_pin,'expected_collected':count,
                'runner_sha256':sha(here),'supervisor_sha256':sha(supervisor),
                'timeout_seconds':SECONDS,'image':IMAGE,'runtime':RUNTIME}
    root.mkdir(mode=0o755)
    for name in pins:
        path = root/'candidate'/name
        path.parent.mkdir(mode=0o755,parents=True,exist_ok=True)
        shutil.copyfile(tree/name,path)
    (root/'diag').mkdir(mode=0o755)
    shutil.copyfile(here,root/'diag/runner.py')
    shutil.copyfile(supervisor,root/'diag/supervised_invocation.py')
    save(root/'diag/manifest.json',manifest)
    need(inventory(tree,clean=True) == pins and inventory(root/'candidate') == pins, 'seal_source_changed')
    return {'manifest_sha256':sha(root/'diag/manifest.json'),'source_inventory_sha256':source_pin,
            'source_files':len(pins),'test_files':sum(p.startswith('tests/') for p in pins),
            'expected_collected':count,'provider_calls':0,'launched':False}


def verify(root, pin):
    need(re.fullmatch('[0-9a-f]{64}',pin) and sha(root/'diag/manifest.json') == pin,'manifest_pin')
    value = json.loads((root/'diag/manifest.json').read_text())
    need(value['schema'] == 'r8-offline-full-suite-v1' and value['timeout_seconds'] == SECONDS
         and value['image'] == IMAGE and value['runtime'] == RUNTIME
         and type(value['expected_collected']) is int and value['expected_collected'] >= 7764,'manifest_bounds')
    need(sha(Path(__file__)) == value['runner_sha256']
         and sha(root/'diag/supervised_invocation.py') == value['supervisor_sha256'],'helper_pin')
    need(inventory(root/'candidate') == value['source_sha256']
         and digest(value['source_sha256']) == value['source_inventory_sha256'],'source_drift')
    return value


def pytest_stage(mode):
    tree = Path('/work/tree')
    sys.path.insert(0,str(tree))
    import hymem
    import pytest
    need(Path(hymem.__file__).resolve() == tree/'hymem/__init__.py','application_import')
    class Counts:
        def __init__(self):
            self.counts = {'passed':0,'failed':0,'skipped':0,'errors':0}
        def pytest_runtest_logreport(self,report):
            if report.failed:
                self.counts['failed' if report.when == 'call' else 'errors'] += 1
            elif report.skipped:
                self.counts['skipped'] += 1
            elif report.when == 'call' and report.passed:
                self.counts['passed'] += 1
        def pytest_sessionfinish(self,session,exitstatus):
            save(Path('/work')/(mode+'-counts.json'),{**self.counts,
                 'collected':session.testscollected,'exit_code':int(exitstatus)})
    args = ['-ra','-p','no:cacheprovider','--basetemp=/work/pytest-'+mode,
            '--rootdir=/work/tree','-c','/work/tree/pyproject.toml',
            '--junitxml=/work/'+mode+'-junit.xml','tests']
    if mode == 'collect':
        args.insert(0,'--collect-only')
    return int(pytest.main(args,plugins=[Counts()]))


def worker(pin):
    os.environ.clear()
    os.environ.update(ENV)
    os.umask(0o077)
    manifest = verify(Path('/'),pin)
    need(sys.stdin.buffer.read(256) == (pin+'\n').encode(),'worker_authority')
    tree = Path('/work/tree')
    need(not tree.exists(),'fresh_work_tree')
    shutil.copytree('/candidate',tree)
    need(inventory(tree) == manifest['source_sha256'],'work_inventory')
    result = {'status':'collect_failed','manifest_sha256':pin,'full_runs_started':0,'provider_calls':0}
    try:
        for mode in ('collect','full'):
            with (Path('/work')/(mode+'.log')).open('xb') as output:
                rc = subprocess.run([sys.executable,'-I','-B','/diag/runner.py','pytest-'+mode],
                    cwd=tree,env=ENV,stdout=output,stderr=output).returncode
            counts = json.loads((Path('/work')/(mode+'-counts.json')).read_text())
            result[mode] = counts
            need(counts['collected'] == manifest['expected_collected'],'collection_drift')
            if mode == 'collect' and rc:
                break
            if mode == 'collect':
                result['full_runs_started'] = 1
                result['status'] = 'tests_failed'
            else:
                result['status'] = 'passed' if rc == 0 and counts['exit_code'] == 0 else 'tests_failed'
    except BaseException as exc:
        result.update(status='error',exception_type=type(exc).__name__)
    finally:
        verify(Path('/'),pin)
        need(inventory(tree) == manifest['source_sha256'],'tested_source_changed')
        result.update(source_unchanged=True,tested_source_unchanged=True)
        save(Path('/work/result.json'),result)
    return 0 if result['status'] == 'passed' else 1


def supervise(pin):
    manifest = verify(Path('/'),pin)
    path = Path('/diag/supervised_invocation.py')
    module = types.ModuleType('r8_suite_supervisor')
    module.__file__ = str(path)
    sys.modules[module.__name__] = module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    def cancel(_signal,_frame):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGINT,cancel)
    signal.signal(signal.SIGTERM,cancel)
    outcome = module.supervise_invocation(
        [sys.executable,'-I','-B','/diag/runner.py','worker','--pin',pin],
        cwd=Path('/work'),env=ENV,output_dir=Path('/work/invocation'),
        timeout_seconds=manifest['timeout_seconds'],cleanup_seconds=10,
        output_limit_bytes=2*1024*1024,stdin_bytes=(pin+'\n').encode())
    verify(Path('/'),pin)
    save(Path('/work/supervisor.json'),{'manifest_sha256':pin,'outcome':asdict(outcome)})
    return 0 if outcome.returncode == 0 and outcome.safe_to_continue else 1


def run(command):
    result = subprocess.run(command,capture_output=True,text=True,timeout=30)
    need(result.returncode == 0,'docker_failed')
    return result.stdout.strip()


def configuration(root,pin):
    mounts = {'/candidate':(str(root/'candidate'),False),'/diag':(str(root/'diag'),False),
              '/work':(str(root/'work'),True),'/home/node/hymem-env':(RUNTIME,False)}
    command = ['docker','create','--name','hymem-r8-suite-'+pin[:16],'--pull','never','--init',
               '--network','none','--user','1000:1000','--read-only','--cap-drop','ALL',
               '--security-opt','no-new-privileges','--pids-limit','256','--memory','2g','--cpus','2',
               '--tmpfs','/tmp:rw,noexec,nosuid,size=64m']
    for key,value in ENV.items():
        command += ['--env',key+'='+value]
    for target,(source,writable) in mounts.items():
        command += ['--mount','type=bind,src='+source+',dst='+target+('' if writable else ',readonly')]
    command += ['--workdir','/work','--entrypoint','/home/node/hymem-env/bin/python3',IMAGE,
                '-I','-B','/diag/runner.py','supervise','--pin',pin]
    return command,mounts


def inspect(cid,root,pin):
    need(re.fullmatch('[0-9a-f]{64}',cid),'container_id')
    data = json.loads(run(['docker','inspect',cid]))[0]
    command,mounts = configuration(root,pin)
    config,host,state = data['Config'],data['HostConfig'],data['State']
    need(data['Id'] == cid and data['Image'] == IMAGE and config['User'] == '1000:1000'
         and config['WorkingDir'] == '/work' and config['Entrypoint'] == ['/home/node/hymem-env/bin/python3']
         and config['Cmd'] == command[command.index(IMAGE)+1:] and host['NetworkMode'] == 'none'
         and host['ReadonlyRootfs'] and not host['Privileged'] and host['Init']
         and host['CapDrop'] == ['ALL'] and host['SecurityOpt'] == ['no-new-privileges']
         and host['PidsLimit'] == 256 and host['Memory'] == 2147483648 and host['NanoCpus'] == 2000000000
         and host['RestartPolicy']['Name'] == 'no' and host['Tmpfs'] == {'/tmp':'rw,noexec,nosuid,size=64m'}
         and not any(v.startswith(('HYMEM_','OPENAI_','DEEPSEEK_')) for v in config['Env'])
         and {m['Destination']:(m['Source'],m['RW']) for m in data['Mounts']} == mounts,'container_configuration')
    return {'container_id':cid,'status':state['Status'],'exit_code':state['ExitCode'],
            'pid':state['Pid'],'oom_killed':state['OOMKilled']}


def host_action(root,pin,status=False):
    verify(root,pin)
    receipt = root/'launch.json'
    if status:
        value = json.loads(receipt.read_text())
        need(value['manifest_sha256'] == pin,'receipt_pin')
        value.update(inspect(value['container_id'],root,pin))
        for name in ('result.json','supervisor.json'):
            if (root/'work'/name).is_file():
                value[name] = json.loads((root/'work'/name).read_text())
        return value
    need(not receipt.exists() and not (root/'work').exists(),'already_launched')
    (root/'work').mkdir(mode=0o700)
    os.chown(root/'work',1000,1000)
    command,_ = configuration(root,pin)
    cid = run(command)
    state = inspect(cid,root,pin)
    need(state['status'] == 'created','container_not_created')
    save(receipt,{'manifest_sha256':pin,'container_id':cid})
    need(run(['docker','start',cid]) == cid,'container_start')
    return {'manifest_sha256':pin,**inspect(cid,root,pin)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('seal','launch','status','supervise','worker','pytest-collect','pytest-full'))
    parser.add_argument('--root',type=Path)
    parser.add_argument('--tree',type=Path)
    parser.add_argument('--pin')
    parser.add_argument('--source-inventory-sha256')
    parser.add_argument('--expected-collected',type=int)
    args = parser.parse_args()
    if args.action.startswith('pytest-'):
        return pytest_stage(args.action.removeprefix('pytest-'))
    if args.action == 'worker':
        return worker(args.pin)
    if args.action == 'supervise':
        return supervise(args.pin)
    value = seal(args.tree,args.root,args.source_inventory_sha256,args.expected_collected) if args.action == 'seal' else host_action(args.root,args.pin,args.action == 'status')
    print(json.dumps(value,sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt,Exception) as exc:
        print(json.dumps({'status':'suite_error','exception_type':type(exc).__name__},sort_keys=True))
        raise SystemExit(1)
