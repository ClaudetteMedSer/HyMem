"""Read-only Hermes1 R7 post-deployment checks (doctor is the sole write probe).

Run only after reviewing the R7 start receipt. SSH, Docker, doctor, and MCP
output is captured; this tool emits a bounded, content-free JSON report.
"""
from __future__ import annotations

import argparse
import base64
import json
import shlex
import subprocess
import sys


MANIFEST_PIN = "1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8"
SSH_OPTIONS = (
    "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
    "-o", "ConnectionAttempts=1", "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=2",
)


CONTAINER_SCRIPT = r'''
import asyncio,hashlib,importlib.metadata as metadata,io,json,os,pathlib,re,subprocess,sys,urllib.request

PIN = "1d8917ca87dbd1b64bf3af3ee699d54b1352a3f51c3da280fd3ddd3ffc9e0ec8"
WRAPPER_EXPECTED=WRAPPER_PIN_SENTINEL
HOOK_EXPECTED=HOOK_PIN_SENTINEL
ROOT=pathlib.Path('/home/node/HyMem')
MANIFEST=pathlib.Path('/home/node/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/diag/manifest.json')
DB=pathlib.Path('/home/node/.hermes/hymem.sqlite')
LABELS={'storage root','extraction LLM','sqlite-vec','embeddings','schema',
        'embedding dimension','embedding identity','stored embedding compatibility',
        'foreign-key integrity','canonical drift','lossless coverage integrity',
        'summary context health'}
report={}

def need(ok,code):
    if not ok: raise RuntimeError(code)

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def source():
    need(sha(MANIFEST)==PIN,'manifest_pin_mismatch')
    data=json.loads(MANIFEST.read_bytes());files={}
    for group in ('source_sha256','test_sha256','auxiliary_sha256'):
        for name,pin in data[group].items():
            p=pathlib.PurePosixPath(name)
            need(name==p.as_posix() and not p.is_absolute() and '..' not in p.parts,
                 'unsafe_manifest_path')
            need(name not in files,'duplicate_manifest_path')
            files[name]=pin
    need(len(files)==479,'manifest_count_mismatch')
    if DOCTOR_EXPECTED is not None:
        need(files['hymem/doctor.py']=='5aa54d3de99a0ed9bd1c3e7cebef6f2d7472f0482783b87a46aed8a2536193bc',
             'unexpected_base_doctor')
        files['hymem/doctor.py']=DOCTOR_EXPECTED
        report['diagnostic_only_override']={'path':'hymem/doctor.py',
                                            'sha256':DOCTOR_EXPECTED}
    if PHASE1_EXPECTED is not None:
        need(files['hymem/dreaming/phase1.py']=='bea40b7a6565542861fadf9683dcbfd2bc70fe51dcf3f6b5bfc5ece5be6488ee',
             'unexpected_base_phase1')
        files['hymem/dreaming/phase1.py']=PHASE1_EXPECTED
        report['claim_conflict_override']={'path':'hymem/dreaming/phase1.py',
                                           'sha256':PHASE1_EXPECTED}
    need(all((ROOT/name).is_file() and not (ROOT/name).is_symlink()
             and sha(ROOT/name)==pin for name,pin in files.items()),
         'live_source_mismatch')
    report['source_files_verified']=len(files)

def process_env():
    honcho=[];mcp=[]
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdecimal():continue
        try: argv=(proc/'cmdline').read_bytes()
        except (OSError,PermissionError):continue
        args=argv.split(b'\0')
        if b'hymem.honcho' in args or any(arg.rsplit(b'/',1)[-1]==b'hymem-honcho'
                                              for arg in args):
            honcho.append(int(proc.name))
        if b'hymem.server' in args or any(arg.rsplit(b'/',1)[-1]==b'hymem-server'
                                              for arg in args):
            mcp.append(int(proc.name))
    need(len(honcho)==1,'honcho_process_count')
    need(bool(mcp),'mcp_process_missing')
    def read_env(pid):
        raw=(pathlib.Path('/proc')/str(pid)/'environ').read_bytes()
        values={}
        for item in raw.split(b'\0'):
            if b'=' in item:
                key,value=item.split(b'=',1)
                values[os.fsdecode(key)]=os.fsdecode(value)
        return values
    env=read_env(honcho[0])
    need(bool(env),'honcho_environment_missing')
    # An explicit HyMem key wins over provider fallback aliases. Require and
    # compare that effective credential; aliases may differ harmlessly between
    # launchers and must not be copied into a diagnostic report.
    need(bool(env.get('HYMEM_LLM_API_KEY')),'explicit_llm_key_missing')
    keys=('HYMEM_ROOT','HYMEM_LLM_MODEL','HYMEM_LLM_BASE_URL',
          'HYMEM_LLM_API_KEY','HYMEM_LLM_EXTRA_BODY','HYMEM_LLM_THINKING',
          'HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP',
          'HYMEM_EMBEDDING_API_KEY','HYMEM_EMBEDDING_BASE_URL',
          'HYMEM_EMBEDDING_MODEL','HYMEM_EMBEDDING_DIM',
          'HYMEM_EMBEDDING_PIN_DIMENSION',
          'HYMEM_EMBEDDING_DEPLOYMENT_REVISION',
          'HYMEM_EMBEDDING_DEPLOYMENT_TENANT')
    need(all(all(read_env(pid).get(key)==env.get(key) for key in keys)
             for pid in mcp),'mcp_honcho_config_mismatch')
    need(env.get('HYMEM_LLM_MODEL')=='deepseek-flash',
         'honcho_model_mismatch')
    report['processes']={'honcho_count':1,'honcho_pid':honcho[0],
                         'mcp_count':len(mcp),'mcp_pids':sorted(mcp),
                         'config_fields_compared':len(keys)}
    return env

def config(env):
    check="from hymem.bootstrap import resolve_env\nc=resolve_env()\nassert c.embedding_dim==384 and c.llm_model=='deepseek-flash'\n"
    try:
        run=subprocess.run([sys.executable,'-I','-B','-c',check],
            cwd=str(ROOT),env=env,capture_output=True,timeout=15)
    except subprocess.TimeoutExpired:
        raise RuntimeError('configuration_timeout') from None
    need(run.returncode==0,'configuration_changed')
    report['configuration']={'embedding_dimension':384,
                             'llm_model':'deepseek-flash'}

def health():
    with urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=5) as response:
        need(response.status==200,'health_http_status')
        value=json.loads(response.read(4096))
    need(value.get('status')=='ok' and value.get('backend')=='hymem',
         'health_payload_mismatch')
    report['health']='ok'

def preservation():
    pins={
      '/home/node/.hermes/bin/hymem-server-wrapper':WRAPPER_EXPECTED,
      '/home/node/.agent37/hooks/post-restart.sh':HOOK_EXPECTED,
    }
    need(all(sha(pathlib.Path(name))==pin for name,pin in pins.items()),
         'wrapper_or_hook_changed')
    rows={d.metadata['Name']:d.version for d in metadata.distributions()
          if d.metadata.get('Name','').lower()!='hymem'}
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()
    need(len(rows)==86 and digest==
         '0cd1ea43c1e3541d662cb2230fd157cce2c51b84b422b3e77a67ac776971f6c3',
         'runtime_distribution_changed')
    scripts={'hymem-server','hymem-honcho','hymem-doctor','hymem-reembed',
             'hymem-recover-summaries'}
    dist=metadata.distribution('hymem')
    actual={e.name for e in dist.entry_points if e.group=='console_scripts'}
    need(scripts<=actual and all((pathlib.Path('/home/node/hymem-env/bin')/name).is_file()
                                 for name in scripts),'entrypoints_missing')
    report['preservation']={'wrapper_sha256':pins['/home/node/.hermes/bin/hymem-server-wrapper'],
      'hook_sha256':pins['/home/node/.agent37/hooks/post-restart.sh'],
      'distribution_count':len(rows),'distribution_sha256':digest,
      'entrypoints_verified':len(scripts)}

def doctor(env):
    try:
        run=subprocess.run([sys.executable,'-I','-B','-m','hymem.doctor'],
            cwd=str(ROOT),env=env,capture_output=True,timeout=180)
    except subprocess.TimeoutExpired:
        raise RuntimeError('doctor_timeout') from None
    statuses={}
    for line in run.stdout.decode('utf-8','replace').splitlines():
        match=re.match(r'^\[( OK |WARN|FAIL)\] ([^:]+):',line)
        if match and match.group(2) in LABELS:
            need(match.group(2) not in statuses,'duplicate_doctor_label')
            statuses[match.group(2)]=match.group(1).strip()
    counts={status:sum(value==status for value in statuses.values())
            for status in ('OK','WARN','FAIL')}
    report['doctor']={'returncode':run.returncode,'statuses':statuses,'counts':counts}
    need(run.returncode==0 and counts['FAIL']==0 and
         {'storage root','extraction LLM','schema','embeddings',
          'summary context health'}<=statuses.keys(),
         'doctor_failed')

async def mcp_probe(env):
    from mcp import ClientSession,StdioServerParameters
    from mcp.client.stdio import stdio_client
    params=StdioServerParameters(command=sys.executable,
        args=['-I','-B','-m','hymem.server'],cwd=str(ROOT),env=env)
    with open(os.devnull,'w') as sink:
        async with asyncio.timeout(50):
            async with stdio_client(params,errlog=sink) as (read,write):
                async with ClientSession(read,write) as session:
                    await session.initialize()
                    tools=await session.list_tools()
                    names={item.name for item in tools.tools}
                    need({'hymem_profile','hymem_augment'}<=names,'mcp_tools_missing')
                    profile=await session.call_tool('hymem_profile',{})
                    retrieval=await session.call_tool('hymem_augment',{'message':'database'})
                    need(profile.isError is False and retrieval.isError is False,
                         'mcp_read_tool_failed')
                    report['mcp']={'handshake':True,'tool_count':len(names),
                        'profile_ok':True,'retrieval_ok':True}

def schema():
    from hymem.core import db
    conn=db.connect(DB)
    try:
        conn.execute('PRAGMA query_only=ON')
        version=db.schema_version(conn)
        good=(version==63 and
              [row[0] for row in conn.execute('PRAGMA integrity_check')]==['ok'] and
              conn.execute('PRAGMA foreign_key_check').fetchone() is None)
        need(good,'database_integrity_or_version')
    finally:conn.close()
    report['database']={'schema_version':63,'integrity_ok':True,
                        'foreign_keys_ok':True}

try:
    source();health();preservation()
    env=process_env();config(env)
    doctor(env)
    try:import mcp
    except ImportError:raise RuntimeError('mcp_sdk_unavailable') from None
    else:asyncio.run(mcp_probe(env))
    schema()
    report['status']='passed'
except BaseException as exc:
    report['status']='failed'
    report['failure_code']=(str(exc) if type(exc) is RuntimeError and
                           re.fullmatch('[a-z_]+',str(exc)) else
                           type(exc).__name__)
print(json.dumps(report,sort_keys=True))
'''


HOST_SCRIPT = r'''
import base64,json,subprocess,sys
script=base64.b64decode(SCRIPT)
try:
    check=subprocess.run(['docker','inspect','hermes-1'],capture_output=True,timeout=20)
    assert check.returncode==0
    rows=json.loads(check.stdout)
    assert len(rows)==1 and rows[0]['Name']=='/hermes-1' and rows[0]['State']['Running']
    run=subprocess.run(['docker','exec','hermes-1','/home/node/hymem-env/bin/python3',
                        '-I','-B','-c',script.decode()],capture_output=True,timeout=280)
    assert run.returncode==0
    report=json.loads(run.stdout)
    assert report.get('status') in ('passed','failed')
    print(json.dumps(report,sort_keys=True))
except BaseException as exc:
    print(json.dumps({'status':'failed','failure_code':'host_'+type(exc).__name__}))
'''


def verify(host: str, wrapper_pin: str, hook_pin: str, doctor_pin: str | None = None,
           phase1_pin: str | None = None) -> dict:
    if any(len(pin) != 64 or any(ch not in '0123456789abcdef' for ch in pin)
           for pin in (wrapper_pin, hook_pin)):
        return {'status': 'failed', 'failure_code': 'invalid_expected_hash'}
    if doctor_pin is not None and doctor_pin != '2ac786c6590aa7fbece726e6380981906d514fefe9f664a3ca4f7b1d1de7c15b':
        return {'status':'failed','failure_code':'unreviewed_doctor_override'}
    if phase1_pin is not None and phase1_pin != '31973309ab72ca0ead5493896fc4b6cad104fc416128e83a80e3c8fb44f94136':
        return {'status':'failed','failure_code':'unreviewed_phase1_override'}
    script = (CONTAINER_SCRIPT
              .replace('WRAPPER_PIN_SENTINEL', repr(wrapper_pin))
              .replace('HOOK_PIN_SENTINEL', repr(hook_pin)))
    script = 'DOCTOR_EXPECTED='+repr(doctor_pin)+'\nPHASE1_EXPECTED='+repr(phase1_pin)+'\n'+script
    encoded = base64.b64encode(script.encode()).decode()
    remote = HOST_SCRIPT.replace('SCRIPT', repr(encoded))
    command = ['ssh', *SSH_OPTIONS, host, 'python3 -I -B -c ' + shlex.quote(remote)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=320)
        if result.returncode != 0:
            return {'status': 'failed', 'failure_code': 'ssh_failed'}
        report = json.loads(result.stdout)
        if not isinstance(report, dict) or report.get('status') not in ('passed', 'failed'):
            raise ValueError('invalid_report')
        return report
    except subprocess.TimeoutExpired:
        return {'status': 'failed', 'failure_code': 'ssh_timeout'}
    except (ValueError, UnicodeDecodeError, TypeError):
        return {'status': 'failed', 'failure_code': 'invalid_report'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ssh_host', help='reviewed Hermes1 SSH alias')
    parser.add_argument('--wrapper-sha256', required=True,
                        help='operator-reviewed post-override wrapper hash')
    parser.add_argument('--hook-sha256', required=True,
                        help='operator-reviewed post-override hook hash')
    parser.add_argument('--doctor-sha256', help='reviewed diagnostic-only hotfix hash')
    parser.add_argument('--phase1-sha256', help='reviewed claim-conflict hotfix hash')
    args = parser.parse_args()
    report = verify(args.ssh_host, args.wrapper_sha256, args.hook_sha256, args.doctor_sha256,
                    args.phase1_sha256)
    print(json.dumps(report, sort_keys=True))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    sys.exit(main())
