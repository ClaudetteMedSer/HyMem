"""Pinned isolated clone/preflight and externally supervised recovery protocol."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import stat
import sys
import time
import types

PACKAGE=Path('/diag')
SOURCE=Path('/candidate')
OUTPUT=Path('/results')
REFERENCE=Path('/reference/hymem.sqlite')
CLONE=Path('/work/hymem.sqlite')
R5=Path('/r5/manifest.json')
R5_PIN='c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'
SUPERVISOR_PIN='9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'

def need(value,code):
    if not value: raise RuntimeError(code)

def read(path):
    need(path.resolve()==path and stat.S_ISREG(path.lstat().st_mode),'regular_file')
    return path.read_bytes()

def sha(raw): return hashlib.sha256(raw).hexdigest()

def load(path,name):
    module=types.ModuleType(name); module.__file__=str(path); sys.modules[name]=module
    exec(compile(read(path),str(path),'exec'),module.__dict__)
    return module

def verify(pin):
    raw=read(PACKAGE/'manifest.json'); need(sha(raw)==pin,'manifest_pin')
    manifest=json.loads(raw)
    need(manifest['schema']=='isolated-summary-recovery-package-v1'
         and manifest['source_manifest_sha256']==R5_PIN,'package_contract')
    pins=manifest['helper_sha256']
    need(set(pins)=={'worker.py','protocol.py','supervised_invocation.py'},'helper_contract')
    need({p.name for p in PACKAGE.iterdir()}==set(pins)|{'manifest.json'},'package_inventory')
    need(pins['supervised_invocation.py']==SUPERVISOR_PIN,'supervisor_pin')
    need(all(sha(read(PACKAGE/name))==expected for name,expected in pins.items()),'helper_drift')
    need(sha(read(R5))==R5_PIN,'r5_pin')
    worker=load(PACKAGE/'worker.py','verified_summary_worker')
    worker.verify_source(SOURCE,R5)
    need(worker.BOUNDS==dict(max_calls=100,max_attempts=3,max_chars=8000,max_tokens=3072,
                           timeout_seconds=1800),'recovery_bounds')
    return manifest,worker

def environment():
    return {'PATH':'/home/node/hymem-env/bin:/usr/bin:/bin','HOME':'/tmp',
            'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1','LANG':'C.UTF-8'}

def arguments(live):
    args=['/diag/worker.py','--source',str(SOURCE),'--r5-manifest',str(R5),
          '--clone',str(CLONE),'--reference',str(REFERENCE),'--receipt',
          str(OUTPUT/('recovery.json' if live else 'preflight.json'))]
    return args+(['--credential-file','/run/deepseek.env'] if live else ['--preflight'])

def clone_reference():
    read(REFERENCE)  # Require a regular canonical file; SQLite backup includes WAL.
    need(CLONE.parent.resolve()==CLONE.parent and CLONE.parent.is_dir(),'clone_parent')
    fd=os.open(CLONE,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600); os.close(fd)
    source=sqlite3.connect(REFERENCE.as_uri()+'?mode=ro',uri=True,timeout=5)
    target=sqlite3.connect(CLONE,timeout=5)
    try:
        expiry=time.monotonic()+120
        def check_progress(_status,_remaining,_total):
            need(time.monotonic()<expiry,'backup_deadline')
        source.execute('PRAGMA query_only=ON')
        # Pin one WAL read snapshot across every sqlite3_backup_step. Without
        # a live read transaction, a paged backup can repeatedly restart after
        # refreshing the source snapshot and never advance beyond its first page batch.
        source.execute('BEGIN')
        source.execute('SELECT COUNT(*) FROM sqlite_schema').fetchone()
        source.backup(target,pages=256,progress=check_progress)
    finally:
        try:
            source.rollback()
        finally:
            try:
                target.close()
            finally:
                source.close()

def cancel(_signal,_frame): raise KeyboardInterrupt('recovery_cancelled')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('preflight','live'))
    parser.add_argument('--manifest-sha256',required=True)
    args=parser.parse_args()
    os.umask(0o077); os.environ.clear(); os.environ.update(environment())
    need(os.geteuid()==1000,'isolated_user')
    manifest,worker=verify(args.manifest_sha256)
    if args.mode=='preflight':
        clone_reference()
        sys.argv=arguments(False)
        code=worker.main()
        verify(args.manifest_sha256)
        return code
    need(CLONE.exists(),'prepared_clone_required')
    supervisor=load(PACKAGE/'supervised_invocation.py','recovery_supervisor')
    signal.signal(signal.SIGINT,cancel); signal.signal(signal.SIGTERM,cancel)
    outcome=None; failure=None
    try:
        outcome=supervisor.supervise_invocation(
            [sys.executable,'-I','-B',*arguments(True)],cwd=SOURCE,env=environment(),
            output_dir=OUTPUT/'invocation',timeout_seconds=1860,cleanup_seconds=10,
            output_limit_bytes=16*1024*1024,
            stdin_bytes=worker.encoded(worker.AUTHORIZATION)+b'\n')
    except BaseException as exc:
        failure=type(exc).__name__
    unchanged=False
    try:
        verify(args.manifest_sha256); unchanged=True
    except BaseException as exc:
        failure=failure or type(exc).__name__
    receipt={'schema':'summary-recovery-supervisor-v1','manifest_sha256':args.manifest_sha256,
             'outcome':asdict(outcome) if outcome else None,'exception_type':failure,
             'package_and_source_unchanged':unchanged,'production_changes':False}
    worker.save(OUTPUT/'supervisor.json',receipt)
    return 0 if (failure is None and outcome is not None and outcome.status=='completed'
        and outcome.returncode==0 and outcome.safe_to_continue and unchanged) else 1

if __name__=='__main__':
    try: raise SystemExit(main())
    except (KeyboardInterrupt,Exception) as exc:
        print(json.dumps({'status':'recovery_protocol_failed','exception_type':type(exc).__name__}))
        raise SystemExit(1)
