"""Separate read-only production admission; final vector verifier is unchanged."""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hymem_v64_rollout as r
import hymem_v64_postdeploy as final

DEPENDENCY_PINS={
    'hymem_v64_rollout.py':'222a4c1222d8a36df94bda6203c53aeeab3a74248142efc39b996f2ac224ebe0',
    'hymem_v64_postdeploy.py':'6289c704984898f611a2b03de40415aa6e49985ca0bad05baa7f254d42e992aa',
    'hymem_v64_vector_check.py':'7d71fb34d3f76277cb522af9a3ff771affe930a5db09d3eb61b0b68f1f8e7031',
    'lme_r7_postdeploy_verify.py':'6f4351ef2df7f07b03d518842ae5ad47df2a18ae8d541202a8ae55759ad29da9'}

BASELINE={'available':True,'verifiable':False,'aligned':False,'expected_count':19,
          'actual_count':27,'surplus_count':8,'missing_count':0,'different_count':0,
          'unverifiable_episodes':17}

def container_script(files,census):
    for name,pin in DEPENDENCY_PINS.items():
        r.need(r.sha(r.regular(Path(__file__).resolve().parent/name))==pin,'predream_dependency_pin_drift')
    baseline=census.get('episode_vector_baseline')
    r.need(isinstance(baseline,dict) and all(baseline.get(k)==v for k,v in BASELINE.items()),
           'inherited_baseline_receipt_required')
    r.need(all(isinstance(baseline.get(k),str) and r.HEX.fullmatch(baseline[k])
               for k in ('expected_sha256','actual_sha256')),'inherited_baseline_fingerprints_required')
    script=final.container_script(files,census)
    old="        need(vectors['verifiable'] is True and vectors['aligned'] is True,'episode_vector_alignment')"
    replacement="        need(vectors==EXPECTED_BASELINE,'inherited_vector_baseline_changed')"
    r.need(script.count(old)==1,'strict_template_shape_changed')
    script=script.replace(old,replacement)
    counts=census.get('processes',{}).get('roles',{})
    r.need(len(counts.get('honcho',[]))==1 and len(counts.get('mcp',[]))==2,'sealed_process_counts_required')
    script=script.replace('    return env',"    need(len(honcho)==1 and len(mcp)==2,'process_count_changed')\n    return env")
    script=script.replace("    conn=db.connect(DB)","    conn=sqlite3.connect('file:/home/node/.hermes/hymem.sqlite?mode=ro',uri=True)\n    conn.row_factory=sqlite3.Row")
    script=script.replace("'episode_vectors':True","'episode_vectors':False")
    module=ast.parse(script)
    # Preserve all function definitions. Replace only the old execution block.
    block=next(n for n in module.body if isinstance(n,ast.Try))
    lines=script.splitlines(keepends=True)
    script='EXPECTED_BASELINE='+repr(baseline)+'\n'+''.join(lines[:block.lineno-1])
    script+=r'''
try:
    source();health();preservation()
    env=process_env();config(env)
    import urllib.request
    status=json.load(urllib.request.urlopen('http://127.0.0.1:8765/dream-status',timeout=5))
    need(status.get('in_progress') is False,'dream_active')
    from hymem.core import db
    import sqlite3
    c=sqlite3.connect('file:/home/node/.hermes/hymem.sqlite?mode=ro',uri=True)
    try:
        c.execute('PRAGMA query_only=ON')
        need(c.execute('SELECT COUNT(*) FROM dream_runs WHERE ended_at IS NULL AND skipped_locked=0').fetchone()[0]==0,'open_dream_run')
    finally:c.close()
    schema()
    report.update(status='passed',phase='pre_dream',vector_health_passed=False,
                  inherited_vector_baseline_preserved=True,doctor_performed=False,mcp_probe_performed=False)
except Exception as exc:
    report={'status':'failed','phase':'pre_dream','failure_code':str(exc) if type(exc) is RuntimeError and re.fullmatch('[a-z_]+',str(exc)) else type(exc).__name__}
print(json.dumps(report,sort_keys=True))
'''
    compile(script,'<v64-predream>','exec')
    return script

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args()
    cfg=r.read_sealed(args.config,args.config_sha256)
    rollout=r.Rollout(cfg,args.config_sha256)
    rollout.prior('start');rollout.prior('migrate');rollout.inspect();rollout.preserve()
    r.verify_files(r.LIVE,rollout.files)
    script=container_script(rollout.files,rollout.census)
    raw=rollout.run(['docker','exec','hermes-1','/home/node/hymem-env/bin/python3','-I','-B','-c',script],timeout=90)
    report=json.loads(raw)
    r.need(report.get('status')=='passed' and report.get('inherited_vector_baseline_preserved') is True
           and report.get('role_environment_preserved') is True,'predream_verification_failed')
    rollout.inspect();rollout.preserve()
    print(json.dumps(rollout.receipt('predream',{'verification':report}),sort_keys=True))

if __name__=='__main__':
    try:main()
    except Exception:
        print(json.dumps({'status':'failed','phase':'pre_dream','inspect_private_evidence':True}));raise SystemExit(1)
