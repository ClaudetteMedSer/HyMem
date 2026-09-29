"""Prepared v64 postdeploy checks; invoke on Afrodite after sealed start receipt.

Preserves existing role-specific aggregation configuration. Doctor is the sole
intentional write probe; source/runtime/config and effective role profiles are
checked before probes. No SSH is invoked.
"""
from __future__ import annotations
import argparse
import ast
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import hymem_v64_rollout as r
import lme_r7_postdeploy_verify as old
from hymem_v64_vector_check import CHECKER

TEMPLATE_SHA256='6f4351ef2df7f07b03d518842ae5ad47df2a18ae8d541202a8ae55759ad29da9'

def replace_function(script,name,replacement):
    module=ast.parse(script)
    node=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=script.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement+'\n'+''.join(lines[node.end_lineno:])

def container_script(files,census):
    r.need(r.sha(r.regular(Path(old.__file__)))==TEMPLATE_SHA256,'postdeploy_template_pin_drift')
    script=old.CONTAINER_SCRIPT
    script=replace_function(script,'source',"""def source():
    need(len(EXPECTED_FILES)==481,'manifest_count_mismatch')
    for name,pin in EXPECTED_FILES.items():
        path=ROOT/name
        need(path.is_file() and not path.is_symlink() and sha(path)==pin,'live_source_mismatch')
    report['source_files_verified']=481
""")
    insertion="""    actual_roles={'honcho':sorted(set(role_digest(read_env(pid)) for pid in honcho)),
                  'mcp':sorted(set(role_digest(read_env(pid)) for pid in mcp))}
    need(actual_roles==EXPECTED_ROLES,'role_environment_changed')
    report['role_environment_preserved']=True
    return env"""
    script=script.replace('    return env',insertion)
    script=script.replace("version==63","version==64").replace("'schema_version':63","'schema_version':64")
    store_checks="""        from hymem.dreaming.canonicalize import find_canonical_drift
        from hymem.dreaming.evidence import count_mismatches
        need(not find_canonical_drift(conn),'canonical_drift')
        need(not count_mismatches(conn),'ledger_count_mismatch')
        conflicts=conn.execute("SELECT COUNT(*) FROM (SELECT edge_id,source_session_id,source_message_id,evidence_kind,prompt_generation,phase1_generation_key FROM kg_claim_observations GROUP BY edge_id,source_session_id,source_message_id,evidence_kind,prompt_generation,phase1_generation_key HAVING MIN(polarity)<>MAX(polarity) OR MIN(interpretation_key)<>MAX(interpretation_key))").fetchone()[0]
        need(conflicts==0,'same_generation_conflict')
        vectors=strict_episode_vectors(conn,db)
        need(vectors['verifiable'] is True and vectors['aligned'] is True,'episode_vector_alignment')
        report['episode_vectors']=vectors
        report['store_checks']={'canonical':True,'ledger':True,'same_generation':True,'episode_vectors':True}
"""
    script=script.replace("        need(good,'database_integrity_or_version')",
                          "        need(good,'database_integrity_or_version')\n"+store_checks)
    script=script.replace('len(rows)==86','len(rows)==EXPECTED_DIST_COUNT')
    script=script.replace("'0cd1ea43c1e3541d662cb2230fd157cce2c51b84b422b3e77a67ac776971f6c3'","EXPECTED_DIST_SHA")
    script=script.replace('WRAPPER_PIN_SENTINEL','EXPECTED_WRAPPER').replace('HOOK_PIN_SENTINEL','EXPECTED_HOOK')
    wrapper=str(r.HOME/'.hermes/bin/hymem-server-wrapper')
    hook=str(r.HOME/'.agent37/hooks/post-restart.sh')
    values={'EXPECTED_FILES':files,'EXPECTED_ROLES':census['role_profile_sha256'],
            'EXPECTED_DIST_COUNT':census['processes']['distribution_count'],
            'EXPECTED_DIST_SHA':census['processes']['distribution_sha256'],
            'EXPECTED_WRAPPER':census['preserved_file_sha256'][wrapper],
            'EXPECTED_HOOK':census['preserved_file_sha256'][hook]}
    prefix=CHECKER+'\nimport hashlib,json\n'
    prefix+='def role_digest(env):return hashlib.sha256(json.dumps({k:v for k,v in env.items() if k.startswith("HYMEM_")},sort_keys=True,separators=(",",":")).encode()).hexdigest()\n'
    prefix+='\n'.join(name+'='+repr(value) for name,value in values.items())+'\n'
    result=prefix+script
    compile(result,'<v64-postdeploy>','exec')
    return result

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--config-sha256',required=True)
    args=parser.parse_args()
    cfg=r.read_sealed(args.config,args.config_sha256)
    rollout=r.Rollout(cfg,args.config_sha256)
    rollout.prior('start');rollout.inspect();rollout.preserve()
    r.verify_files(r.LIVE,rollout.files)
    script=container_script(rollout.files,rollout.census)
    raw=rollout.run(['docker','exec','hermes-1','/home/node/hymem-env/bin/python3','-I','-B','-c',script],timeout=300)
    report=json.loads(raw)
    r.need(report.get('status')=='passed' and report.get('role_environment_preserved') is True
           and report.get('source_files_verified')==481,'v64_postdeploy_failed')
    rollout.preserve()
    print(json.dumps(rollout.receipt('postdeploy',{'verification':report}),sort_keys=True))

if __name__=='__main__':
    try:main()
    except Exception:
        print(json.dumps({'status':'failed','inspect_private_evidence':True}));raise SystemExit(1)
