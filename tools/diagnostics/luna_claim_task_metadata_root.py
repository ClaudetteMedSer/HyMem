"""Finite metadata only from the completed, independently replayed A/B run."""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

ROOT=Path('/home/atta/.hymem-luna-claim-task-probe-oo85_tyi')
RECEIPT='8ed35fb94e77bc731d2d7b3b98fa0a33361175ff8c2104d02145eb241c1e0ce1'
ENTRY='975d30fd4f7787b65d31b0eef08df6483dd6ba1cd1e291e84a32ebe722793ebe'
RESULT='46eb87cf31c4c34e713c542f86e4f5d88af825283eedf24f218697907523c461'


def read(path,sha):
    assert stat.S_ISREG(path.lstat().st_mode)
    raw=path.read_bytes(); assert hashlib.sha256(raw).hexdigest()==sha
    return raw


def main():
    path=ROOT/'code/tools/diagnostics/luna_claim_task_run_v1.py'
    read(path,ENTRY)
    spec=importlib.util.spec_from_file_location('claim_task_metadata_entry',path)
    entry=importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    receipt,loaded,proof,retained=entry.preflight(ROOT,RECEIPT,True)
    host,core,old,cases,gold,chunk,semantic,transport,warm,concurrent=loaded
    result=json.loads(read(ROOT/'run/private-result.json',RESULT))
    core.validate_public_result(result)
    assert result['diagnostic_completed'] and len(result['units'])==29
    terminal=json.loads((ROOT/'safe-terminal.json').read_bytes())
    elapsed=terminal['elapsed_seconds']; assert type(elapsed) in (int,float) and 0<=elapsed<=2000
    batches=core.derive_canary_batches(canary_module=gold,chunk_module=chunk,
        semantic_probe_module=old,candidate=ROOT/'candidate',retained=retained,record=lambda e:None)
    codes=set()
    for module in (core.contract,core.contract.v3,core.contract.v2):
        tree=ast.parse(Path(module.__file__).read_bytes())
        codes.update(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and
            type(n.value) is str and re.fullmatch('[a-z_]+:[a-z_]+',n.value))
    records=[]
    for item,unit in zip(result['units'],core.schedule(),strict=True):
        out={k:item[k] for k in ('ordinal','kind','index','arm','label','scope_ambiguous','outcome','known_tokens')}
        if item['result'] is not None:
            out['original_states']=[s['state'] for s in item['result']['original']]
            out['final_verdicts']=item['result']['final_verdicts']
            out['nonnegative_alternatives']=None if item['arm']=='B' else [
                None if row is None else [p for p in row if p['state']!='not_established']
                for row in item['result']['alternative_states']]
        else:
            events=[json.loads(p.read_bytes()) for p in sorted((ROOT/'run/private-journal').glob(f'*unit-{item["ordinal"]:02d}.json'))]
            raw=next(e['response'] for e in events if e.get('phase')=='response_returned')
            triples,sources,_=core._unit_input(unit,cases.cases(),batches)
            request,batch=core.contract.build_arm_request(unit.arm,triples,sources)
            try: core.contract.parse_arm_response(unit.arm,raw,batch)
            except core.contract.v2.GroundingContractError as exc:
                out['failure_code']=str(exc) if str(exc) in codes else 'unclassified_contract_failure'
            else: raise AssertionError('malformed_changed')
            try: wire=json.loads(raw)
            except ValueError: wire={}
            rows=wire.get('classifications',[]) if type(wire) is dict else []
            states=[]
            if type(rows) is list and len(rows)<=8:
                for row in rows:
                    state=row.get('original',{}).get('state') if type(row) is dict and type(row.get('original')) is dict else None
                    states.append(state if state in {'supported','ambiguous','not_established'} else 'invalid')
            out['unvalidated_original_states']=states
        records.append(out)
    print(json.dumps(dict(verified=True,result_sha256=RESULT,units=records,
        elapsed_seconds=elapsed,paid_budget=result['paid_budget'],model_calls=0,
        semantic_accuracy_accepted=False,full_lme_ready=False),sort_keys=True))


if __name__=='__main__':
    sys.dont_write_bytecode=True
    def deny(*a,**k): raise RuntimeError('metadata_external_action_forbidden')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=deny
    subprocess.Popen=deny
    def audit(event,args):
        if event=='open':
            mode,flags=args[1:3]
            if (type(mode) is str and any(c in mode for c in 'wax+') or
                type(flags) is int and flags & (os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC)): deny()
        if event in {'os.remove','os.rename','os.rmdir','os.mkdir','os.symlink','os.link','os.chmod',
            'os.chown','os.utime','os.truncate','os.system','os.fork','os.exec','os.posix_spawn',
            'subprocess.Popen','socket.connect','socket.getaddrinfo'}: deny()
    sys.addaudithook(audit)
    try: main()
    except BaseException:
        print(json.dumps(dict(verified=False,model_calls=0,reason='finite_metadata_failed')))
        raise SystemExit(1)
