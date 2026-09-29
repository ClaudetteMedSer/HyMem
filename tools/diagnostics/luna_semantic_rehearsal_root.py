"""Root's read-only, zero-inference rehearsal of the sealed semantic probe.

Run by SSH stdin with explicit entry/receipt pins. Retained text stays on host.
Every judgment is synthetic: this verifies wiring, not model accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys


def verify(root: Path, receipt_sha: str, entry_sha: str) -> dict:
    entry_path = root / 'code/tools/diagnostics/luna_semantic_probe_run.py'
    if not stat.S_ISREG(entry_path.lstat().st_mode) or hashlib.sha256(entry_path.read_bytes()).hexdigest() != entry_sha:
        raise ValueError('entry_pin_invalid')
    spec = importlib.util.spec_from_file_location('root_semantic_rehearsal_entry',entry_path)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    receipt,loaded,proof,retained = entry.preflight(root,receipt_sha)
    host,core,cases,grounding,gold,chunk,semantic,stage,warm,concurrent = loaded
    if (root/'launch-attempt.json').exists() or (root/'run').exists():
        raise ValueError('not_an_unlaunched_rehearsal')
    closed=[]

    class SyntheticJudge:
        def __init__(self,key,cap,budget):
            self.key,self.budget=key,budget
            self.case=None if key=='canary' else cases.cases()[int(key.split('-')[-1])]
            budget.register(key,warm.BudgetLimits(*cap))
        @property
        def observed_turns(self): return self.budget.snapshot()['questions'][self.key]['turns']
        @property
        def observed_tokens(self): return self.budget.snapshot()['questions'][self.key]['known_tokens']
        @property
        def usage_complete(self): return self.budget.snapshot()['questions'][self.key]['usage_complete']
        def complete(self,request):
            wire=json.loads(request.user)
            self.budget.reserve(self.key)
            self.budget.before_turn(self.key,dict(auth='chatgpt',model=warm.base.MODEL,
                config_isolation_admitted=True,inference_enabled=False,
                quota_windows=[dict(remaining_percent=90)]))
            # The real budget requires positive settled usage. This is an
            # invented accounting sentinel, never observed provider usage.
            self.budget.settle(self.key,used=7,turn_started=True)
            verdicts=[]
            for index,item in enumerate(wire['batch']['candidates']):
                if self.case is not None:
                    label=self.case.expected[index]
                    status='supported' if self.observed_turns>1 else sorted(label.statuses)[0]
                    predicate=(label.predicate if status=='replace_predicate' else
                               item['predicate'] if status=='supported' else None)
                    evidence=[dict(source_message_id=item['source_message_id'],region=region,quote=quote)
                              for region,quote in label.evidence] if predicate is not None else []
                else:
                    i=0 if item['source_message_id']==gold._CANARY_EXPECTED_CLAIMS[0][6] else 1
                    predicate=gold._CANARY_EXPECTED_CLAIMS[i][2]
                    status='supported' if item['predicate']==predicate else 'replace_predicate'
                    evidence=[dict(source_message_id=item['source_message_id'],region='owned',
                                   quote=gold._TABLE_CLAIM_ROW if i==0 else gold._PROSE_BOUNDARY_RIGHT)]
                verdicts.append(dict(index=index,status=status,predicate=predicate,evidence=evidence))
            return json.dumps(dict(schema='source-grounding-v1',batch_sha256=wire['batch_sha256'],
                                   complete=True,verdicts=verdicts))
        def close(self): closed.append(self.key)

    class MemoryJournal:
        def record(self,*args): pass

    out=core.run_campaign(concurrent=concurrent,warm=warm,binary='offline-not-executable',
        cases_module=cases,grounding_module=grounding,journal=MemoryJournal(),retained=retained,
        canary_module=gold,chunk_module=chunk,semantic_module=semantic,stage_module=stage,
        candidate=root/'candidate',client_factory=SyntheticJudge)
    if not(out['core_completed'] and out['all_semantic_checks_passed'] and
           out['paid_budget']['turns']==29 and out['paid_budget']['known_tokens']==203 and
           out['hybrid']['replayed_ordinary_calls']==8 and
           out['hybrid']['new_paid_grounding_calls']==3 and
           len(closed)==len(set(closed))==25 and not(root/'run').exists()):
        raise ValueError('rehearsal_failed')
    return {'schema':'luna-semantic-root-rehearsal-v1','verified':True,
        'receipt_sha256':receipt_sha,'candidate_files':proof['candidate_files'],
        'synthetic_judgments':29,'synthetic_token_sentinels':203,
        'retained_ordinary_replays':8,'model_calls':0,
        'files_written':0,'semantic_model_accuracy_measured':False}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--entry-sha256',required=True)
    args=parser.parse_args()
    def deny(*a,**k): raise RuntimeError('offline_rehearsal_forbids_external_action')
    socket.socket.connect=deny
    socket.socket.connect_ex=deny
    socket.create_connection=deny
    subprocess.Popen=deny
    def audit(event,args):
        if event=='open':
            mode=args[1]
            flags=args[2]
            if ((type(mode) is str and any(c in mode for c in 'wax+')) or
                    type(flags) is int and flags &
                    (os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC)):
                deny()
        if event in {'os.remove','os.rename','os.rmdir','os.mkdir','os.symlink',
                     'os.link','os.chmod','os.chown','os.utime','os.truncate',
                     'os.system','os.fork','os.exec','os.posix_spawn',
                     'subprocess.Popen','socket.connect','socket.getaddrinfo'}:
            deny()
    sys.addaudithook(audit)
    try:
        print(json.dumps(verify(Path(args.root),args.receipt_sha256,args.entry_sha256),sort_keys=True))
    except BaseException:
        print(json.dumps({'verified':False,'model_calls':0,'reason':'root_offline_rehearsal_failed'}))
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
