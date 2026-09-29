"""Read-only private replay of captured judgments; no new model invocation."""
from __future__ import annotations
import argparse
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


def replay(root,receipt_sha,entry_sha):
    path=root/'code/tools/diagnostics/luna_semantic_probe_run.py'
    assert stat.S_ISREG(path.lstat().st_mode) and hashlib.sha256(path.read_bytes()).hexdigest()==entry_sha
    spec=importlib.util.spec_from_file_location('root_verdict_replay_entry',path)
    entry=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    receipt,loaded,proof,retained=entry.preflight(root,receipt_sha)
    host,core,cases,grounding,gold,chunk,semantic,stage,warm,concurrent=loaded
    result_path=root/'run/private-result.json'
    assert stat.S_ISREG(result_path.lstat().st_mode)
    result=json.loads(result_path.read_bytes())
    assert result['schema']=='luna-semantic-probe-v1'
    from tools.diagnostics import luna_semantic_probe_progress as progress
    terminal=json.loads((root/'safe-terminal.json').read_bytes())
    manifest=hashlib.sha256(json.dumps(receipt['source_sha256'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert progress._terminal_valid(terminal,result,receipt_sha,manifest,warm.serialize_failure)
    directory=root/'run/private-journal'
    assert directory.is_dir() and not directory.is_symlink()
    grouped={}
    files=sorted(directory.iterdir())
    assert len(files)<=256
    for i,file in enumerate(files,1):
        match=re.fullmatch(r'(\d{4})-(control-\d{2}|canary)\.json',file.name)
        assert match and int(match[1])==i and stat.S_ISREG(file.lstat().st_mode)
        assert file.stat().st_size<=2_000_000
        grouped.setdefault(match[2],[]).append(json.loads(file.read_bytes()))

    class Captured:
        usage_complete=True
        def __init__(self,events):
            self.pairs=[]
            pending=None
            for event in events:
                phase=event.get('phase','')
                if phase.endswith('_before_dispatch'):
                    assert pending is None
                    pending=(phase[:-len('_before_dispatch')],event['request'])
                elif phase.endswith('_returned'):
                    assert pending is not None and pending[0]==phase[:-len('_returned')]
                    self.pairs.append((pending[1],event['response']))
                    pending=None
            assert pending is None
            self.observed_turns=0
            # Synthetic replay sentinels satisfy the canary accounting shape;
            # never exported or treated as observed provider token usage.
            self.observed_tokens=0
        def complete(self,request):
            expected,raw=self.pairs[self.observed_turns]
            assert core.canonical(asdict(request))==core.canonical(expected)
            self.observed_turns+=1
            self.observed_tokens+=1
            return raw
        def all_consumed(self): return self.observed_turns==len(self.pairs)

    consumed=0
    control_passes=[]
    controls=result['control_results']
    assert len(controls)<=24
    for i,original in enumerate(controls):
        client=Captured(grouped[f'control-{i:02d}'])
        rebuilt=core.run_control(cases.cases()[i],client,grounding,record=lambda _:None)
        assert all(core.canonical(original[k])==core.canonical(v) for k,v in rebuilt.items())
        assert client.all_consumed()
        consumed+=client.observed_turns
        control_passes.append(rebuilt['passed'])
    hybrid_original=result['hybrid']
    hybrid_replayed=False
    if hybrid_original is not None:
        client=Captured(grouped['canary'])
        rebuilt=core.run_hybrid(canary_module=gold,chunk_module=chunk,
            semantic_module=semantic,candidate=root/'candidate',paid=client,retained=retained,
            record=lambda _:None,stage_module=stage)
        for key,value in rebuilt.items():
            if key in {'canary','paid_known_tokens'}:
                continue
            assert core.canonical(value)==core.canonical(hybrid_original[key])
        assert set(rebuilt['canary'])==set(hybrid_original['canary'])
        for key,value in rebuilt['canary'].items():
            if key!='observed_token_delta':
                assert core.canonical(value)==core.canonical(hybrid_original['canary'][key])
        assert client.all_consumed()
        consumed+=client.observed_turns
        hybrid_replayed=True
    returned=sum(1 for events in grouped.values() for event in events
                 if event.get('phase','').endswith('_returned'))
    assert consumed==returned
    semantic_pass=(len(control_passes)==24 and all(control_passes) and
                   hybrid_replayed and rebuilt['passed'])
    assert result['all_semantic_checks_passed'] is semantic_pass
    return {'schema':'luna-semantic-verdict-root-replay-v1','verified':True,
        'receipt_sha256':receipt_sha,'private_result_sha256':hashlib.sha256(result_path.read_bytes()).hexdigest(),
        'replayed_control_units':len(controls),'hybrid_replayed':hybrid_replayed,
        'captured_returned_judgments_replayed':consumed,'new_model_calls':0,
        'all_semantic_checks_passed':semantic_pass,
        'core_completed':result['core_completed'],'cleanup_certified_here':False,
        'token_usage_source':'validated_terminal_not_synthetic_replay'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--receipt-sha256',required=True)
    parser.add_argument('--entry-sha256',required=True)
    args=parser.parse_args()
    def deny(*a,**k): raise RuntimeError('replay_external_action_forbidden')
    socket.socket.connect=deny
    socket.socket.connect_ex=deny
    socket.create_connection=deny
    subprocess.Popen=deny
    def audit(event,args):
        if event=='open':
            mode,flags=args[1:3]
            if (type(mode) is str and any(c in mode for c in 'wax+') or
                    type(flags) is int and flags & (os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC)):
                deny()
        if event in {'os.remove','os.rename','os.rmdir','os.mkdir','os.symlink','os.link',
                     'os.chmod','os.chown','os.utime','os.truncate','os.system','os.fork',
                     'os.exec','os.posix_spawn','subprocess.Popen','socket.connect','socket.getaddrinfo'}:
            deny()
    sys.addaudithook(audit)
    try:
        print(json.dumps(replay(Path(args.root),args.receipt_sha256,args.entry_sha256),sort_keys=True))
    except BaseException:
        print(json.dumps({'verified':False,'new_model_calls':0,'reason':'offline_verdict_replay_failed'}))
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
