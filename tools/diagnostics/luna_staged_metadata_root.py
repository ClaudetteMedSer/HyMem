"""Hash-bound finite stage metadata; no source, quote or model text is emitted."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys

sys.dont_write_bytecode = True


def verify(root, receipt_sha, entry_sha, result_sha):
    path = root/'code/tools/diagnostics/luna_staged_run_v1.py'
    if not stat.S_ISREG(path.lstat().st_mode) or hashlib.sha256(path.read_bytes()).hexdigest() != entry_sha:
        raise ValueError('entry_pin_invalid')
    spec = importlib.util.spec_from_file_location('root_staged_finite_metadata_entry', path)
    entry = importlib.util.module_from_spec(spec); spec.loader.exec_module(entry)
    receipt, loaded, _, _ = entry.preflight(root, receipt_sha, postrun=True)
    path = root/'run/private-result.json'
    if not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_size > 300000:
        raise ValueError('result_invalid')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != result_sha:
        raise ValueError('result_pin_invalid')
    result = json.loads(raw)
    loaded[1].validate_public_result(result)
    rows = []
    for item in result['units']:
        stages = []
        for stage in item['stages']:
            stages.append(dict(stage=stage['stage'], recheck=stage['recheck'],
                original_states=stage['states'],
                positive_alternatives=[dict(index=row['index'],predicate=row['predicate'])
                    for row in stage['predicate_states'] if row['state']=='supported'],
                ambiguous_alternatives=[dict(index=row['index'],predicate=row['predicate'])
                    for row in stage['predicate_states'] if row['state']=='ambiguous']))
        rows.append({key:item[key] for key in ('ordinal','kind','index','outcome',
            'error_code','contract_code','expected_gold_match','admitted_turns',
            'known_tokens','elapsed_seconds')} | dict(stages=stages))
    return dict(schema='luna-staged-root-finite-metadata-v1', verified=True,
        result_sha256=result_sha, units=rows, new_model_calls=0)


def main():
    parser=argparse.ArgumentParser()
    for name in ('root','receipt-sha256','entry-sha256','result-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    # Same read-only boundary as the independently reviewed rehearsal/replay.
    import os,socket,subprocess
    def deny(*a,**k):raise RuntimeError('metadata_external_action_forbidden')
    socket.socket.connect=socket.socket.connect_ex=socket.create_connection=deny
    subprocess.Popen=deny
    def audit(event,values):
        if event=='open':
            mode,flags=values[1:3]
            if (type(mode) is str and any(c in mode for c in 'wax+') or
                type(flags) is int and flags & (os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC)):
                deny()
        if event in {'os.remove','os.rename','os.rmdir','os.mkdir','os.symlink','os.link',
            'os.chmod','os.chown','os.utime','os.truncate','os.system','os.fork','os.exec',
            'os.posix_spawn','subprocess.Popen','socket.connect','socket.getaddrinfo'}:deny()
    sys.addaudithook(audit)
    try:
        print(json.dumps(verify(Path(args.root),args.receipt_sha256,args.entry_sha256,args.result_sha256),sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps(dict(verified=False,new_model_calls=0,reason='finite_metadata_unverified')))
        return 1


if __name__=='__main__':raise SystemExit(main())
