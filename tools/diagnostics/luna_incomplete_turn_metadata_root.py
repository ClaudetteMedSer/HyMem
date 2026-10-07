"""Read-only finite terminal/log census for the stopped attribution pilot.

Private content stays on Afrodite. This projects only fixed labels, numeric
ledger fields and source-defined traceback function names after identity and
independent cleanup verification. It never imports or runs benchmark code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

READER = Path(__file__).with_name("luna_lme_diagnostic_progress_v4.py")
READER_SHA = "59dfc88ccb7ce846e8d4e2ba09ce1f0ab7c8e4139a56732b72bc2d682ef67d82"
REMOTE = r'''
import ast,json,re
namespace={'__name__':'reviewed_reader','__file__':'<reviewed-reader>'}
exec(compile(SOURCE,'<reviewed-reader>','exec'),namespace)
root=namespace['Path']('/home/atta/.hymem-lme-diagnostic-preflight-a5olbwr6')
receipt_sha='305a15b3196e4bcb851a6fd39930599909429930330dc977f714642ff8d6154e'
report=namespace['inspect'](root,receipt_sha)
if (report['status']!='terminal_incomplete_or_unclean'
    or not report['runtime_cleanup_verified']
    or report['budget_stop_code']!='incomplete_turn_or_usage'):
    raise ValueError('terminal_not_verified')
terminal=namespace['_read'](root/'run'/'diagnostic-result.json',root,128000)
budget=terminal['budget']
out={'schema':'luna-incomplete-turn-metadata-root-v1',
     'terminal_and_cleanup_verified':True,'questions':{},'logs':{}}
for key in ('turns','reserved','known_tokens','in_flight'):
    value=budget.get(key)
    if type(value) is not int or not 0<=value<=1000000000:
        raise ValueError('ledger_number_invalid')
    out[key]=value
for key in ('canary','q-0000','q-0001','q-0002','q-0003'):
    value=budget.get('questions',{}).get(key)
    if type(value) is not dict:
        continue
    projected={}
    for field in ('turns','known_tokens','in_flight'):
        number=value.get(field)
        if type(number) is not int or not 0<=number<=1000000000:
            raise ValueError('question_number_invalid')
        projected[field]=number
    for field in ('usage_complete','stopped'):
        flag=value.get(field)
        if type(flag) is not bool:
            raise ValueError('question_flag_invalid')
        projected[field]=flag
    out['questions'][key]=projected
modules=('hymem/extraction/chunk.py','hymem/dreaming/digest.py',
    'hymem/dreaming/user_profile.py','hymem/dreaming/facts.py',
    'hymem/dreaming/summary_recovery.py','hymem/query/coref.py',
    'hymem/query/rerank.py','hymem/dreaming/runner.py')
allowed={}
for relative in modules:
    path=root/'candidate'/relative
    tree=ast.parse(path.read_text())
    allowed[str(path)]={node.name for node in ast.walk(tree)
                       if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))}
events=('profile.extraction_failure','facts.extraction_failure',
    'chunk_extraction.call_failure','chunk_grounding.call_failure',
    'incomplete_turn_or_usage','stage_accounting_failure','quota_floor',
    'turn_failed','protocol_failure','timeout','usage_unknown')
for name in ('private-diagnostic-run.log','private-launch-stderr.log'):
    path=root/name
    if not path.exists():
        out['logs'][name]={'present':False}
        continue
    if not namespace['_file'](path,root,1048576):
        raise ValueError('log_bound_invalid')
    content=path.read_text(errors='replace')
    observed={}
    for match in re.finditer(r'File "([^"\n]+)", line [0-9]+, in ([A-Za-z0-9_]+)',content):
        path_value,func=match.groups()
        if func in allowed.get(path_value,set()):
            label=path_value.removeprefix(str(root/'candidate')+'/')+':'+func
            observed[label]=observed.get(label,0)+1
    out['logs'][name]={'present':True,'bytes':path.stat().st_size,
        'fixed_event_counts':{event:content.count(event) for event in events},
        'source_function_counts':observed}
print(json.dumps(out,sort_keys=True))
'''


def main() -> int:
    source = READER.read_bytes()
    if hashlib.sha256(source).hexdigest() != READER_SHA:
        raise SystemExit("reader_pin_mismatch")
    result = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o",
        "ConnectionAttempts=1", "afrodite", "/usr/bin/python3 -I -B -",
    ], input="SOURCE=" + repr(source.decode()) + "\n" + REMOTE,
        text=True, capture_output=True, timeout=30)
    try:
        output = json.loads(result.stdout)
        if type(output) is not dict or output.get("schema") != "luna-incomplete-turn-metadata-root-v1":
            raise ValueError("metadata_shape")
        print(json.dumps(output, sort_keys=True))
    except (ValueError, TypeError):
        print(json.dumps({"schema": "luna-incomplete-turn-metadata-root-v1",
                          "status": "metadata_unavailable", "returncode": result.returncode}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
