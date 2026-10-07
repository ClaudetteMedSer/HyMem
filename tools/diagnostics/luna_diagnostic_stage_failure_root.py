"""Root read-only finite call-site census for the failed 256-task pilot.

Logs stay on Afrodite. Only source/receipt-verified status, fixed event counts
and source-defined function labels are exported; no raw text or private rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

READER_SHA = "91d2388a2ec558ac49e040b7bd5e2378c0c9abd1aab6b993d785bd7838f34242"
READER = Path(__file__).with_name("luna_lme_diagnostic_progress_v3.py")
REMOTE = r'''
import ast,json,re
namespace={'__name__':'reviewed_reader','__file__':'<reviewed-reader>'}
exec(compile(SOURCE,'<reviewed-reader>','exec'),namespace)
root=namespace['Path']('/home/atta/.hymem-lme-diagnostic-preflight-h0nfxj0v')
receipt_sha='8905ddc19dc381cb95ad74250b06dc3a924809f5de92094c3653365972aed771'
report=namespace['inspect'](root,receipt_sha)
if report['status']!='terminal_incomplete_or_unclean' or not report['runtime_cleanup_verified']:
    raise ValueError('terminal_not_verified')
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
        'RuntimeError: stage_accounting_failure')
output={'schema':'luna-diagnostic-stage-failure-root-v1',
        'terminal_and_cleanup_verified':True,'logs':{}}
for name in ('private-diagnostic-run.log','private-launch-stderr.log'):
    path=root/name
    if not path.exists():
        output['logs'][name]={'present':False}
        continue
    if not namespace['_file'](path,root,1048576):
        raise ValueError('log_bound_invalid')
    text=path.read_text(errors='replace')
    observed={}
    for match in re.finditer(r'File "([^"\n]+)", line [0-9]+, in ([A-Za-z0-9_]+)',text):
        path_value,func=match.groups()
        if func in allowed.get(path_value,set()):
            label=path_value.removeprefix(str(root/'candidate')+'/')+':'+func
            observed[label]=observed.get(label,0)+1
    output['logs'][name]={'present':True,'bytes':path.stat().st_size,
        'fixed_event_counts':{event:text.count(event) for event in events},
        'source_function_counts':observed}
print(json.dumps(output,sort_keys=True))
'''


def main() -> int:
    source = READER.read_bytes()
    if hashlib.sha256(source).hexdigest() != READER_SHA:
        raise SystemExit("reader_pin_mismatch")
    payload = "SOURCE=" + repr(source.decode()) + "\n" + REMOTE
    result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
        "-o", "ConnectionAttempts=1", "afrodite", "/usr/bin/python3 -I -B -"],
        input=payload, text=True, capture_output=True, timeout=30)
    try:
        value = json.loads(result.stdout)
        if value.get("schema") != "luna-diagnostic-stage-failure-root-v1":
            raise ValueError("shape")
        print(json.dumps(value, sort_keys=True))
    except (ValueError, TypeError, AttributeError):
        print(json.dumps({"schema": "luna-diagnostic-stage-failure-root-v1",
            "status": "metadata_unavailable", "returncode": result.returncode}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
