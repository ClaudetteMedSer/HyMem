"""Finite-only diagnosis of the stopped V2 semantic campaign. Read-only."""
import ast
import hashlib
import json
from pathlib import Path

root=Path('/home/atta/.hymem-luna-semantic-policy-v2-hsgdr8yy')
path=root/'run/private-result.json'
assert not path.is_symlink() and hashlib.sha256(path.read_bytes()).hexdigest()=='7b0ad987a235288d4af84db14aa50cc726de41e853de916ca1b8feb95c66c458'
result=json.loads(path.read_bytes())
grounding=root/'candidate/hymem/extraction/grounding.py'
assert hashlib.sha256(grounding.read_bytes()).hexdigest()=='377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec'
codes={node.args[0].value for node in ast.walk(ast.parse(grounding.read_bytes()))
       if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='_fail'
       and node.args and isinstance(node.args[0],ast.Constant) and type(node.args[0].value) is str}
statuses={'supported','unsupported','uncertain','replace_predicate'}
outcomes={'passed','false_support','false_rejection','missed_recovery','malformed','recheck_failed'}
failures=[]
for i,control in enumerate(result['control_results']):
    assert control['outcome'] in outcomes
    if control['passed']:
        continue
    code=control['malformed_code']
    assert code is None or code in codes
    assert all(s in statuses for s in control['initial_statuses'])
    failures.append({'control_index':i,'outcome':control['outcome'],
        'malformed_code':code,'initial_statuses':control['initial_statuses'],
        'new_calls':control['new_calls']})
hybrid=[]
for file in sorted((root/'run/private-journal').iterdir()):
    if not file.name.endswith('-canary.json'):
        continue
    event=json.loads(file.read_bytes())
    phase=event.get('phase')
    if phase not in {'table_initial_returned','prose_initial_returned','prose_recheck_returned'}:
        continue
    try:
        parsed=json.loads(event['response'])
        verdicts=parsed['verdicts']
        actual=[v.get('status') if v.get('status') in statuses else 'invalid' for v in verdicts]
        predicates=[v.get('predicate') for v in verdicts]
        hybrid.append({'phase':phase,'statuses':actual,
            'predicate_is_prefers':[p=='prefers' for p in predicates],
            'predicate_is_uses':[p=='uses' for p in predicates]})
    except (ValueError,KeyError,TypeError):
        hybrid.append({'phase':phase,'malformed':True})
print(json.dumps({'control_failures':failures,'hybrid_verdicts':hybrid},sort_keys=True))

