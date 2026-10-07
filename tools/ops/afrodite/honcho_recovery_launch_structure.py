"""Read-only structural map of the maintained Honcho launch block.

No shell is evaluated. The container returns only line numbers, assignment
names, value kinds, variable-reference names, and fixed source hashes. It does
not return values, raw script lines, process environments, or secrets.
"""
from __future__ import annotations

import json
import shlex
import subprocess


SSH = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", "ConnectionAttempts=1", "afrodite"]

REMOTE = r"""
import hashlib,json,re,shlex,stat
from pathlib import Path

HOOK=Path('/home/node/.agent37/hooks/post-restart.sh')
WRAPPER=Path('/home/node/.hermes/bin/hymem-server-wrapper')
ENV=Path('/home/node/.hermes/.env')
EXPECTED={
    HOOK:'4bd4cc0011f2bfa298cf073d823bce7aaf3a3a1fd3b4f801c010c89a8dcaef68',
    WRAPPER:'685b198a87e22c877d562e126b0f13780e500e30383b7a3f43ae2b6b73930402',
    ENV:'0fea90464e6f2ac8b6cf8cd907e43e6535c03eeeb3492f2737369b0aae8e19de',
}
def need(ok,code):
    if not ok: raise RuntimeError(code)
def validate():
    for path,digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink(),'source_missing')
        need(hashlib.sha256(path.read_bytes()).hexdigest()==digest,'source_changed')
def classify_value(raw):
    raw=raw.rstrip()
    if raw.endswith('\\'): raw=raw[:-1].rstrip()
    refs=sorted(set(re.findall(r'\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))',raw)))
    refs=sorted(set(a or b for a,b in refs))
    if '$(' in raw or '`' in raw: kind='command_substitution'
    elif raw.startswith("'" ) and raw.endswith("'"): kind='single_quoted_literal'
    elif refs: kind='variable_reference'
    elif raw.startswith('"') and raw.endswith('"'): kind='double_quoted_literal'
    elif not raw: kind='empty'
    else: kind='literal_or_shell_word'
    parameter_ops=[]
    for match in re.finditer(r'\$\{([A-Za-z_][A-Za-z0-9_]*)(:?[-+?=]?)([^}]*)\}',raw):
        parameter_ops.append({'name':match.group(1),'operator':match.group(2),
                              'operand_present':bool(match.group(3))})
    return {'kind':kind,'references':refs,'parameter_ops':parameter_ops,'has_backslash':('\\' in raw),
            'has_semicolon':(';' in raw),'has_space':any(c.isspace() for c in raw)}
def classify_line(line,number):
    stripped=line.strip()
    if not stripped or stripped.startswith('#'): return None
    match=re.fullmatch(r'(export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)',stripped)
    if match:
        exported,key,value=match.groups()
        shape=classify_value(value)
        value_kind=shape.pop('kind')
        return {'line':number,'kind':'export_assignment' if exported else 'assignment',
                'name':key,'value_kind':value_kind,**shape}
    # The remaining categories expose only syntax, never shell words/paths.
    if re.match(r'^(?:source|\.)\s+',stripped): kind='source'
    elif re.search(r'\bnohup\b',stripped): kind='nohup'
    elif re.match(r'^(?:if|elif|then|else|fi|for|do|done|while|case|esac)\b',stripped): kind='control'
    elif re.match(r'^export\s+',stripped): kind='export_other'
    elif re.search(r'hymem-honcho|hymem\.honcho',stripped): kind='honcho_invocation'
    elif re.match(r'^\w+\s*\(\)',stripped): kind='function_definition'
    else: kind='other_shell'
    refs=sorted(set(a or b for a,b in re.findall(
        r'\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))',stripped)))
    return {'line':number,'kind':kind,'references':refs,
            'has_redirection':('>' in stripped or '<' in stripped),
            'background':stripped.endswith('&'),'has_command_substitution':('$(' in stripped or '`' in stripped)}
def structure(path,first,last):
    lines=path.read_text().splitlines()
    return [item for i,line in enumerate(lines,1) if first<=i<=last
            if (item:=classify_line(line,i)) is not None]
def launch_token_shape(line,number):
    stripped=line.strip()
    continuation=stripped.endswith('\\')
    if continuation: stripped=stripped[:-1].rstrip()
    try:
        lex=shlex.shlex(stripped,posix=True)
        lex.whitespace_split=True
        lex.commenters='#'
        tokens=list(lex)
        syntax_ok=True
    except ValueError:
        tokens=[]
        syntax_ok=False
    labels=[]
    for token in tokens:
        if token in ('nohup','env','exec','setsid','disown','&'):
            label=token
        elif token=='/home/node/hymem-env/bin/hymem-honcho':
            label='known_console'
        elif token=='/home/node/hymem-env/bin/python3':
            label='known_venv_python'
        elif re.fullmatch(r'[A-Z][A-Z0-9_]*=.*',token):
            label='assignment'
        elif re.fullmatch(r'(?:[0-9]*>>?|[0-9]*<|[0-9]*>&[0-9]+)',token):
            label='redirection_operator'
        elif token.startswith('-'):
            label='option'
        else:
            label='other'
        labels.append(label)
    return {'line':number,'syntax_ok':syntax_ok,'token_count':len(tokens),
        'token_labels':labels,'assignment_names':[
            t.split('=',1)[0] for t,l in zip(tokens,labels) if l=='assignment'],
        'continuation':continuation,'has_comment':('#' in line),
        'ends_background':stripped.endswith('&')}

validate()
env_names=[]
for i,line in enumerate(ENV.read_text().splitlines(),1):
    item=classify_line(line,i)
    if item and item.get('kind') in ('assignment','export_assignment'):
        env_names.append({'name':item['name'],'kind':item['kind'],
                          'value_kind':item['value_kind']})
print(json.dumps({'action':'launch-structure',
    'hook_prelude':structure(HOOK,1,50),
    'hook_honcho_block':structure(HOOK,65,100),
    'launch_token_shape':[launch_token_shape(line,i) for i,line in enumerate(HOOK.read_text().splitlines(),1)
                          if i in (77,92,93,94)],
    'wrapper':structure(WRAPPER,1,200),
    'env_assignment_names':env_names,
    'source_sha256':{'hook':EXPECTED[HOOK],'wrapper':EXPECTED[WRAPPER],'env':EXPECTED[ENV]}},sort_keys=True))
"""


def main() -> int:
    command = SSH + [shlex.join(["docker", "exec", "-i", "-u", "node", "hermes-1",
                                 "/usr/bin/python3.11", "-B", "-c", REMOTE])]
    try:
        result = subprocess.run(command, input="", text=True, capture_output=True,
                                timeout=25)
    except subprocess.TimeoutExpired:
        print(json.dumps({"error": "inspection_timeout"}))
        return 1
    if result.returncode:
        print(json.dumps({"error": "inspection_failed", "returncode": result.returncode}))
        return 1
    try: value = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(json.dumps({"error": "invalid_metadata"}))
        return 1
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
