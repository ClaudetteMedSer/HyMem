"""Read-only structural probe of the retained LME failure; no source text exits.

Run on Afrodite via stdin. The container has no network or credentials and all
mounts are read-only. Reconstructs source through the genuine coverage validator.
"""
from __future__ import annotations

import json
import subprocess

BASE = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ'
IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
CODE = r'''
import collections,hashlib,json,pathlib,re,sqlite3,sys
sys.path.insert(0,'/candidate')
from hymem.dreaming.phase1 import _claim_sources_for_chunk,_claim_source_record
from hymem.dreaming.chunks import Chunk
from hymem.extraction import chunk as c
from hymem.core.db import register_read_authority_functions
p=pathlib.Path('/reference/hymem.sqlite')
files=[p]+[pathlib.Path(str(p)+suffix) for suffix in ('-wal','-shm') if pathlib.Path(str(p)+suffix).exists()]
before={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
conn=sqlite3.connect(p.as_uri()+'?mode=ro'+('&immutable=1' if len(files)==1 else ''),uri=True,isolation_level=None)
conn.row_factory=sqlite3.Row
conn.execute('PRAGMA query_only=ON'); register_read_authority_functions(conn)
cid='chk_d9c7f58f71d38ae508b690115183b0000dd34416'
row=conn.execute('SELECT * FROM chunks WHERE id=?',(cid,)).fetchone()
assert row is not None
item=Chunk(**{k:row[k] for k in ('id','session_id','start_message_id','end_message_id','salience_reason','text')})
sources=_claim_sources_for_chunk(conn,item)
assert sources
records=tuple((source.message_id,_claim_source_record(source)) for source in sources)
root=c._ExtractionUnit(text='\n'.join(encoded for _,encoded in records),source_records=records)
nodes=[]
def structure(current,depth):
    if len(nodes)>=100: raise RuntimeError('probe_tree_limit')
    split=c._split_unit(current) if len(current.text)>c._MAX_LEAF_INPUT_CHARS else None
    node={'depth':depth,'encoded_chars':len(current.text),'owned_records':len(current.source_records or ()),
          'context_records':len(current.context_records),'split_succeeded':split is not None,
          'bounded_whole_allowed':c._bounded_whole_unit_allowed(current,depth),'records':[]}
    for mid,encoded in current.source_records or ():
        payload=c._source_payload((mid,encoded)); assert payload is not None
        content=payload['content']; blocks=c._markdown_block_analysis(content)
        record={'message_id':mid,'content_chars':len(content),'encoded_chars':len(encoded),
                'content_sha256':hashlib.sha256(content.encode()).hexdigest(),
                'content_start':payload.get('source_content_start',0),
                'newlines':content.count('\n'),'paragraph_boundaries':len(list(c._PARAGRAPH_BOUNDARY_RE.finditer(content))),
                'sentence_boundaries':len(c._sentence_boundary_points(content)),
                'protected_spans':[list(span) for span in blocks.protected_spans],
                'structural_boundaries':list(blocks.structural_boundaries),
                'table_context_boundaries':list(blocks.contextual_table_boundaries),
                'has_fragment_context':payload.get('source_fragment_context') is not None,
                'has_boundary_context':payload.get('source_boundary_context') is not None,
                'line_shapes':{'fences':len(re.findall(r'(?m)^\s*(?:```|~~~)',content)),
                               'lists':len(re.findall(r'(?m)^\s*(?:[-+*]|\d+[.)])\s',content)),
                               'headings':len(re.findall(r'(?m)^\s*#{1,6}\s',content)),
                               'pipes':content.count('|')}}
        record['semantic_cut']=c._semantic_split_point(content,trusted_table_boundaries=current.trusted_table_boundaries)
        if len(content)>8000 and record['semantic_cut'] is None:
            def shape(line):
                # Only counts/classes and punctuation: never words or digits.
                encoded=[]
                for char in line:
                    kind='L' if char.isalpha() else 'N' if char.isdigit() else char if char in ' \t{}[]()<>=:;,./\\\"\x27_-+*#|%!?@&$' else 'O'
                    if encoded and encoded[-1][0]==kind: encoded[-1][1]+=1
                    else: encoded.append([kind,1])
                return json.dumps(encoded,separators=(',',':'))
            shapes=collections.Counter(shape(line) for line in content.splitlines())
            record['line_shape_counts']=[{'shape':json.loads(value),'count':count} for value,count in shapes.most_common(30)]
            record['line_lengths']=sorted(collections.Counter(len(line) for line in content.splitlines()).items())
            record['punctuation_counts']={char:content.count(char) for char in '{}[]()<>=:;,./\\\"\x27_-+*#|%!?@&$\t'}
        node['records'].append(record)
    nodes.append(node)
    if split is not None and depth<c._MAX_SPLIT_DEPTH:
        structure(split[0],depth+1); structure(split[1],depth+1)
structure(root,0)
leaves,failure=c._prepartition(root)
class NeverLLM:
    def complete(self,*a,**k): raise AssertionError('unexpected_paid_path')
result=c.extract_chunk(NeverLLM(),item.text,source_records=records)
assert result.failed and result.failure_reason=='resource_limit' and result.completion_calls==0
assert failure is not None and result.failure_details==failure.failure_details
conn.close()
assert before=={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files}
print(json.dumps({'schema':'r6-retained-split-offline-probe-v1','chunk_id':cid,
 'canonical_sources_verified':True,'store_files_unchanged':True,'store_file_sha256':before,
 'source_records_sha256':hashlib.sha256(root.text.encode()).hexdigest(),
 'source_messages':len(records),'failure_reason':result.failure_reason,
 'failure_details':list(result.failure_details),'completion_calls':result.completion_calls,
 'provider_attempts':result.provider_attempts,'nodes':nodes},sort_keys=True))
'''


def main():
    name = 'hymem-r6-retained-split-probe-v2'
    command = ['docker', 'run', '--name', name, '-i', '--pull', 'never', '--network', 'none',
        '--user', '1000:1000', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--pids-limit', '64', '--memory', '2g', '--cpus', '1',
        '--mount', 'type=bind,src='+RUNTIME+',dst=/home/node/hymem-env,readonly',
        '--mount', 'type=bind,src='+BASE+'/offline-r5/candidate,dst=/candidate,readonly',
        '--mount', 'type=bind,src='+BASE+'/sample8-stock-v1/live-results/stores/hymem-lme-3_kjdx3h,dst=/reference,readonly',
        '--entrypoint', '/home/node/hymem-env/bin/python3', IMAGE, '-I', '-B', '-']
    try:
        result = subprocess.run(command, input=CODE, text=True, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        subprocess.run(['docker', 'stop', '--time', '10', name], capture_output=True, timeout=20)
        raise SystemExit('probe_timeout_no_private_output_exported')
    if result.returncode:
        # Trace locations only, not the arbitrary source/exception line.
        import re
        frames = re.findall(r'File "([^"\n]+)", line (\d+)', result.stderr)
        print(json.dumps({'probe_failed': True, 'frames': frames}))
        raise SystemExit(1)
    value = json.loads(result.stdout)
    assert value['completion_calls'] == value['provider_attempts'] == 0
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()
