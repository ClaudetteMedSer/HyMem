"""Read-only, no-network census of a predeclared stock sample on Afrodite.

Selection is fixed before inspecting metadata; no question/answer/session text
or credentials are exported. This does not execute the benchmark.
"""
import json
import subprocess

IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
DATA = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json'
CODE = r'''
import hashlib,json,pathlib
import ijson
p=pathlib.Path('/data.json')
digest=hashlib.sha256()
with p.open('rb') as stream:
    for block in iter(lambda:stream.read(1048576),b''): digest.update(block)
if digest.hexdigest()!='d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442':
    raise RuntimeError('dataset_drift')
indices=[213,262,329,339,370,372,392,400]
selected=[]; count=0
with p.open('rb') as stream:
    for i,row in enumerate(ijson.items(stream,'item')):
        count+=1
        if i in indices:
            sessions=row['haystack_sessions']
            selected.append({'source_index':i,'question_id':row['question_id'],
                'question_type':row['question_type'],'sessions':len(sessions),
                'messages':sum(len(s) for s in sessions)})
if count!=500 or len(selected)!=8: raise RuntimeError('census_drift')
print(json.dumps({'schema':'lme-fixed-sample8-census-v1','seed':0,'sample':8,
    'source_indices':indices,'dataset_sha256':digest.hexdigest(),
    'source_question_count':count,'questions':selected,'provider_calls':0,
    'selection_predeclared':True,'raw_content_exported':False},sort_keys=True))
'''

if __name__ == '__main__':
    command = ['docker','run','--rm','-i','--pull','never','--network','none',
        '--user','1000:1000','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--pids-limit','64','--memory','2g','--cpus','1',
        '--mount','type=bind,src='+RUNTIME+',dst=/home/node/hymem-env,readonly',
        '--mount','type=bind,src='+DATA+',dst=/data.json,readonly',
        '--entrypoint','/home/node/hymem-env/bin/python3',IMAGE,'-I','-B','-']
    result = subprocess.run(command,input=CODE,text=True,capture_output=True,timeout=120)
    if result.returncode:
        raise SystemExit('isolated_census_failed_no_private_output_exported')
    value=json.loads(result.stdout)
    print(json.dumps(value,sort_keys=True))
