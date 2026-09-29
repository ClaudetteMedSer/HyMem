"""Seal reviewed helpers with the exact accepted R5 source map, offline only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import types

R5_PIN='c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'
ROOT='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/sample8-stock-v1'
SOURCE='/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r5/candidate'
NAMES={'q1_stock_run.py','q1_stock_host.py','q1_stock_validate.py',
       'lme_q1_startup_preflight.py','supervised_invocation.py','transport_common.py','RUNBOOK.md'}
SUPPORT={'supervised_invocation.py':'9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc',
         'transport_common.py':'1c98c73cfb1b607113c726616058862fd6bb16a57901a57c8962ee4a0a2807a0'}

def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()

def load(path,name):
    module=types.ModuleType(name); module.__file__=str(path); sys.modules[name]=module
    exec(compile(path.read_bytes(),str(path),'exec'),module.__dict__)
    return module

def checked_census(census,run,util):
    util.need(type(census) is dict and set(census)=={
        'schema','dataset_sha256','source_question_count','sample','seed','source_indices',
        'questions','provider_calls','selection_predeclared','raw_content_exported'},'census_fields')
    util.need(census['schema']=='lme-fixed-sample8-census-v1'
        and census['dataset_sha256']==run.DATASET_SHA
        and type(census['source_question_count']) is int and census['source_question_count']==500
        and type(census['sample']) is int and census['sample']==8
        and type(census['seed']) is int and census['seed']==0
        and type(census['provider_calls']) is int and census['provider_calls']==0
        and census['selection_predeclared'] is True and census['raw_content_exported'] is False
        and type(census['source_indices']) is list
        and all(type(n) is int for n in census['source_indices'])
        and census['source_indices']==run.SOURCE_INDICES,'census')
    rows=census['questions']
    util.need(type(rows) is list and len(rows)==8,'census_rows')
    types_expected=['multi-session']+['temporal-reasoning']*3+['knowledge-update']*4
    projected=[]
    for row,index,qid,pair,category in zip(rows,run.SOURCE_INDICES,run.QUESTION_IDS,run.QUESTION_CENSUS,types_expected):
        util.need(type(row) is dict and set(row)=={
            'source_index','question_id','question_type','sessions','messages'},'census_row_fields')
        util.need(type(row['source_index']) is int and row['source_index']==index
            and type(row['question_id']) is str and row['question_id']==qid
            and type(row['question_type']) is str and row['question_type']==category
            and type(row['sessions']) is int and type(row['messages']) is int
            and (row['sessions'],row['messages'])==pair,'census_row')
        projected.append({key:row[key] for key in
            ('source_index','question_id','question_type','sessions','messages')})
    return projected

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tree',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--census',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    base=Path(__file__).resolve().parent
    util=load(base.parent/'lme_stock_q1/seal_successor.py','sample8_seal_util')
    raw=util.regular_bytes(args.manifest)
    util.need(sha(raw)==R5_PIN,'r5_manifest')
    original=util.decode(raw)
    expected={}
    for field in ('source_sha256','test_sha256','auxiliary_sha256'):
        values=util.mapping(original[field]); util.need(not(set(values)&set(expected)),'overlap')
        expected.update(values)
    util.exact_tree(args.tree,expected)
    bundle=base/'bundle'
    util.need({p.name for p in bundle.iterdir()}==NAMES,'helper_inventory')
    helpers={name:util.regular_bytes(bundle/name) for name in NAMES}
    util.need(all(sha(helpers[n])==pin for n,pin in SUPPORT.items()),'support_drift')
    run=load(bundle/'q1_stock_run.py','sample8_seal_runner')
    host=load(bundle/'q1_stock_host.py','sample8_seal_host')
    selector=run.selector_from_source(args.tree/'benchmarks/longmemeval_adapter.py')
    util.need(selector(list(range(500)),sample=8,seed=0)==run.SOURCE_INDICES,'selector')
    census=util.decode(util.regular_bytes(args.census))
    questions=checked_census(census,run,util)
    manifest={'schema':'stock-lme-sample8-development-run-preparation-v1',
        'source_revision':'r5','approved_source_manifest_sha256':R5_PIN,
        'source_sha256':original['source_sha256'],'source_files':len(original['source_sha256']),
        'source_mapping_sha256':sha(run.canonical(original['source_sha256'])),
        'helper_sha256':{n:sha(b) for n,b in helpers.items()},
        'dataset_sha256':run.DATASET_SHA,'sample':8,'seed':0,
        'source_indices':run.SOURCE_INDICES,'question_ids':run.QUESTION_IDS,
        'questions':questions,'sessions':sum(q['sessions'] for q in questions),
        'messages':sum(q['messages'] for q in questions),'source_question_count':500,
        'selector_sha256':run.SELECTOR_SHA,'stock_arguments':run.stock_arguments(),
        'supervision_seconds':run.TIMEOUT,'cleanup_seconds':10,
        'requested_model':run.MODEL,'endpoint':run.ENDPOINT,'thinking':'disabled',
        'docker_image':host.IMAGE,'remote_run_root':ROOT,'remote_source':SOURCE,
        'global_paid_call_cap':None,'rerolls_allowed':False,'resume_allowed':False,
        'raw_content_exported':False,'production_memory_mounted':False,
        'production_changes':False,'paid_calls_by_preparation':0}
    host.command(ROOT,SOURCE,'0'*64)
    util.need(args.output.is_absolute() and args.output.resolve()==args.output
              and not args.output.is_relative_to(args.tree) and not args.output.is_relative_to(base),'output_scope')
    args.output.mkdir(mode=0o700)
    for name,body in helpers.items(): util.exclusive(args.output/'bundle'/name,body)
    manifest_raw=encoded(manifest)
    util.exclusive(args.output/'bundle/manifest.json',manifest_raw)
    files={'bundle/'+name:sha(body) for name,body in helpers.items()}
    files['bundle/manifest.json']=sha(manifest_raw)
    util.exact_tree(args.output,files)
    archive=args.output.parent/(args.output.name+'.tgz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream,mode='w:gz',format=tarfile.USTAR_FORMAT) as tar:
            for name in sorted(files):
                tar.add(args.output/name,arcname=name,recursive=False)
    receipt={'schema':'lme-sample8-seal-v1','manifest_sha256':sha(manifest_raw),
        'archive_sha256':sha(archive.read_bytes()),'files_sha256':files,
        'source_manifest_sha256':R5_PIN,'source_files':len(original['source_sha256']),
        'output':str(args.output),'archive':str(archive),'paid_calls':0}
    util.exclusive(args.output.parent/(args.output.name+'-seal.json'),encoded(receipt))
    print(json.dumps(receipt,sort_keys=True))

if __name__=='__main__': main()
