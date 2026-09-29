"""Metadata-only full-S census; no network or provider calls in container."""
from pathlib import Path
import runpy
import json
import subprocess

original=runpy.run_path(str(Path(__file__).with_name('lme_sample8_census.py')))
CODE=original['CODE'].replace('indices=[213,262,329,339,370,372,392,400]',
                              'indices=list(range(500))')
CODE=CODE.replace('len(selected)!=8','len(selected)!=500')
CODE=CODE.replace("'schema':'lme-fixed-sample8-census-v1','seed':0,'sample':8",
                  "'schema':'lme-full500-census-v1','seed':0,'sample':0")

if __name__=='__main__':
    command=['docker','run','--rm','-i','--pull','never','--network','none',
        '--user','1000:1000','--read-only','--cap-drop','ALL',
        '--security-opt','no-new-privileges','--pids-limit','64','--memory','2g','--cpus','1',
        '--mount','type=bind,src='+original['RUNTIME']+',dst=/home/node/hymem-env,readonly',
        '--mount','type=bind,src='+original['DATA']+',dst=/data.json,readonly',
        '--entrypoint','/home/node/hymem-env/bin/python3',original['IMAGE'],'-I','-B','-']
    result=subprocess.run(command,input=CODE,text=True,capture_output=True,timeout=120)
    if result.returncode:raise SystemExit('isolated_full500_census_failed_no_private_output_exported')
    value=json.loads(result.stdout)
    assert value['schema']=='lme-full500-census-v1' and value['sample']==0
    assert value['source_indices']==list(range(500)) and len(value['questions'])==500
    print(json.dumps(value,sort_keys=True))
