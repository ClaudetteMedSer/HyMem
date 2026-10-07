"""Bound v3 failure localization; emits only finite fixture/state metadata."""
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys


def run():
    root = Path('/home/atta/.hymem-luna-classification-v3-probe-4ipv54t4')
    result_path = root / 'run/private-result.json'
    assert result_path.is_file() and not result_path.is_symlink()
    blob = result_path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == '8411ccaa86974a0b271091bd709043f30cef2454bc3f414f281c4fc0f6489ea6'
    result = json.loads(blob)
    states = {'supported','unsupported','uncertain','replace_predicate'}
    assessment_states = {'supported','not_established','ambiguous'}
    predicates = {'uses','depends_on','prefers','rejects','avoids','replaces','conflicts_with',
        'deploys_to','part_of','equivalent_to','implements','contains','configured_with',
        'requires_version','runs_on','connects_to','generates','tested_by','owns','located_in',
        'participates_in','has_attribute'}
    outcomes = {'passed','false_support','false_rejection','missed_recovery','recheck_failed','malformed'}
    phases = {'initial_returned','recheck_returned','table_initial_returned',
              'prose_initial_returned','prose_recheck_returned'}
    controls = result['control_results']
    assert len(controls) == 24
    failures = []
    for i,c in enumerate(controls):
        assert c['outcome'] in outcomes and len(c['initial_statuses']) <= 8
        assert all(s in states for s in c['initial_statuses'])
        if not c['passed']:
            failures.append({'index':i,'outcome':c['outcome'],'statuses':c['initial_statuses']})
    wanted = {f'control-{x["index"]:02d}' for x in failures} | {'canary'}
    summaries = []
    files = sorted((root/'run/private-journal').iterdir())
    assert len(files) <= 256
    for file in files:
        match = re.fullmatch(r'\d{4}-(control-\d{2}|canary)\.json',file.name)
        assert match and file.stat().st_size <= 2000000 and not file.is_symlink()
        key = match[1]
        if key not in wanted:
            continue
        event = json.loads(file.read_bytes())
        if not event.get('phase','').endswith('_returned'):
            continue
        assert event['phase'] in phases
        raw = event['response'].strip()
        fenced = re.fullmatch(r'```(?:json)?\s*\n([\s\S]*?)\n```',raw)
        payload = json.loads(fenced[1] if fenced else raw)
        assert len(payload['classifications']) <= 8
        rows = []
        for item in payload['classifications']:
            assert type(item['index']) is int and 0 <= item['index'] < 8
            state = item['original']['state']
            assert state in assessment_states
            alts = item['alternatives']
            if alts is not None:
                assert set(alts) <= predicates and all(a['state'] in assessment_states for a in alts.values())
            rows.append({'index':item['index'],'original_state':state,
                'supported_alternatives':sorted(p for p,a in (alts or {}).items() if a['state']=='supported'),
                'ambiguous_alternatives':sorted(p for p,a in (alts or {}).items() if a['state']=='ambiguous')})
        summaries.append({'unit':key,'phase':event['phase'],'rows':rows})
    return {'failed_controls':failures,'assessment_summary':summaries,
            'raw_text_exported':False,'model_calls':0}


def main():
    def deny(*args,**kwargs):
        raise RuntimeError('offline_metadata_external_action_forbidden')
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    subprocess.Popen = deny
    def audit(event,args):
        if event == 'open':
            mode,flags = args[1:3]
            if (type(mode) is str and any(c in mode for c in 'wax+') or
                    type(flags) is int and flags & (os.O_WRONLY|os.O_RDWR|os.O_APPEND|os.O_CREAT|os.O_TRUNC)):
                deny()
        if event in {'os.remove','os.rename','os.rmdir','os.mkdir','os.symlink','os.link',
            'os.chmod','os.chown','os.utime','os.truncate','os.system','os.fork','os.exec',
            'os.posix_spawn','subprocess.Popen','socket.connect','socket.getaddrinfo'}:
            deny()
    sys.addaudithook(audit)
    try:
        print(json.dumps(run(),sort_keys=True))
    except BaseException as error:
        category = ('assertion' if type(error) is AssertionError else
                    'json' if type(error) is json.JSONDecodeError else
                    'io' if isinstance(error,OSError) else 'other')
        trace = error.__traceback__
        while trace.tb_next is not None:
            trace = trace.tb_next
        print(json.dumps({'verified':False,'reason':'finite_metadata_validation_failed',
                          'category':category,'line':trace.tb_lineno,'model_calls':0}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
