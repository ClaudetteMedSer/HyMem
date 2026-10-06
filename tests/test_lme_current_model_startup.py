"""Exercise actual CLI checkpoint setup without a provider call."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("ijson", reason="LongMemEval CLI requires the optional ijson parser")

STARTUP = r'''
import json, os, pathlib, runpy, sys
source, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
explicit = sys.argv[3] == 'yes'
os.environ.clear()
os.environ.update(HOME=str(work), TMPDIR=str(work), PATH='/usr/bin:/bin',
                  PYTHONDONTWRITEBYTECODE='1', DEEPSEEK_API_KEY='synthetic-no-network-key')
connections = []
def audit(event, args):
    if event in {'socket.connect', 'socket.getaddrinfo', 'socket.sendto'}:
        connections.append(event)
        raise AssertionError('startup attempted network I/O')
sys.addaudithook(audit)
sys.path.insert(0, str(source))
class ClientBoundary(BaseException):
    pass
adapter = source / 'benchmarks' / 'longmemeval_adapter.py'
def boundary(frame, event, _arg):
    if event == 'call' and frame.f_code.co_filename == str(adapter):
        client = frame.f_globals.get('LLMClient')
        if client is not None and frame.f_code is client.__init__.__code__:
            raise ClientBoundary()
    return boundary
sys.argv = [str(adapter), '--scales', 'S', '--sample', '1', '--seed', '53',
            '--workers', '1', '--top-k', '15', '--auto-ability', '--permissive-default',
            '--no-prereg', '--protocol-split', 'full', '--indexing-max-cycles', '100',
            '--indexing-timeout-s', '3600', '--indexing-require-healthy', '--keep-db',
            '--judge-protocol', 'legacy-custom', '--data-dir', str(work),
            '--results-dir', str(work / 'results'),
            '--checkpoint', str(work / 'results' / 'checkpoint.json')]
if explicit:
    for role in ('hymem', 'answer', 'judge'):
        sys.argv.extend(['--' + role + '-model', 'deepseek-flash',
                         '--' + role + '-base-url', 'https://api.deepseek.com'])
    sys.argv.extend(['--hymem-thinking', 'disabled', '--answer-extra-body',
                     '{"thinking":{"type":"disabled"}}', '--judge-extra-body',
                     '{"thinking":{"type":"disabled"}}'])
sys.settrace(boundary)
try:
    runpy.run_path(str(adapter), run_name='__main__')
except ClientBoundary:
    sys.settrace(None)
else:
    raise AssertionError('actual CLI did not reach its reader client boundary')
assert not connections
checkpoint = json.loads((work / 'results' / 'checkpoint.json').read_text())
manifest = checkpoint['manifest']
assert checkpoint['expected_ids'] == ['startup-qid']
assert manifest['config']['indexing_completion_policy'] == 'source-backed-index-with-explicit-summary-degradation-v1'
for role in ('reader', 'judge', 'memory_pipeline'):
    assert manifest['models'][role]['model'] == 'deepseek-flash'
pipeline = manifest['models']['memory_pipeline']
assert pipeline['aggregation_producer']['identity_exact'] is True
assert pipeline['deployment_revision_sha256'] is pipeline['deployment_tenant_sha256'] is None
assert pipeline['effective_extra_body'] == {'thinking': {'type': 'disabled'}}
assert manifest['models']['reader']['extra_body'] == {'thinking': {'type': 'disabled'}}
assert manifest['models']['judge']['extra_body'] == {'thinking': {'type': 'disabled'}}
print('ACTUAL_CLI_CHECKPOINT_CLIENT_BOUNDARY_NO_PROVIDER_CALLS')
'''

@pytest.mark.parametrize('explicit', [False, True])
def test_actual_cli_reaches_checkpoint_and_client_boundary_without_provider_calls(tmp_path, explicit):
    rows = [{
        'question_id': 'startup-qid', 'question_type': 'multi-session',
        'question': 'Which city?', 'answer': 'Utrecht', 'question_date': '2025-01-03',
        'answer_session_ids': ['session-0'], 'haystack_session_ids': ['session-0'],
        'haystack_dates': ['2025-01-01'],
        'haystack_sessions': [[{'role': 'user', 'content': 'I live in Utrecht.',
                               'has_answer': True}]],
    }]
    (tmp_path / 'longmemeval_s_cleaned.json').write_text(json.dumps(rows))
    result = subprocess.run(
        [sys.executable, '-I', '-B', '-c', STARTUP, str(Path(__file__).resolve().parents[1]),
         str(tmp_path), 'yes' if explicit else 'no'], cwd=tmp_path, env={},
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'ACTUAL_CLI_CHECKPOINT_CLIENT_BOUNDARY_NO_PROVIDER_CALLS' in result.stdout
