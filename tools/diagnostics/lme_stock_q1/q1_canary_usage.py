"""Extract bounded canary accounting from the independently validated archive.

Run on Afrodite over SSH stdin with the archive SHA returned by offline
postvalidation. Benchmark text stays on the host; no application imports,
provider calls, credentials or writes are involved.
"""
import hashlib
import json
import math
from pathlib import Path
import re
import stat
import sys

ROOT = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/q1-stock-v4/live-results/benchmark')


def read(path):
    assert path.resolve() == path and path.parent == ROOT
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_size <= 16 * 1024 * 1024
    return path.read_bytes()


def main():
    assert len(sys.argv) == 2 and re.fullmatch('[0-9a-f]{64}', sys.argv[1])
    pointer = json.loads(read(ROOT / 'longmemeval-v2-hymem.json'))
    name = pointer['archive']
    assert type(name) is str and Path(name).name == name
    raw = read(ROOT / name)
    assert hashlib.sha256(raw).hexdigest() == sys.argv[1]
    data = json.loads(raw)
    segments = data['execution']['segments']
    assert len(segments) == 1
    canary = segments[0]['extraction_canary']
    assert canary['status'] == 'passed' and canary['client_closed'] is True
    usage = {}
    for field in ('calls', 'calls_available', 'request_attempts', 'request_attempts_available',
                  'successful_responses', 'successful_responses_available',
                  'prompt_tokens', 'completion_tokens', 'total_tokens', 'token_usage_available',
                  'cost_usd', 'cost_available'):
        value = canary['usage'][field]
        assert value is None or type(value) in (bool, int, float)
        assert type(value) is not float or math.isfinite(value)
        usage[field] = value
    counts = {field: canary[field] for field in ('completion_calls', 'provider_attempts')}
    counts['provider_output_truncations'] = canary['execution_path']['provider_output_truncations']
    assert all(type(value) is int and value >= 0 for value in counts.values())
    version = canary['version']
    assert type(version) is int or (type(version) is str and re.fullmatch('[a-zA-Z0-9_-]{1,80}', version))
    print(json.dumps({'archive_sha256': sys.argv[1], 'status': 'passed',
                     'client_closed': True, 'version': version, 'usage': usage,
                     **counts, 'new_provider_calls': 0}, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
