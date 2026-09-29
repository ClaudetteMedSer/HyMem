"""Retrieve only the finished R7 offline test's allowlisted verification receipts."""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-independent-summary-20260919-9fl8JQ/offline-r7/results'
REPO = Path(__file__).resolve().parents[2]
FILES = ('receipt.json', 'supervisor.json', 'junit.xml')
REMOTE = r'''
import base64,hashlib,json,pathlib
root=pathlib.Path(ROOT)
assert root.resolve()==root and root.is_dir()
files={}
for name in ('receipt.json','supervisor.json','junit.xml'):
    path=root/name
    assert not path.is_symlink() and path.is_file()
    assert path.stat().st_size<=4*1024*1024
    raw=path.read_bytes()
    files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),
                 'base64':base64.b64encode(raw).decode('ascii')}
assert json.loads(base64.b64decode(files['supervisor.json']['base64']))['status']=='passed'
assert json.loads(base64.b64decode(files['receipt.json']['base64']))['gate_passed'] is True
print(json.dumps(files,sort_keys=True))
'''


def main():
    if sys.flags.optimize:
        raise RuntimeError('optimized_execution_forbidden')
    destinations = {name: REPO / ('docs/patches/2026-09-25-lme-r7-target-' + name)
                    for name in FILES}
    assert all(not path.exists() for path in destinations.values())
    command = ['ssh', '-C', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
               '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=15',
               '-o', 'ServerAliveCountMax=2', 'afrodite',
               'python3 -I -B -c ' + shlex.quote('ROOT=' + repr(ROOT) + '\n' + REMOTE)]
    try:
        result = subprocess.run(command, capture_output=True, timeout=120)
    except subprocess.TimeoutExpired:
        raise SystemExit('receipt_download_timeout_no_automatic_retry') from None
    if result.returncode:
        raise SystemExit('receipt_download_failed_no_local_receipts_written')
    received = json.loads(result.stdout)
    assert set(received) == set(FILES)
    decoded = {}
    for name, entry in received.items():
        raw = base64.b64decode(entry['base64'], validate=True)
        assert len(raw) == entry['bytes'] <= 4 * 1024 * 1024
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        decoded[name] = raw
    for name, raw in decoded.items():
        with destinations[name].open('xb') as stream:
            stream.write(raw)
    print(json.dumps({name: {key: entry[key] for key in ('sha256', 'bytes')}
                      for name, entry in received.items()}, sort_keys=True))


if __name__ == '__main__':
    main()
