"""Offline one-shot host dispatch and private-path checks; no systemd launch."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


PATH = Path(__file__).resolve().parents[1] / 'luna_grounding_probe_host.py'
SPEC = importlib.util.spec_from_file_location('reviewed_grounding_probe_host', PATH)
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)


def test_command_has_exact_bounded_supervision(tmp_path):
    argv = host.command(tmp_path, 'fixed.service')
    for option in ('--property=RuntimeMaxSec=1830s', '--property=TasksMax=256',
                   '--property=MemoryMax=4294967296', '--property=CPUQuota=200%',
                   '--property=KillMode=control-group', '--property=Restart=no'):
        assert option in argv
    assert '--candidate' in argv and '--grounded-prompt' in argv
    assert argv[argv.index('--inventory-sha256') + 1] == host.PINS['headless-source-map.json']


def test_dangling_destination_symlink_is_not_followed(tmp_path):
    source = tmp_path / 'source'
    source.write_bytes(b'pinned')
    outside = tmp_path / 'outside'
    destination = tmp_path / 'destination'
    destination.symlink_to(outside)
    with pytest.raises(FileExistsError):
        host.copy_pinned_file(source, destination, hashlib.sha256(b'pinned').hexdigest())
    assert not outside.exists()


def test_workdirs_must_remain_private_real_and_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(host, 'HOST_UID', os.getuid())
    for name in ('empty', 'tmp'):
        (tmp_path / name).mkdir(mode=0o700)
    host.valid_workdirs(tmp_path)
    (tmp_path / 'empty' / 'stray').write_text('x')
    with pytest.raises(RuntimeError, match='workdir_invalid'):
        host.valid_workdirs(tmp_path)


def _launch_fixture(tmp_path, monkeypatch):
    root = tmp_path / '.hymem-luna-grounding-probe-abcdefgh'
    root.mkdir()
    binary = tmp_path / 'codex'
    binary.write_bytes(b'offline')
    monkeypatch.setattr(host, 'BINARY', binary)
    monkeypatch.setattr(host, 'valid_root', lambda path: None)
    monkeypatch.setattr(host, 'verify', lambda path: SimpleNamespace(host_admission=lambda: None))
    monkeypatch.setattr(host, 'unit_for', lambda path: 'offline.service')
    monkeypatch.setattr(host, 'command', lambda path, unit: ['/usr/bin/systemd-run', '--user', unit])
    receipt = {'root': str(root), 'unit': 'offline.service', 'source_pins': host.PINS,
               'host_sha256': host.digest(PATH), 'binary_sha256': host.digest(binary),
               'command': host.command(root, 'offline.service')}
    host.write_once(root / 'launch-receipt.json', receipt)
    monkeypatch.setattr(sys, 'argv', [str(PATH), 'launch', '--root', str(root),
                                    '--receipt-sha256', host.digest(root / 'launch-receipt.json')])
    return root


def test_timeout_consumes_one_shot_and_retains_private_dispatch_output(tmp_path, monkeypatch):
    root = _launch_fixture(tmp_path, monkeypatch)
    calls = []
    def ambiguous(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args[0], 20, output=b'private stdout', stderr=b'private stderr')
    monkeypatch.setattr(host.subprocess, 'run', ambiguous)
    with pytest.raises(subprocess.TimeoutExpired):
        host.main()
    with pytest.raises(FileExistsError):
        host.main()
    assert len(calls) == 1
    assert (root / 'launch-attempt.json').is_file()
    assert (root / 'private-dispatch-stdout.bin').read_bytes() == b'private stdout'
    assert (root / 'private-dispatch-stderr.bin').read_bytes() == b'private stderr'
    assert (root / 'private-dispatch-stderr.bin').stat().st_mode & 0o077 == 0


def test_failed_dispatch_records_private_output_and_no_second_attempt(tmp_path, monkeypatch):
    root = _launch_fixture(tmp_path, monkeypatch)
    calls = []
    def failed(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0], 1, stdout=b'private out', stderr=b'private err')
    monkeypatch.setattr(host.subprocess, 'run', failed)
    assert host.main() == 1
    with pytest.raises(FileExistsError):
        host.main()
    assert len(calls) == 1
    assert json.loads((root / 'launch-command-result.json').read_text())['returncode'] == 1
    assert (root / 'private-dispatch-stderr.bin').read_bytes() == b'private err'
