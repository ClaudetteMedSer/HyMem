"""Carry forward the previously accepted doctor bytes; no SSH or gate changes."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parent))
import hymem_v64_rollout as r
import hymem_v64_postdeploy as post

OLD = '5aa54d3de99a0ed9bd1c3e7cebef6f2d7472f0482783b87a46aed8a2536193bc'
NEW = '2ac786c6590aa7fbece726e6380981906d514fefe9f664a3ca4f7b1d1de7c15b'
POST_PIN = '6289c704984898f611a2b03de40415aa6e49985ca0bad05baa7f254d42e992aa'
DOCTOR = 'hymem/doctor.py'

def production_files(files):
    r.need(len(files) == 481 and r.digest(files) == r.CANDIDATE_PIN,
           'base_manifest_drift')
    r.need(files.get(DOCTOR) == OLD, 'unexpected_base_doctor')
    return {**files, DOCTOR: NEW}

def identity(files):
    actual = production_files(files)
    return {'base_manifest_sha256': r.CANDIDATE_PIN,
            'production_manifest_sha256': r.digest(actual),
            'diagnostic_only_override': {'path': DOCTOR, 'before_sha256': OLD,
                                         'sha256': NEW},
            'unchanged_base_files': 480}

def install(root, backup, stage, files):
    """Exclusive local installation from the sealed rollout source backup."""
    actual = production_files(files)
    r.verify_files(root, files)
    source = r.regular(r.safe_target(backup, DOCTOR))
    r.need(r.sha(source) == NEW, 'accepted_doctor_backup_drift')
    target = r.regular(r.safe_target(root, DOCTOR))
    info = target.stat()
    r.need(info.st_uid == os.geteuid() and info.st_gid in os.getgroups(),
           'doctor_owner_mismatch')
    r.need(stage.is_dir() and not stage.is_symlink(), 'stage_identity_invalid')
    r.exclusive_json(stage / 'doctor-carryforward-intent.json', identity(files))
    fd = os.open(stage / 'doctor-carryforward-failed-candidate.py',
                 os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as out:
        out.write(target.read_bytes()); out.flush(); os.fsync(out.fileno())
    body = source.read_bytes()
    r.need(r.digest(files) == r.CANDIDATE_PIN and r.sha(source) == NEW,
           'accepted_doctor_backup_drift')
    import hashlib
    r.need(hashlib.sha256(body).hexdigest() == NEW, 'accepted_doctor_backup_drift')
    fd, name = tempfile.mkstemp(prefix='.doctor-carryforward-', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(body); out.flush(); os.fsync(out.fileno())
            os.fchmod(out.fileno(), stat.S_IMODE(info.st_mode))
            os.fchown(out.fileno(), info.st_uid, info.st_gid)
        r.verify_files(root, files)
        os.replace(name, target)
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if Path(name).exists(): Path(name).unlink()
    r.verify_files(root, actual)
    report = {'status': 'passed', **identity(files), 'restart_required': False,
              'frozen_candidate_changed': False}
    r.exclusive_json(stage / 'doctor-carryforward.json', report)
    return report

def container_script(files, census):
    r.need(r.sha(r.regular(Path(post.__file__))) == POST_PIN, 'postdeploy_adapter_pin_drift')
    script = post.container_script(production_files(files), census)
    # Keep every existing probe and invariant; add only truthful source identity.
    return script.replace("    report['source_files_verified']=481",
                          "    report['source_files_verified']=481\n    report.update(" + repr(identity(files)) + ")")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('install', 'verify'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--config-sha256', required=True)
    args = parser.parse_args()
    cfg = r.read_sealed(args.config, args.config_sha256)
    rollout = r.Rollout(cfg, args.config_sha256)
    rollout.prior('start'); rollout.inspect(); rollout.preserve()
    if args.action == 'install':
        r.verify_files(rollout.stage / 'source-backup', rollout.old)
        report = install(r.LIVE, rollout.stage / 'source-backup', rollout.stage, rollout.files)
    else:
        receipt = json.loads(r.regular(rollout.stage / 'doctor-carryforward.json').read_bytes())
        r.need(receipt == {'status': 'passed', **identity(rollout.files),
                          'restart_required': False, 'frozen_candidate_changed': False},
               'carryforward_receipt_drift')
        r.verify_files(r.LIVE, production_files(rollout.files))
        raw = rollout.run(['docker', 'exec', 'hermes-1', '/home/node/hymem-env/bin/python3',
                           '-I', '-B', '-c', container_script(rollout.files, rollout.census)], timeout=300)
        report = json.loads(raw)
        r.need(report.get('status') == 'passed' and report.get('role_environment_preserved') is True
               and report.get('source_files_verified') == 481, 'v64_postdeploy_failed')
        report = rollout.receipt('postdeploy-doctor-carryforward', {'verification': report,
                                                                  **identity(rollout.files)})
    rollout.preserve()
    print(json.dumps(report, sort_keys=True))

if __name__ == '__main__':
    try: main()
    except Exception:
        print(json.dumps({'status': 'failed', 'inspect_private_evidence': True})); raise SystemExit(1)
