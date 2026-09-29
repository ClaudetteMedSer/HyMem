"""Versioned host adapter; preserves the exact v1 seal and acceptance gates.

Stage the new worker as claim_conflict_private_dream_postflight.py and the
unchanged v1 worker as postflight_v1.py in postflight-v2. The unchanged host
is staged as postflight_host_v1.py. No upload or remote action is provided.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

HOST_FILE = Path(__file__).with_name('postflight_host_v1.py')
HOST_SHA256 = '6b1ad8ded38f76adba0b5005caa13ee95f5eec1a2026f8d84470eacb9a4f6aae'
CHECKER_SHA256 = 'abc1293423616ccb042ed18325b367a4af69b050c91b527fe6884d747b541d15'
V1_CHECKER_SHA256 = '1ca6318fdac58d7b0715547baa8b80d4119c9a1a2012fb9f58bf96c3a443b117'


def load_host(path=HOST_FILE):
    import hashlib
    import stat

    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or path.is_symlink()
            or stat.S_IMODE(info.st_mode) != 0o400):
        raise RuntimeError('postflight_v1_host_invalid')
    if hashlib.sha256(path.read_bytes()).hexdigest() != HOST_SHA256:
        raise RuntimeError('postflight_v1_host_pin_drift')
    spec = importlib.util.spec_from_file_location('sealed_postflight_host_v1', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('postflight_v1_host_unloadable')
    host = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(host)
    host.STAGE = host.ROOT / 'postflight-v2'
    host.CHECKER_SHA = CHECKER_SHA256
    original = host.configure

    def configure(helper):
        prior = host.STAGE / 'postflight_v1.py'
        host.regular(prior, 0o400)
        host.need(host.sha(prior) == V1_CHECKER_SHA256, 'postflight_v1_checker_pin_drift')
        command, mounts = original(helper)
        mount = (str(prior), '/diag/postflight_v1.py', False)
        mounts.append(mount)
        index = command.index('--workdir')
        command[index:index] = ['--mount', 'type=bind,src=' + str(prior)
                                + ',dst=/diag/postflight_v1.py,readonly']
        command[command.index('--name') + 1] = 'hymem-private-dream-postflight-v2'
        return command, mounts

    host.configure = configure
    return host


if __name__ == '__main__':
    try:
        code = load_host().main()
    except Exception:
        print('{"status":"error"}')
        code = 1
    raise SystemExit(code)
