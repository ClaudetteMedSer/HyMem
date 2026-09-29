"""Pure Docker plan builder. This helper cannot execute Docker or start work."""
from pathlib import PurePosixPath
import re

IMAGE = 'sha256:8e0221ce80304b093d8a86a4285c29c2f2d8936ba3bc768a0339f1e92fbedce5'
RUNTIME = '/opt/stacks/hermes/instance1/home/hymem-env'
KEY = '/opt/stacks/hermes/instance1/home/.hermes/.env'
DATA = '/opt/stacks/hermes/instance1/home/.hermes/benchmarks/lme-full-20260918-5HQ61m/data/longmemeval_s_cleaned.json'


def command(root, source, manifest_sha, *, live=False):
    """Return an isolated full500 create plan; never starts work.

    `root` is a NEW dedicated run directory containing a pinned `bundle`,
    empty `home`, and empty UID1000 private results directories. `source`
    is the independently verified manifest-exact candidate, never production.
    """
    for value in (root, source):
        path = PurePosixPath(value)
        if (type(value) is not str or str(path) != value or not path.is_absolute()
                or '..' in path.parts or ',' in value
                or not value.startswith('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/')):
            raise ValueError('invalid_isolated_run_path')
    if type(live) is not bool or re.fullmatch('[0-9a-f]{64}', manifest_sha) is None:
        raise ValueError('invalid_launch_mode_or_pin')
    mode = 'live' if live else 'preflight'
    name = 'hymem-r9-full500-' + manifest_sha[:12] + '-' + mode
    mounts = {
        '/home/node/hymem-env': (RUNTIME, False),
        '/candidate': (source, False), '/diag': (root + '/bundle', False),
        '/data/longmemeval_s_cleaned.json': (DATA, False),
        '/home/node/.hermes': (root + '/home', False),
        '/results': (root + '/' + mode + '-results', True),
    }
    if live:
        mounts['/run/deepseek.env'] = (KEY, False)
    args = ['docker', 'create', '--name', name, '--pull', 'never', '--init',
            '--network', 'bridge' if live else 'none', '--user', '1000:1000',
            '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--pids-limit', '128', '--memory', '2g', '--cpus', '2',
            '--tmpfs', '/tmp:rw,noexec,nosuid,size=64m']
    for target, (origin, writable) in mounts.items():
        args += ['--mount', 'type=bind,src=' + origin + ',dst=' + target + ('' if writable else ',readonly')]
    args += ['--workdir', '/candidate', '--entrypoint', '/home/node/hymem-env/bin/python3',
             IMAGE, '-I', '-B', '/diag/q1_stock_run.py',
             'supervise' if live else 'preflight', '--manifest-sha256', manifest_sha]
    return {'command': args, 'name': name, 'image': IMAGE,
            'network': 'bridge' if live else 'none', 'mounts': mounts,
            'raw_logs_remain_remote': True, 'starts_work': False}


def validation_command(root, source, manifest_sha):
    """Create-only plan for offline postvalidation, all bind mounts read-only."""
    plan = command(root, source, manifest_sha, live=False)
    old_results = 'type=bind,src=' + root + '/preflight-results,dst=/results'
    args = plan['command']
    args[args.index(old_results)] = 'type=bind,src=' + root + '/live-results,dst=/results,readonly'
    plan['mounts']['/results'] = (root + '/live-results', False)
    plan['name'] = 'hymem-r9-full500-' + manifest_sha[:12] + '-validation'
    args[args.index('--name') + 1] = plan['name']
    args[args.index('/diag/q1_stock_run.py')] = '/diag/q1_stock_validate.py'
    args.remove('preflight')
    plan['new_provider_calls'] = 0
    return plan


