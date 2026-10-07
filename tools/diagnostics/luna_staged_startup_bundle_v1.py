"""Derive a versioned staged startup sidecar without changing accepted code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat

FAMILY = "luna-staged-probe"
BASE = Path("/private/tmp/hymem-staged-bundle-root-B0isuUYP/bundle")
BASE_RECEIPT_SHA = "d121f9d8c89add8f3ebe6b3c2a56d37bfa3744b2c4392e30e7a98bdcca9d4a90"
OLD_ADAPTER = Path("/private/tmp/hymem-claim-task-accepted-KXzUiPyJ/bundle/adapter-v2.py")
OLD_ADAPTER_SHA = "2c057cf2eb099f2d226439c36ff48e55cf4013790e6f46e19f060b7d6fb4d223"
ADAPTER = "adapter-run-staged-v1.py"
SIDECAR = "adapter-receipt-staged-v1.json"
RECEIPT = "startup-derivation-receipt.json"
HOST_SHA = "b7da6e026b52e304aa5e225f23195c062da56596f6fd461f9d837d3bebe95e5f"
RUN_SHA = "82f34e19702b03b48ffeade68a94794c7aa1df18bcd006e340ff8f6272078987"
READER_SHA = "1be546cf33b9e5364e29b8c2d196ed34caca478c417aff233c98c0d8f2f44e45"
REPLAY_SHA = "33c5fb61f87fe228559c609e63bb19803b72d22dfa188f539b14306b542ff0a7"
HEX = re.compile(r"[0-9a-f]{64}\Z")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pinned(path: Path, expected: str) -> bytes:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("source_path_invalid")
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError("source_pin_invalid:" + path.name)
    return raw


def base_inventory(base: Path = BASE) -> dict[str, bytes]:
    if not base.is_absolute() or base != base.resolve() or not base.is_dir():
        raise ValueError("base_boundary_invalid")
    receipt_raw = pinned(base / "derivation-receipt.json", BASE_RECEIPT_SHA)
    receipt = json.loads(receipt_raw)
    outputs = receipt.get("output_sha256")
    if (type(receipt) is not dict or receipt.get("schema") != FAMILY + "-bundle-v1" or
            receipt.get("candidate_files") != 514 or receipt.get("model_calls") != 0 or
            receipt.get("launched") is not False or type(outputs) is not dict or
            len(outputs) != 37 or any(type(k) is not str or not k.startswith("code/") or
            Path(k).is_absolute() or ".." in Path(k).parts or type(v) is not str or
            HEX.fullmatch(v) is None for k, v in outputs.items()) or
            outputs.get("code/tools/diagnostics/luna_staged_host_v1.py") != HOST_SHA or
            outputs.get("code/tools/diagnostics/luna_staged_run_v1.py") != RUN_SHA or
            outputs.get("code/tools/diagnostics/luna_staged_progress_v1.py") != READER_SHA or
            outputs.get("code/tools/diagnostics/luna_staged_replay_v1.py") != REPLAY_SHA):
        raise ValueError("base_receipt_invalid")
    found = set()
    for path in base.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError("base_path_invalid")
        if path.is_file():
            name = path.relative_to(base).as_posix()
            if name not in set(outputs) | {"derivation-receipt.json"}:
                raise ValueError("base_extra_file")
            found.add(name)
    if found != set(outputs) | {"derivation-receipt.json"}:
        raise ValueError("base_inventory_incomplete")
    return {name: pinned(base / name, digest) for name, digest in outputs.items()}


def one(raw: bytes, before: str, after: str) -> bytes:
    needle = before.encode()
    if raw.count(needle) != 1:
        raise ValueError("adapter_binding_invalid:" + before[:60])
    return raw.replace(needle, after.encode())


def section(raw: bytes, start: str, end: str, replacement: str) -> bytes:
    first, last = raw.find(start.encode()), raw.find(end.encode())
    if first < 0 or last <= first or raw.count(start.encode()) != 1:
        raise ValueError("adapter_section_invalid:" + start)
    return raw[:first] + replacement.encode() + raw[last:]


def derive_adapter() -> bytes:
    raw = pinned(OLD_ADAPTER, OLD_ADAPTER_SHA)
    raw = one(raw, "Source-bound v2 startup adapter for a fresh semantic diagnostic attempt.",
              "Source-bound v1 startup adapter for the staged diagnostic probe.")
    raw = one(raw, "The accepted v1 bundle and receipt remain byte-for-byte unchanged.",
              "The accepted staged bundle and receipt remain byte-for-byte unchanged.")
    raw = raw.replace(b"luna-claim-task-probe", b"luna-staged-probe")
    raw = one(raw, "SIDECAR = 'adapter-receipt-v2.json'", f"SIDECAR = '{SIDECAR}'")
    raw = one(raw, "ADAPTER = 'adapter-run-v2.py'", f"ADAPTER = '{ADAPTER}'")
    for key, digest in (("HOST_SHA", HOST_SHA), ("RUN_SHA", RUN_SHA),
                        ("READER_SHA", READER_SHA)):
        raw = re.sub(rb"(?m)^" + key.encode() + rb" = '[0-9a-f]{64}'$",
                     (key + " = '" + digest + "'").encode(), raw, count=1)
    raw = one(raw, f"READER_SHA = '{READER_SHA}'",
              f"READER_SHA = '{READER_SHA}'\nREPLAY_SHA = '{REPLAY_SHA}'")
    raw = one(raw, "from types import SimpleNamespace", "from types import SimpleNamespace")
    if raw.count(b"'bus_scope': 'systemctl-only'") != 2:
        raise ValueError("bus_scope_binding_invalid")
    raw = raw.replace(b"'bus_scope': 'systemctl-only'",
                      b"'bus_scope': 'systemctl-and-systemd-run-only'")
    raw = raw.replace(b"adapter-v2", b"adapter-staged-v1")
    raw = raw.replace(b"startup-failure-v2", b"startup-failure-staged-v1")
    raw = raw.replace(b"containment-smoke-v2", b"containment-smoke-staged-v1")
    raw = raw.replace(b"progress-v2", b"progress-staged-v1")
    raw = raw.replace(b"adapter-error-v2", b"adapter-error-staged-v1")
    if raw.count(b"root / 'code/tools/diagnostics/luna_claim_task_run_v1.py'") != 2:
        raise ValueError("run_binding_count_invalid")
    raw = raw.replace(b"root / 'code/tools/diagnostics/luna_claim_task_run_v1.py'",
                      b"root / 'code/tools/diagnostics/luna_staged_run_v1.py'")
    if raw.count(b"root / 'code/tools/diagnostics/luna_semantic_probe_host.py'") != 3:
        raise ValueError("host_binding_count_invalid")
    raw = raw.replace(b"root / 'code/tools/diagnostics/luna_semantic_probe_host.py'",
                      b"root / 'code/tools/diagnostics/luna_staged_host_v1.py'")
    raw = one(raw, "diagnostics / 'luna_semantic_probe_host.py'",
              "diagnostics / 'luna_staged_host_v1.py'")
    raw = one(raw, "diagnostics / 'luna_semantic_probe_progress.py'",
              "diagnostics / 'luna_staged_progress_v1.py'")
    raw = section(raw, "def contained(", "def entry(", CONTAINED)
    raw = section(raw, "def entry(", "def launch(", ENTRY)
    raw = section(raw, "def launch(", "def observe(", LAUNCH)
    raw = section(raw, "def observe(", "def main(", OBSERVE)
    raw = one(raw, "    if mode not in {'smoke', 'inference'}:\n        raise ValueError('adapter_mode_invalid')\n    if not host.root_valid(root)",
              "    if mode not in {'smoke', 'inference'}:\n        raise ValueError('adapter_mode_invalid')\n    if (source != Path(__file__).resolve() or source.is_symlink() or\n            not source.is_file()):\n        raise ValueError('adapter_source_invalid')\n    if not host.root_valid(root)")
    raw = one(raw, "out = host.prepare(Path(args.staged_root))",
              "out = prepare_with_bus(host, Path(args.staged_root))")
    raw = one(raw, "sidecar = seal(root, Path(args.adapter_source or __file__), host, args.mode)",
              "sidecar = seal(root, Path(args.adapter_source or __file__), host, args.mode)")
    return raw


CONTAINED = '''def contained(root: Path, receipt: dict, run) -> None:
    """Scope the fixed user bus to the staged runner's exact show command."""
    original_module = run.subprocess
    original = original_module.run
    def scoped(args, **kwargs):
        if (type(args) is not list or len(args) != 6 or
                args[:3] != ['/usr/bin/systemctl', '--user', 'show'] or
                args[3] != receipt['unit'] or
                args[4] != '--property=ActiveState,SubState,MainPID,ControlGroup,NRestarts,MemoryMax,TasksMax,CPUQuotaPerSecUSec,KillMode,Restart,RemainAfterExit,OOMPolicy,RuntimeMaxUSec,TimeoutStopUSec' or
                args[5] != '--no-pager' or 'env' in kwargs):
            raise ValueError('containment_command_invalid')
        return original(args, env=bus_environment(), **kwargs)
    run.subprocess = SimpleNamespace(run=scoped)
    try:
        run.verify_live_containment(root, receipt)
    finally:
        run.subprocess = original_module


def control_plane(host, action, *args, **action_kwargs):
    """Expose the user bus only to host admission's fixed systemctl calls."""
    original_module = host.subprocess
    original = original_module.run
    def scoped(command, **kwargs):
        if type(command) is not list or 'env' in kwargs:
            raise ValueError('host_command_invalid')
        if command and command[0] == '/usr/bin/systemctl':
            listed = ['/usr/bin/systemctl', '--user', 'list-units', 'hymem-luna*',
                      '--all', '--plain', '--no-legend', '--no-pager']
            shown = (len(command) == 6 and command[:3] ==
                     ['/usr/bin/systemctl', '--user', 'show'] and
                     command[3].startswith('hymem-luna') and
                     command[3].endswith('.service') and
                     command[4] == '--property=MainPID,ControlGroup,NRestarts' and
                     command[5] == '--no-pager')
            if command != listed and not shown:
                raise ValueError('host_command_invalid')
            return original(command, env=bus_environment(), **kwargs)
        if command == ['/usr/bin/docker', 'inspect', '--format',
                       '{{.State.Running}}', host.OLD_DEEPSEEK]:
            return original(command, **kwargs)
        raise ValueError('host_command_invalid')
    host.subprocess = SimpleNamespace(run=scoped)
    try:
        return action(*args, **action_kwargs)
    finally:
        host.subprocess = original_module


def prepare_with_bus(host, staged: Path) -> dict:
    return control_plane(host, host.prepare, staged)


'''

ENTRY = '''def entry(root: Path, receipt_sha: str, mode: str, adapter_sha: str,
          sidecar_sha: str) -> int:
    verified = False
    try:
        verify(root, receipt_sha, adapter_sha, mode, sidecar_sha)
        if Path(__file__) != root / ADAPTER or digest(Path(__file__)) != adapter_sha:
            raise ValueError('entry_pin_invalid')
        verified = True
        run = load_source(root / 'code/tools/diagnostics/luna_staged_run_v1.py',
                          'sealed_staged_entry_v1', RUN_SHA)
        if mode == 'smoke':
            receipt, _, _, _ = run.preflight(root, receipt_sha)
            contained(root, receipt, run)
            host = load_source(root / 'code/tools/diagnostics/luna_staged_host_v1.py',
                               'sealed_staged_host_v1_smoke', HOST_SHA)
            host.write_once(root / 'safe-terminal.json', {
                'schema': 'luna-staged-probe-containment-smoke-staged-v1',
                'receipt_sha256': receipt_sha,
                'adapter_sha256': adapter_sha,
                'policy_verified': True, 'model_calls': 0,
                'paid_turns': 0, 'known_tokens': 0,
                'semantic_accuracy_accepted': False, 'full_lme_ready': False})
            return 0
        # The staged execute preflight and containment precede run/ creation.
        # A later exception is never interpreted as zero usage.
        result = run.execute(root, receipt_sha,
                             containment=lambda r, p: contained(r, p, run))
        return 0 if result['diagnostic_completed'] else 1
    except BaseException:
        if verified and not (root / 'run').exists() and not (root / 'safe-terminal.json').exists():
            try:
                host = load_source(root / 'code/tools/diagnostics/luna_staged_host_v1.py',
                                   'sealed_staged_host_v1_failure', HOST_SHA)
                receipt_path = root / 'launch-receipt.json'
                marker = root / 'launch-attempt.json'
                if (host.root_valid(root) and host.regular(receipt_path) and
                        digest(receipt_path) == receipt_sha and host.regular(marker) and
                        host.strict_equal(json.loads(marker.read_bytes()),
                            {'receipt_sha256': receipt_sha, 'one_shot': True})):
                    receipt = json.loads(receipt_path.read_bytes())
                    if host.strict_equal(receipt, host.receipt_for(root,
                            receipt['source_sha256'], receipt['binary_sha256'])):
                        host.verify_bundle(root, receipt)
                        host.write_once(root / 'safe-terminal.json', {
                            'schema': 'luna-staged-probe-startup-failure-staged-v1',
                            'receipt_sha256': receipt_sha,
                            'adapter_sha256': adapter_sha,
                            'zero_admission_proof': 'run_directory_never_created',
                            'paid_turns': 0, 'known_tokens': 0, 'model_calls': 0,
                            'completed_and_clean': False,
                            'semantic_accuracy_accepted': False, 'full_lme_ready': False,
                            'stop_code': 'pre_inference_startup_failure'})
            except BaseException:
                pass
        elif verified and (root / 'run').is_dir() and not (root / 'safe-terminal.json').exists():
            try:
                run = load_source(root / 'code/tools/diagnostics/luna_staged_run_v1.py',
                                  'sealed_staged_entry_v1_failure', RUN_SHA)
                _, loaded, _, _ = run.preflight(root, receipt_sha, postrun=True)
                loaded[0].write_once(root / 'safe-terminal.json', {
                    'schema': 'luna-staged-probe-terminal-failure-v1',
                    'receipt_sha256': receipt_sha,
                    'diagnostic_completed': False, 'completed_and_clean': False,
                    'semantic_accuracy_accepted': False, 'full_lme_ready': False,
                    'paid_budget': None, 'stop_code': 'entrypoint_failure'})
            except BaseException:
                pass
        return 1


'''

LAUNCH = '''def launch(root: Path, receipt_sha: str, adapter_sha: str, sidecar_sha: str,
           host, mode: str) -> dict:
    verify(root, receipt_sha, adapter_sha, mode, sidecar_sha)
    def dispatch(command, **kwargs):
        old = str(root / 'code/tools/diagnostics/luna_staged_run_v1.py')
        if type(command) is not list or command.count(old) != 1 or 'env' in kwargs:
            raise ValueError('base_command_invalid')
        adapted = [str(root / ADAPTER) if item == old else item for item in command]
        index = adapted.index(str(root / ADAPTER))
        adapted.insert(index + 1, mode)
        adapted.extend(['--adapter-sha256', adapter_sha,
                        '--adapter-receipt-sha256', sidecar_sha])
        if (adapted[:2] != ['/usr/bin/systemd-run', '--user'] or
                adapted.count('/usr/bin/env') != 1 or
                adapted[adapted.index('/usr/bin/env') + 1] != '-i' or
                not any(item == 'TMPDIR=' + str(root / 'tmp') for item in adapted) or
                any(item.startswith(('DBUS_SESSION_BUS_ADDRESS=', 'XDG_RUNTIME_DIR='))
                    for item in adapted)):
            raise ValueError('launch_command_invalid')
        return subprocess.run(adapted, env=bus_environment(), **kwargs)
    return control_plane(host, host.launch, root, receipt_sha, dispatch=dispatch)


'''

OBSERVE = '''def observe(root: Path, receipt_sha: str, adapter_sha: str,
            sidecar_sha: str, progress) -> dict:
    sidecar = verify(root, receipt_sha, adapter_sha, None, sidecar_sha)
    host = load_source(root / 'code/tools/diagnostics/luna_staged_host_v1.py',
                       'sealed_staged_host_for_reader_v1', HOST_SHA)
    receipt = json.loads((root / 'launch-receipt.json').read_bytes())
    if (not host.root_valid(root) or not host.strict_equal(
            receipt, host.receipt_for(root, receipt['source_sha256'],
                                      receipt['binary_sha256']))):
        raise ValueError('base_receipt_invalid')
    host.verify_bundle(root, receipt, require_empty_workdirs=False)
    marker = root / 'launch-attempt.json'
    admission = root / 'launch-admission.json'
    expected_admission = {'receipt_sha256': receipt_sha,
        'old_luna_stopped': True, 'old_deepseek_stopped': True,
        'memory_floor_bytes': 6 * 1024**3, 'disk_floor_bytes': 20 * 1024**3,
        'memory_floor_met': True, 'disk_floor_met': True}
    if (not host.regular(marker) or not host.strict_equal(
            json.loads(marker.read_bytes()),
            {'receipt_sha256': receipt_sha, 'one_shot': True}) or
            not host.regular(admission) or not host.strict_equal(
            json.loads(admission.read_bytes()), expected_admission)):
        raise ValueError('launch_admission_invalid')
    source_sha = hashlib.sha256(json.dumps(receipt['source_sha256'], sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    reference = load_source(root / 'code/tools/diagnostics/luna_classification_progress_reference_v3.py',
        'sealed_staged_service_reference_v1',
        receipt['source_sha256']['tools/diagnostics/luna_classification_progress_reference_v3.py'])
    original_module = progress.subprocess
    original = original_module.run
    def scoped(args, **kwargs):
        if (type(args) is not list or len(args) != 6 or
                args[:3] != ['/usr/bin/systemctl', '--user', 'show'] or
                args[3] != receipt['unit'] or args[5] != '--no-pager' or
                args[4] != '--property=ActiveState,SubState,MainPID,ControlGroup,NRestarts,Result,OOMPolicy,ExecMainStatus,MemoryMax,TasksMax,CPUQuotaPerSecUSec,KillMode,Restart,RemainAfterExit,RuntimeMaxUSec,TimeoutStopUSec' or
                'env' in kwargs):
            raise ValueError('observer_command_invalid')
        return original(args, env=bus_environment(), **kwargs)
    progress.subprocess = SimpleNamespace(run=scoped)
    try:
        unit = progress._systemd(receipt['unit'], receipt['expected_cgroup'], reference)
    finally:
        progress.subprocess = original_module
    terminal_path = root / 'safe-terminal.json'
    startup = None
    if terminal_path.is_file() and not terminal_path.is_symlink():
        startup = json.loads(terminal_path.read_bytes())
    expected = {'schema': 'luna-staged-probe-startup-failure-staged-v1',
                'receipt_sha256': receipt_sha, 'adapter_sha256': adapter_sha,
                'zero_admission_proof': 'run_directory_never_created',
                'paid_turns': 0, 'known_tokens': 0, 'model_calls': 0,
                'completed_and_clean': False,
                'semantic_accuracy_accepted': False, 'full_lme_ready': False,
                'stop_code': 'pre_inference_startup_failure'}
    valid = (type(startup) is dict and
             json.dumps(startup, sort_keys=True, separators=(',', ':')) ==
             json.dumps(expected, sort_keys=True, separators=(',', ':')) and
             not (root / 'run').exists())
    smoke = {'schema': 'luna-staged-probe-containment-smoke-staged-v1',
             'receipt_sha256': receipt_sha, 'adapter_sha256': adapter_sha,
             'policy_verified': True, 'model_calls': 0,
             'paid_turns': 0, 'known_tokens': 0,
             'semantic_accuracy_accepted': False, 'full_lme_ready': False}
    failed_cleanup = (unit.get('available') is True and unit.get('active_state') == 'failed'
                      and unit.get('main_pid_zero') is True
                      and unit.get('n_restarts_zero') is True
                      and unit.get('cgroup_empty') is True
                      and unit.get('resource_policy_verified') is True)
    if valid:
        return {'schema': 'luna-staged-probe-progress-staged-v1', 'phase': 'startup_failed',
                'zero_admission_verified': True, 'paid_turns': 0,
                'known_tokens': 0, 'model_calls': 0,
                'failed_unit_cleanup_verified': failed_cleanup,
                'completed_and_clean': False, 'semantic_accuracy_accepted': False, 'full_lme_ready': False}
    if (sidecar['entry_action'] == 'smoke' and type(startup) is dict and
            json.dumps(startup, sort_keys=True, separators=(',', ':')) ==
            json.dumps(smoke, sort_keys=True, separators=(',', ':')) and
            not (root / 'run').exists()):
        return {'schema': 'luna-staged-probe-progress-staged-v1',
                'phase': 'containment_smoke_passed' if unit.get('clean') is True else 'containment_smoke_cleanup_pending',
                'effective_policy_verified': True,
                'zero_admission_verified': True, 'paid_turns': 0,
                'known_tokens': 0, 'model_calls': 0,
                'unit_cleanup_verified': unit.get('clean') is True,
                'completed_and_clean': False, 'semantic_accuracy_accepted': False, 'full_lme_ready': False}
    if sidecar['entry_action'] == 'smoke':
        raise ValueError('smoke_terminal_invalid')
    result = progress.observe(root, receipt_sha, source_sha,
        systemd=lambda unit_name, group: unit)
    result['zero_admission_verified'] = False
    result['failed_unit_cleanup_verified'] = failed_cleanup
    return result


'''


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(target: Path, base: Path = BASE) -> dict:
    if (not target.is_absolute() or target.exists() or target.is_symlink() or
            target.parent != target.parent.resolve() or not target.parent.is_dir() or
            target.is_relative_to(base) or base.is_relative_to(target)):
        raise ValueError("output_boundary_invalid")
    code = base_inventory(base)
    adapter = derive_adapter()
    compile(adapter, ADAPTER, "exec")
    receipt = {"schema": FAMILY + "-startup-bundle-v1",
               "base_derivation_receipt_sha256": BASE_RECEIPT_SHA,
               "old_adapter_sha256": OLD_ADAPTER_SHA,
               "input_sha256": {"base/derivation-receipt.json": BASE_RECEIPT_SHA,
                                  "old/adapter-v2.py": OLD_ADAPTER_SHA},
               "output_sha256": {**{name: sha(raw) for name, raw in sorted(code.items())},
                                   "derivation-receipt.json": BASE_RECEIPT_SHA,
                                   ADAPTER: sha(adapter)},
               "code_files": 37, "model_calls": 0, "launched": False}
    target.mkdir(mode=0o700)
    for name, raw in code.items():
        _write(target / name, raw)
    _write(target / "derivation-receipt.json", pinned(base / "derivation-receipt.json", BASE_RECEIPT_SHA))
    _write(target / ADAPTER, adapter)
    _write(target / RECEIPT, (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    return {"target": str(target), "receipt_sha256": sha((target / RECEIPT).read_bytes()),
            "adapter_sha256": sha(adapter), "code_files": 37,
            "model_calls": 0, "launched": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.target), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
