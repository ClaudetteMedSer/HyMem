"""Derive an inactive, immutable claim-task bundle from the accepted v3 bundle."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat

FAMILY = 'luna-claim-task-probe'
ACCEPTED = Path('/private/tmp/hymem-classification-v3-accepted-Yy41cZsw/bundle')
ACCEPTED_RECEIPT_SHA = 'cf426350d77615dbb589a74823d49432a7d7bd0583b26a9b4d3e3360ed64f814'
V3_BUNDLE = 'tools/diagnostics/luna_classification_bundle_v3.py'
V3_BUNDLE_SHA = '9207e7d757dcc1f2be10c51e5bb8b5666da7788950f1af832c88407751a70029'
FROZEN = {
    'tools/diagnostics/luna_claim_task_contract_v1.py': '6e8df574632533437d6c10db7391cc419da792f180f7794e96950bc318307f89',
    'benchmarks/codex_subscription_claim_task_v1.py': '55d5fb9de06206156011e1c8cd603823273d3d8923a76c914ae861c508485346',
    'tools/diagnostics/luna_claim_task_core_v1.py': '34b16df04807a71dea1d535591b0a481d88fa3b9185080f31e2545a79ffdd4e7',
}
TEMPLATES = ('tools/diagnostics/luna_claim_task_run_v1.py',
             'tools/diagnostics/luna_claim_task_progress_v1.py',
             'tools/diagnostics/luna_claim_task_replay_v1.py')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def pinned(path: Path, expected: str) -> bytes:
    if (not path.is_absolute() or path.is_symlink() or
            not stat.S_ISREG(path.lstat().st_mode)):
        raise ValueError('source_path_invalid')
    raw = path.read_bytes()
    if sha(raw) != expected:
        raise ValueError('source_pin_invalid:' + path.name)
    return raw


def one(raw: bytes, before: str, after: str) -> bytes:
    b = before.encode()
    if raw.count(b) != 1:
        raise ValueError('binding_count_invalid:' + before[:70])
    return raw.replace(b, after.encode())


def many(raw: bytes, before: str, after: str, minimum: int = 1) -> bytes:
    b = before.encode()
    if raw.count(b) < minimum:
        raise ValueError('binding_missing:' + before[:70])
    return raw.replace(b, after.encode())


def span(raw: bytes, start: str, end: str, replacement: str) -> bytes:
    b, e = start.encode(), end.encode()
    if raw.count(b) != 1 or raw.count(e) != 1:
        raise ValueError('span_anchor_invalid')
    i = raw.index(b)
    j = raw.index(e, i)
    return raw[:i] + replacement.encode() + raw[j:]


def write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def prepare(repo: Path, target: Path) -> dict:
    if (not repo.is_absolute() or repo != repo.resolve() or not repo.is_dir() or
            not target.is_absolute() or target.exists() or target.is_symlink() or
            target.parent != target.parent.resolve() or not target.parent.is_dir() or
            target.is_relative_to(repo) or target.is_relative_to(ACCEPTED)):
        raise ValueError('output_boundary_invalid')
    pinned(repo / V3_BUNDLE, V3_BUNDLE_SHA)
    spec = importlib.util.spec_from_file_location('pinned_claim_task_v3_bundle', repo / V3_BUNDLE)
    if spec is None or spec.loader is None:
        raise ValueError('v3_bundle_load_invalid')
    v3_bundle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v3_bundle)
    if len(v3_bundle.validate_candidate()) != 513:
        raise ValueError('physical_candidate_invalid')
    receipt = json.loads(pinned(ACCEPTED / 'derivation-receipt.json', ACCEPTED_RECEIPT_SHA))
    if (receipt.get('candidate_files') != 513 or receipt.get('model_calls') != 0 or
            receipt.get('launched') is not False or
            receipt.get('schema') != 'luna-classification-v3-probe-bundle-v1'):
        raise ValueError('accepted_receipt_invalid')
    previous = {}
    for name, digest in receipt['output_sha256'].items():
        previous[name] = pinned(ACCEPTED / name, digest)
    local = {name: pinned(repo / name, digest) for name, digest in FROZEN.items()}
    for name in TEMPLATES:
        path = repo / name
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError('template_path_invalid')
        local[name] = path.read_bytes()
    code = {name[5:]: raw for name, raw in previous.items() if name.startswith('code/')}
    old_family = 'luna-classification-v3-probe'
    new_paths = {
        'tools/diagnostics/luna_claim_task_contract_v1.py': local['tools/diagnostics/luna_claim_task_contract_v1.py'],
        'tools/diagnostics/luna_claim_task_core_v1.py': local['tools/diagnostics/luna_claim_task_core_v1.py'],
        'tools/diagnostics/luna_claim_task_run_v1.py': local['tools/diagnostics/luna_claim_task_run_v1.py'],
        'tools/diagnostics/luna_claim_task_replay_v1.py': local['tools/diagnostics/luna_claim_task_replay_v1.py'],
        'tools/diagnostics/luna_classification_progress_reference_v3.py': code['tools/diagnostics/luna_semantic_probe_progress.py'],
    }
    transport = one(local['benchmarks/codex_subscription_claim_task_v1.py'],
        'd3c955922219d99b359aa6619b821e7da1662e36ecea54d09b1267dce38d2c32',
        sha(code['benchmarks/codex_subscription_classification_v3.py']))
    new_paths['benchmarks/codex_subscription_claim_task_v1.py'] = transport
    code.update(new_paths)

    reference = code['tools/diagnostics/luna_semantic_probe_run.py']
    reference = many(reference, old_family, FAMILY)
    reference = one(reference, "def preflight(root: Path, receipt_sha: str) -> tuple[dict, object, object, tuple]:",
        "def preflight(root: Path, receipt_sha: str, *, require_empty_workdirs: bool = True) -> tuple[dict, object, object, tuple]:")
    reference = one(reference, "    host.verify_bundle(root, receipt)\n",
        "    host.verify_bundle(root, receipt, require_empty_workdirs=require_empty_workdirs)\n")
    code['tools/diagnostics/luna_semantic_probe_run.py'] = reference

    host = code['tools/diagnostics/luna_semantic_probe_host.py']
    host = many(host, old_family, FAMILY)
    for name, raw in new_paths.items():
        anchor = "    'tools/diagnostics/luna_semantic_candidate_v2.py':"
        host = one(host, anchor, f"    '{name}': '{sha(raw)}',\n" + anchor)
    host = one(host, "NEW = ('tools/diagnostics/luna_semantic_probe_host.py',\n       'tools/diagnostics/luna_semantic_probe_run.py',\n       'tools/diagnostics/luna_semantic_probe_progress.py')",
        "NEW = ('tools/diagnostics/luna_semantic_probe_host.py',\n       'tools/diagnostics/luna_semantic_probe_run.py',\n       'tools/diagnostics/luna_semantic_probe_progress.py')")
    host = one(host, "'control_turns': 2, 'control_known_tokens': 100000, 'control_seconds': 240,\n          'hybrid_turns': 3, 'hybrid_known_tokens': 160000, 'hybrid_seconds': 600,",
        "'control_turns': 1, 'control_known_tokens': 100000, 'control_seconds': 240,\n          'hybrid_turns': 0, 'hybrid_known_tokens': 0, 'hybrid_seconds': 0,")
    host = one(host, "'invocation_seconds': 120, 'workers': 1, 'units': 25,",
        "'invocation_seconds': 120, 'workers': 1, 'units': 29,")
    host = one(host, "'schedule': {'controls': 24, 'supported': 12, 'reject': 10,\n                         'correction': 2, 'hybrid_last': True},",
        "'schedule': {'control_pairs': 12, 'paired_turns': 24,\n                         'nominated_prefers': 1, 'canary_turns': 4},")
    host = one(host, "str(root / 'code/tools/diagnostics/luna_semantic_probe_run.py'),",
        "str(root / 'code/tools/diagnostics/luna_claim_task_run_v1.py'),")
    code['tools/diagnostics/luna_semantic_probe_host.py'] = host

    reader = local['tools/diagnostics/luna_claim_task_progress_v1.py']
    code['tools/diagnostics/luna_semantic_probe_progress.py'] = reader
    startup = many(previous['adapter-v2.py'], old_family, FAMILY)
    startup = one(startup, "HOST_SHA = 'c37c5a5462541e4a705bd069a1ba1540f4d1e88883a98be45860da28d49835aa'",
                  f"HOST_SHA = '{sha(host)}'")
    startup = one(startup, "RUN_SHA = '79c675a1aa147df6dced69819d0025b13066208df6c5947ab59b0ec467866c2d'",
                  f"RUN_SHA = '{sha(new_paths['tools/diagnostics/luna_claim_task_run_v1.py'])}'")
    startup = one(startup, "READER_SHA = '85a9e178342bc56ed740633e7ba82e938282823c5699cbc1deea3274cc005939'",
                  f"READER_SHA = '{sha(reader)}'")
    startup = many(startup, 'luna_semantic_probe_run.py', 'luna_claim_task_run_v1.py')
    startup = one(startup, "return 0 if result['core_completed'] else 1",
                  "return 0 if result['diagnostic_completed'] else 1")
    startup = one(startup, "    original_module = run.subprocess\n    original = original_module.run",
        "    reference = run._reference(root, digest(root / 'launch-receipt.json'))\n    original_module = reference.subprocess\n    original = original_module.run")
    startup = one(startup, "    run.subprocess = SimpleNamespace(run=scoped)\n    try:\n        run.verify_live_containment(root, receipt)\n    finally:\n        run.subprocess = original_module",
        "    reference.subprocess = SimpleNamespace(run=scoped)\n    try:\n        reference.verify_live_containment(root, receipt)\n    finally:\n        reference.subprocess = original_module")
    startup = one(startup, "    original_module = progress.subprocess\n    original = original_module.run",
        "    import sys\n    sys.path.insert(0, str(root / 'code'))\n    sys.path.insert(0, str(root / 'candidate'))\n    original_module = progress.subprocess\n    original = original_module.run")
    startup = many(startup, "'semantic_fix_accepted': False", "'semantic_accuracy_accepted': False, 'full_lme_ready': False")
    startup = many(startup, "'completed_and_clean': False,\n                    'stop_code': 'pre_inference_startup_failure'",
        "'completed_and_clean': False,\n                    'semantic_accuracy_accepted': False, 'full_lme_ready': False,\n                    'stop_code': 'pre_inference_startup_failure'", 1)
    startup = many(startup, "'paid_turns': 0, 'known_tokens': 0}",
        "'paid_turns': 0, 'known_tokens': 0,\n                'semantic_accuracy_accepted': False, 'full_lme_ready': False}", 1)
    startup = one(startup, "                'completed_and_clean': False,\n                'stop_code': 'pre_inference_startup_failure'}",
        "                'completed_and_clean': False,\n                'semantic_accuracy_accepted': False, 'full_lme_ready': False,\n                'stop_code': 'pre_inference_startup_failure'}")
    startup = one(startup, "result['schema'] = 'luna-claim-task-probe-progress-v2'",
                  "result['schema'] = 'luna-claim-task-probe-progress-v2'")
    # The observer's scoped systemctl runner remains the only live control-plane action.
    startup = one(startup, "        unit = progress._systemd(receipt['unit'], receipt['expected_cgroup'])",
                  "        unit = progress._systemd(receipt['unit'], receipt['expected_cgroup'])")
    output = {'code/' + name: raw for name, raw in code.items()}
    output['adapter-v2.py'] = startup
    output['verdict-replay.py'] = new_paths['tools/diagnostics/luna_claim_task_replay_v1.py']
    receipt_new = {'schema': FAMILY + '-bundle-v1',
        'accepted_v3_derivation_receipt_sha256': ACCEPTED_RECEIPT_SHA,
        'candidate_inventory_sha256': receipt['candidate_inventory_sha256'],
        'candidate_files': 513, 'extraction_identity': receipt['extraction_identity'],
        'input_sha256': {**receipt['input_sha256'], V3_BUNDLE: V3_BUNDLE_SHA,
                         **{name: sha(raw) for name, raw in local.items()}},
        'output_sha256': {name: sha(raw) for name, raw in sorted(output.items())},
        'model_calls': 0, 'launched': False}
    target.mkdir(mode=0o700)
    for name, raw in output.items():
        write(target / name, raw)
    write(target / 'derivation-receipt.json',
          (json.dumps(receipt_new, sort_keys=True, indent=2) + '\n').encode())
    return {'target': str(target), 'receipt_sha256': sha((target / 'derivation-receipt.json').read_bytes()),
            'code_files': len(code), 'model_calls': 0, 'launched': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--target', required=True, type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.repo, args.target), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
