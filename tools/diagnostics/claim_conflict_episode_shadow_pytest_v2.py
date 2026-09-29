"""Prepared offline suite for the accepted episode-shadow candidate.

No SSH, provider calls, or upload operation exists here. Remote execution stays
review-gated; the original suite engine is imported only after an explicit pin.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

BASE = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky')
ROOT = BASE / 'episode-shadow-pytest-v2'
SELF = ROOT / 'claim_conflict_episode_shadow_pytest_v2.py'
WORK, OVERLAY = ROOT / 'work', ROOT / 'overlay'
REPLAY_ROOT = BASE / 'cold-replay-dream-v1/episode-shadow-replay-v1'
CANDIDATE = REPLAY_ROOT / 'candidate'
CANDIDATE_SHA = '5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576'
APPLICATION_PINS = {
    'hymem/core/db.py': 'ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3',
    'hymem/dreaming/runner.py': '25387efe4ef6cf5ca6d6178f96a0c7872748cdb3758bbf68836c222027691f5a',
}
REPLAY_RESULT_SHA = '9a05dfe4b6383b263e3d510a77e3550aab7f140036d538d0cdebf650a75ac823'
BASE_SUITE = BASE / 'proof-pytest-v2/claim_conflict_proof_pytest.py'
LOCAL_BASE_SUITE = Path(__file__).resolve().with_name('claim_conflict_proof_pytest.py')
BASE_CONTAINER_SUITE = Path('/diag/base_suite.py')
BASE_SUITE_SHA = '3b4a1180dc50e176ea1964534f2562b4cc1a60425e04485ccc056f747b0f5a5e'
PRIOR_TEST_ROOT = Path('/private/tmp/hymem-r7-cold-pytest-v2.OldfBd/tests')
TEST_ROOT = Path('/private/tmp/hymem-episode-shadow-fix.RjDDmJ/tests')
LOCAL_CANDIDATE = TEST_ROOT.parent
LOCAL_MANIFEST = Path('/Users/attavanwestreenen/AGprojects/HyMem/docs/patches/2026-09-26-episode-shadow-manifest.json')
PRIOR_TEST_NAMES = (
    'test_fact_authority.py', 'test_claim_semantic_dedup_guard.py',
    'test_alias_registration_idempotence.py', 'test_alias_registration_root_controls.py',
    'test_claim_replay_binding_regressions.py', 'test_claim_replay_proof_root.py',
    'test_claim_replay_local_proof_root.py', 'test_shared_embedding_bounds.py',
    'test_embedding_bounds_root_controls.py', 'test_chunk_embedding_batches.py',
    'test_chunk_embedding_batches_root.py', 'test_claim_replay_r7_upgrade_root.py',
    'test_summary_recovery_v63.py', 'test_r7_v64_summary_preservation.py',
    'test_claim_cold_import_replay.py', 'test_claim_cold_import_replay_root.py',
    'test_claim_cold_replay_reviewer_retired.py', 'test_orphan_quarantine_application.py',
)
NEW_TEST_NAMES = ('test_db_shadows.py', 'test_indexing_deadlines.py',
                  'test_episode_shadow_root_controls.py')
TEST_NAMES = PRIOR_TEST_NAMES + NEW_TEST_NAMES
PRIOR_OVERLAY_SHA = '5bb544abc0df397bf6b284a29d44de11b129d9c17d86b6740e04857764fc37e7'
OVERLAY_SHA = 'acbb892d530f2426c8a81dc527eded867542d6ebb7e0858f8bb8e418163a02ab'
FULL_SECONDS, HOST_SECONDS = 10800, 11200
EXPECTED_COLLECTED = 7764  # Prior 7752 + seven root controls + five deadline tests.
EFFECTIVE_TEST_FILES = 257
EFFECTIVE_TEST_SHA = '9d5edaef1b9c6a4fa0c2875b3a2ee5dfa97d4225641bd016f9569f9c809f7f08'
REVIEWED_REMOTE_EXECUTION = True  # Root reviewed source, pins, isolation and 36 adapter controls.


def need(ok, code):
    if not ok:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_base(context):
    paths = {'local': LOCAL_BASE_SUITE, 'host': BASE_SUITE,
             'container': BASE_CONTAINER_SUITE}
    need(context in paths, 'invalid_execution_context')
    path = paths[context]
    need(path.is_file() and not path.is_symlink() and sha(path) == BASE_SUITE_SHA,
         'base_suite_pin_drift')
    spec = importlib.util.spec_from_file_location('episode_shadow_pinned_suite', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sources():
    sources = {}
    for root, names in ((PRIOR_TEST_ROOT, PRIOR_TEST_NAMES), (TEST_ROOT, NEW_TEST_NAMES)):
        need(root.is_dir() and not root.is_symlink(), 'test_root_invalid')
        for name in names:
            path = root / name
            need(path.is_file() and not path.is_symlink(), 'test_source_invalid')
            sources['tests/' + name] = path
    manifest = {name: sha(path) for name, path in sources.items()}
    need(digest({name: manifest[name] for name in ('tests/' + n for n in PRIOR_TEST_NAMES)})
         == PRIOR_OVERLAY_SHA and digest(manifest) == OVERLAY_SHA, 'test_source_pin_drift')
    return sources, manifest


def validate_candidate(manifest):
    need(len(manifest) == 481 and digest(manifest) == CANDIDATE_SHA
         and all(manifest.get(name) == pin for name, pin in APPLICATION_PINS.items()),
         'episode_candidate_pin_drift')
    return manifest


def prepare(destination):
    need(not destination.exists() and not destination.is_symlink(), 'output_already_exists')
    sources, manifest = test_sources()
    base = load_base('local')
    need(LOCAL_MANIFEST.is_file() and not LOCAL_MANIFEST.is_symlink(), 'local_manifest_missing')
    candidate = validate_candidate(json.loads(LOCAL_MANIFEST.read_text()))
    need(LOCAL_CANDIDATE.is_dir() and not LOCAL_CANDIDATE.is_symlink(), 'local_candidate_invalid')
    for name, pin in candidate.items():
        if name in manifest:
            continue  # Exact reviewed overlay replaces these baseline test files.
        path = LOCAL_CANDIDATE / name
        need(not Path(name).is_absolute() and '..' not in Path(name).parts
             and path.is_file() and not path.is_symlink() and sha(path) == pin,
             'local_candidate_source_pin_drift')
    tests = {name: pin for name, pin in {**candidate, **manifest}.items() if name.startswith('tests/')}
    need(len(tests) == EFFECTIVE_TEST_FILES and digest(tests) == EFFECTIVE_TEST_SHA,
         'effective_test_inventory_drift')
    # All source admissions happen before creating the directory. On a partial
    # copy failure remove only this invocation's freshly created directory.
    destination.mkdir(mode=0o700)
    try:
        for name, source in sources.items():
            target = destination / 'overlay' / name
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o400)
        need(base.inventory(destination / 'overlay') == manifest, 'overlay_copy_drift')
        shutil.copyfile(Path(__file__), destination / SELF.name)
        (destination / SELF.name).chmod(0o400)
        base.write_json(destination / 'test-overlay.json', manifest)
        base.write_json(destination / 'reviewed-candidate.json', candidate)
        return {'status': 'prepared_not_uploaded', 'test_overrides': len(manifest),
                'overlay_sha256': OVERLAY_SHA, 'candidate_sha256': CANDIDATE_SHA,
                'application_pins': APPLICATION_PINS,
                'effective_test_inventory_sha256': digest(tests),
                'effective_test_files': len(tests),
                'expected_collected': EXPECTED_COLLECTED,
                'controller_sha256': sha(destination / SELF.name)}
    except BaseException:
        shutil.rmtree(destination)
        raise


def configure_base(context):
    base = load_base(context)
    base.ROOT, base.SELF, base.WORK = ROOT, SELF, WORK
    base.OVERLAY, base.CANDIDATE = OVERLAY, CANDIDATE
    base.CANDIDATE_SHA, base.REVIEWED_OVERLAY_SHA = CANDIDATE_SHA, OVERLAY_SHA
    base.TEST_NAMES = TEST_NAMES
    base.FULL_SECONDS, base.HOST_SECONDS = FULL_SECONDS, HOST_SECONDS
    base.MIN_COLLECTED = EXPECTED_COLLECTED
    original_configure, original_project = base.configure, base.project_worker
    original_run_pytest = base.run_pytest
    original_pins = base.pins

    def source_pins(*_):
        return validate_candidate(base.inventory(CANDIDATE))

    def replay_gate(_proof, helper):
        path = REPLAY_ROOT / 'result.json'
        need(path.is_file() and not path.is_symlink() and sha(path) == REPLAY_RESULT_SHA,
             'episode_replay_receipt_pin_drift')
        result = helper.read_json(path)
        need(result.get('status') == 'verified' and all(result.get(name) is True for name in (
            'source_unchanged', 'exact_vectors_preserved', 'semantic_rows_unchanged',
            'repeat_noop', 'reopen_aligned', 'health_clean')), 'episode_replay_not_verified')
        return REPLAY_RESULT_SHA

    def configure(helper):
        command, mounts = original_configure(helper)
        extra = (str(BASE_SUITE), str(BASE_CONTAINER_SUITE), False)
        mounts.append(extra)
        position = command.index('--workdir')
        command[position:position] = ['--mount', 'type=bind,src=' + extra[0] + ',dst=' + extra[1] + ',readonly']
        command[command.index('--name') + 1] = 'hymem-episode-shadow-pytest-v2'
        return command, mounts

    def project_worker(raw):
        result = original_project(raw)
        need(result['collect']['collected'] == EXPECTED_COLLECTED, 'episode_collection_count_drift')
        need(result['test_inventory_sha256'] == EFFECTIVE_TEST_SHA, 'effective_test_inventory_drift')
        return result

    def pins(proof, shared, helper):
        return {**original_pins(proof, shared, helper),
                'episode_shadow_replay_result_sha256': REPLAY_RESULT_SHA,
                'new_candidate_replay_verified': True,
                'application_pins': APPLICATION_PINS,
                'effective_test_files': EFFECTIVE_TEST_FILES,
                'effective_test_inventory_sha256': EFFECTIVE_TEST_SHA,
                'expected_collected': EXPECTED_COLLECTED}

    def run_pytest(mode):
        rc = original_run_pytest(mode)
        if mode == 'collect' and rc == 0:
            counts = json.loads((base.CONTAINER_WORK / 'collect-counts.json').read_text())
            need(counts.get('collected') == EXPECTED_COLLECTED,
                 'episode_collection_count_drift')
        return rc

    base.source_pins, base.replay_gate = source_pins, replay_gate
    base.configure, base.project_worker = configure, project_worker
    base.run_pytest = run_pytest
    base.pins = pins
    return base


def remote_install(base):
    need(REVIEWED_REMOTE_EXECUTION, 'episode_suite_root_review_pending')
    need(not WORK.exists() and not WORK.is_symlink()
         and not (ROOT / 'install.json').exists(), 'episode_install_already_attempted')
    return base.remote('remote-install')


def worker(base):
    validate_candidate(json.loads(base.SOURCE_MANIFEST.read_text()))
    return base.worker()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'remote-install', 'remote-launch',
        'remote-status', 'supervise', 'worker', 'pytest-collect', 'pytest-full'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        if args.action == 'prepare':
            need(args.output is not None, 'output_required')
            result = prepare(args.output)
        else:
            if args.action in ('remote-install', 'remote-launch', 'supervise'):
                need(REVIEWED_REMOTE_EXECUTION, 'episode_suite_root_review_pending')
            context = 'container' if args.action in ('worker', 'pytest-collect', 'pytest-full') else 'host'
            base = configure_base(context)
            if args.action == 'remote-install':
                result = remote_install(base)
            elif args.action == 'worker':
                result = worker(base)
            elif args.action.startswith('pytest-'):
                return base.pytest_stage(args.action.removeprefix('pytest-'))
            else:
                result = base.remote(args.action)
    except BaseException:
        result = {'status': 'failed_inspect_private_artifacts'}
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get('status') in ('prepared_not_uploaded', 'installed_not_launched',
        'detached_supervisor_started', 'passed', 'not_terminal') else 1


if __name__ == '__main__':
    raise SystemExit(main())
