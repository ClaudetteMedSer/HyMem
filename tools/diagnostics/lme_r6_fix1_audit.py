"""Read-only revalidation of the unchanged failed R5 run with one audited fix.

Mount the original R5 source/package/results/data and a reviewed replacement
protocol module read-only, with network disabled and no credential mount.
This validates a failed run; it never claims the failed question succeeded.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import types

R5_PIN = 'c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'
PACKAGE_PIN = 'fa471c38440dd70e8527b75171b1af8fc9b48ee98081df9738b45093f41efffc'
VERIFIER_PIN = '36531f10a6b484482d373bbb7a0c344e2710a737ee80ad2045bbdf542e94a8c8'


def require(value, code):
    if not value:
        raise RuntimeError(code)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol-sha256', required=True)
    args = parser.parse_args()
    require(re.fullmatch('[0-9a-f]{64}', args.protocol_sha256), 'audit_protocol_pin_shape')
    require(sha(Path('/r5/manifest.json')) == R5_PIN, 'audit_r5_manifest')
    raw = Path('/diag/q1_stock_validate.py').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == VERIFIER_PIN, 'audit_original_verifier')
    verifier = types.ModuleType('r6_audit_original_postvalidator')
    verifier.__file__ = '/diag/q1_stock_validate.py'
    exec(compile(raw, verifier.__file__, 'exec'), verifier.__dict__)
    run = verifier.load_run_helper(PACKAGE_PIN)
    manifest = run.validate_package(PACKAGE_PIN)
    require(manifest['approved_source_manifest_sha256'] == R5_PIN, 'audit_source_binding')
    run.verify_source(manifest)
    require(run.sha(run.DATA) == run.DATASET_SHA, 'audit_dataset')
    sys.path[:0] = [str(run.SOURCE), str(run.SOURCE / 'benchmarks')]
    from benchmarks.lme_registry import _load_registry_artifact
    from benchmarks.lme_protocol import validate_strict_artifact as baseline_validate
    from benchmarks.archive_evidence import checkpoint_attestation
    from benchmarks.strictness import (
        BenchmarkIntegrityError, aggregate_usage_snapshots, bounded_failure_text,
        reconcile_results,
    )

    benchmark = run.OUTPUT / 'benchmark'
    pointer = benchmark / 'longmemeval-v2-hymem.json'
    data, name, _digest, compatibility = _load_registry_artifact(pointer)
    require(Path(name).name == name and compatibility == 'pointer-target-digest-validated',
            'audit_archive_pointer')
    archive = benchmark / name
    checkpoint = benchmark / 'checkpoint.json'
    archive_sha, checkpoint_sha = sha(archive), sha(checkpoint)
    try:
        baseline_validate(data, path=archive, require_scored=True)
    except BenchmarkIntegrityError as exc:
        require(str(exc) == 'LongMemEval indexing failure row fields differ',
                'audit_unexpected_baseline_failure')
    else:
        raise RuntimeError('audit_baseline_did_not_reproduce')

    replacement = Path('/audit/lme_protocol.py')
    require(replacement.resolve() == replacement and replacement.is_file()
            and sha(replacement) == args.protocol_sha256, 'audit_replacement_pin')
    # Keep the R5 execution namespace/source intact. Only this explicitly named
    # audit validator is corrected, with its own file path and recorded digest.
    spec = importlib.util.spec_from_file_location('benchmarks.r6_fix1_audit_protocol', replacement)
    corrected = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(corrected)
    validated = corrected.validate_strict_artifact(data, path=archive, require_scored=True)
    supervisor = verifier.read_json(run.OUTPUT / 'supervisor-summary.json')
    require(verifier.read_json(run.OUTPUT / 'invocation/terminal.json') == supervisor['outcome'],
            'audit_terminal_binding')
    result = verifier.summarize(
        data, validated, verifier.read_json(checkpoint), supervisor,
        manifest=manifest, manifest_sha=PACKAGE_PIN, archive_sha=archive_sha,
        checkpoint_sha=checkpoint_sha, checkpoint_attestation=checkpoint_attestation,
        reconcile_results=reconcile_results, bounded_failure_text=bounded_failure_text,
        aggregate_usage_snapshots=aggregate_usage_snapshots, run=run,
    )
    require(result['counts']['completed'] == 7 and result['counts']['failed'] == 1
            and result['counts']['missing'] == 0
            and result['benchmark_completed_without_faults'] is False,
            'audit_failure_not_preserved')
    require(sha(archive) == archive_sha and sha(checkpoint) == checkpoint_sha,
            'audit_artifacts_changed')
    run.validate_package(PACKAGE_PIN)
    run.verify_source(manifest)
    require(run.sha(run.DATA) == run.DATASET_SHA
            and sha(replacement) == args.protocol_sha256, 'audit_inputs_changed')
    print(json.dumps({
        'schema': 'r6-fix1-saved-r5-failed-run-audit-v1',
        'artifact_audit_passed': True, 'baseline_rejection_reproduced': True,
        'execution_source_manifest_sha256': R5_PIN,
        'corrected_validator_sha256': args.protocol_sha256,
        'artifacts_unchanged': True, 'paid_rerun': False, 'new_provider_calls': 0,
        'result': result,
    }, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never emit arbitrary exception/source/provider text across the boundary.
        print(json.dumps({'schema': 'r6-fix1-saved-r5-failed-run-audit-v1',
                          'artifact_audit_passed': False, 'exception_type': type(exc).__name__,
                          'new_provider_calls': 0}, sort_keys=True))
        raise SystemExit(1)
