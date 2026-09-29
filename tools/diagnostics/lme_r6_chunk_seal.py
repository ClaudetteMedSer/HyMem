"""Generate an immutable, credential-free replay package from reviewed source."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

BASELINE_PIN = 'c67a4ba8bd9c1b29259484151a6c8687a17bacd82f64d54092f59327907b9ffc'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--baseline-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    raw = args.baseline_manifest.read_bytes()
    assert digest(raw) == BASELINE_PIN
    baseline = json.loads(raw)
    source = {}
    for relative in baseline['source_sha256']:
        path = args.tree / relative
        assert path.resolve() == path and path.is_file()
        source[relative] = path.read_bytes()
    assert len(source) == 231
    # Closed known source inventory; unreviewed additions require a new seal.
    for folder in ('hymem', 'benchmarks'):
        actual = {p.relative_to(args.tree).as_posix() for p in (args.tree/folder).rglob('*.py')}
        expected = {p for p in source if p.startswith(folder+'/') and p.endswith('.py')}
        assert actual == expected
    worker = (root/'lme_r6_chunk_replay.py').read_bytes()
    support = (root/'lme_summary_recovery_v1/worker.py').read_bytes()
    supervisor = (root/'lme_sample8_v1/bundle/supervised_invocation.py').read_bytes()
    assert digest(support) == 'bd8cc72c9bec26e391632af0f133d226e00f367807120b7a9a395e4dc55bf7a5'
    assert digest(supervisor) == '9bab7fc77e68cbea050b774791aee94893c7eb54d3cbb87f8b3e7bee33ef85bc'
    manifest = {'schema': 'r6-fix2-retained-chunk-package-v1', 'baseline_manifest_sha256': BASELINE_PIN,
                'source_sha256': {p: digest(b) for p, b in sorted(source.items())},
                'worker_sha256': digest(worker), 'support_sha256': digest(support),
                'supervisor_sha256': digest(supervisor), 'paid_invocations': 1,
                'completion_cap': 96, 'http_attempt_cap': 288, 'timeout_seconds': 1200,
                'production_changes': False, 'original_store_writable': False}
    manifest_raw = (json.dumps(manifest, sort_keys=True, indent=2)+'\n').encode()
    files = {'candidate/'+p: b for p, b in source.items()}
    files.update({'diag/worker.py': worker, 'diag/manifest.json': manifest_raw,
                  'support/worker.py': support, 'support/supervised_invocation.py': supervisor})
    assert args.output.is_absolute() and args.output.resolve() == args.output
    assert not args.output.is_relative_to(args.tree) and not args.output.is_relative_to(root)
    args.output.mkdir(mode=0o755)
    for relative, body in files.items():
        path = args.output/relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        with path.open('xb') as stream:
            stream.write(body)
    archive = args.output.with_suffix('.tgz')
    with archive.open('xb') as stream:
        with tarfile.open(fileobj=stream, mode='w:gz', format=tarfile.USTAR_FORMAT) as tar:
            for relative in sorted(files):
                tar.add(args.output/relative, arcname=relative, recursive=False)
    print(json.dumps({'manifest_sha256': digest(manifest_raw), 'archive': str(archive),
                      'archive_sha256': digest(archive.read_bytes()), 'source_files': len(source),
                      'worker_sha256': digest(worker), 'api_calls': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
