"""Root's read-only, network-disabled replay of one paired diagnostic arm."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import socket
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--arm', choices=['baseline', 'candidate'], required=True)
    args = parser.parse_args()
    root, candidate = args.root, args.candidate
    assert hashlib.sha256((root / 'launch-receipt.json').read_bytes()).hexdigest() == '87ba2faaedc6e843d48323bd5d865defb1110d863522dd7a6bc51957f7f9fe79'
    receipt = json.loads((root / 'launch-receipt.json').read_text())
    for name, expected in receipt['source_pins'].items():
        path = root / name
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    sys.path[:0] = [str(candidate), str(root)]
    import luna_grounding_candidate as builder
    import luna_grounding_cases as cases
    import luna_subscription_lme_warm_v2 as pinned
    from benchmarks import extraction_canary as canary
    from hymem.extraction import chunk
    for module in (canary, chunk):
        assert Path(module.__file__).resolve().is_relative_to(candidate.resolve())
    original = Path('/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky/lme-r9-full-suite-v1/candidate')
    if args.arm == 'baseline':
        assert candidate == original
        builder.pilot_module().verify_inventory(candidate, root / 'headless-source-map.json', receipt['source_pins']['headless-source-map.json'])
    else:
        builder.verify_derived(original, candidate, candidate.parent / 'headless-grounding-source-map.json', root / 'headless-source-map.json')
    def no_network(*a, **k):
        raise AssertionError('network_forbidden')
    socket.socket = no_network
    state = json.loads((root / 'run/private-result.json').read_text())
    assert state['completed_and_clean'] is True
    selected = [x for x in state['units'] if x['arm'] == args.arm]
    assert len(selected) == 22
    calls = 0
    passes = 0
    for unit in selected:
        evidence = json.loads((root / 'run' / ('private-' + unit['id'] + '.json')).read_text())
        requests, responses = evidence['requests'], evidence['responses']
        assert len(requests) == len(responses) == unit['completion_calls']
        class Replay:
            observed_turns = 0
            observed_tokens = 0
            usage_complete = True
            def complete(self, request):
                index = self.observed_turns
                assert index < len(responses)
                assert asdict(request) == requests[index]['after_request'], 'wire_request_mismatch'
                self.observed_turns += 1
                self.observed_tokens += 1
                return responses[index]
        client = Replay()
        if unit['kind'] == 'canary':
            grade = pinned.old.experimental_canary(canary, chunk, client)
        else:
            case = next(c for c in cases.CASES if c.case_id == unit['case_id'])
            result = chunk.extract_chunk(client, case.text, source_records=case.source_records, completion_call_limit=8)
            grade = cases.grade_case(case.case_id, result)
        assert client.observed_turns == len(responses)
        assert grade['passed'] == unit['passed']
        passes += grade['passed'] is True
        calls += client.observed_turns
    print(json.dumps({'arm': args.arm, 'exact_wire_requests_replayed': calls,
                      'units_replayed': len(selected), 'passed': passes, 'new_model_calls': 0}))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(json.dumps({'verified': False, 'failure': 'offline_replay_mismatch', 'new_model_calls': 0}))
        raise SystemExit(1)
