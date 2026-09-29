"""Read-only, network-disabled replay of the one stopped observed canary."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace


ROOT = Path('/home/atta/.hymem-luna-lme-observed-4e8qycex')
RECEIPT = 'ec4b060b2122c742c4e0dac6d95037ae04693ee65d5866523a0f9b4a194fadfc'
RESULT = '0730f970ec291473121bde497f855bfa4e49bda2b7b108a60d7efd7e55058147'
PHASE = 'setup'


def private_bytes(path, maximum, *, source_only=False):
    state = path.lstat()
    assert stat.S_ISREG(state.st_mode) and state.st_uid == 1000
    assert (source_only or not state.st_mode & 0o077) and state.st_size <= maximum
    return path.read_bytes()


def main():
    global PHASE
    def readonly_audit(event, args):
        if event.startswith(('socket.connect', 'socket.bind', 'socket.getaddrinfo',
                             'socket.gethostby', 'subprocess.', 'os.exec',
                             'os.spawn', 'os.posix_spawn')) or event == 'os.system':
            raise AssertionError('network_or_process_forbidden')
        if event == 'open' and type(args[2]) is int and args[2] & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError('write_forbidden')
    sys.addaudithook(readonly_audit)
    root_state = ROOT.lstat()
    assert stat.S_ISDIR(root_state.st_mode) and root_state.st_uid == 1000
    assert not root_state.st_mode & 0o077
    PHASE = 'receipt'
    raw = private_bytes(ROOT / 'launch-receipt.json', 100000)
    assert hashlib.sha256(raw).hexdigest() == RECEIPT
    receipt = json.loads(raw)
    PHASE = 'source_pins'
    for name, digest in receipt['source_sha256'].items():
        assert Path(name).name == name
        assert hashlib.sha256(private_bytes(ROOT / name, 2000000, source_only=True)).hexdigest() == digest
    PHASE = 'result'
    result_raw = private_bytes(ROOT / 'run/private-result.json', 1000000)
    assert hashlib.sha256(result_raw).hexdigest() == RESULT
    recorded = json.loads(result_raw)['canary']
    PHASE = 'evidence'
    evidence_raw = private_bytes(ROOT / 'run/private-canary-evidence.json', 2000000)
    evidence = json.loads(evidence_raw)
    requests, responses = evidence['requests'], evidence['responses']
    assert len(requests) == len(responses) == 8
    assert evidence['provider_output_truncations'] == 0
    PHASE = 'builder'
    sys.path[:0] = [str(ROOT), str(ROOT / 'candidate')]
    import luna_grounding_candidate as builder
    proof = builder.verify_derived(Path(receipt['original_candidate']), ROOT / 'candidate',
        ROOT / 'headless-grounding-source-map.json', ROOT / 'headless-source-map.json')
    assert proof['source_files'] == 508
    PHASE = 'candidate_imports'
    from benchmarks import extraction_canary as canary
    from hymem.extraction import chunk
    from hymem.extraction.jsonio import loads_exact_or_fenced
    import luna_subscription_pilot as pilot
    for module in (canary, chunk):
        assert Path(module.__file__).resolve().is_relative_to((ROOT / 'candidate').resolve())
    saved = []
    original_extract = chunk.extract_chunk
    def capture(*args, **kwargs):
        outcome = original_extract(*args, **kwargs)
        saved.append(outcome)
        return outcome
    chunk.extract_chunk = capture
    class Replay:
        observed_turns = 0
        observed_tokens = 0
        usage_complete = True
        def complete(self, request):
            index = self.observed_turns
            assert index < 8
            canonical = lambda value: json.dumps(value, ensure_ascii=False,
                sort_keys=True, separators=(',', ':')).encode('utf-8')
            assert canonical(asdict(request)) == canonical(requests[index])
            self.observed_turns += 1
            # Synthetic accounting only satisfies the replay gate; not usage.
            self.observed_tokens += 1
            return responses[index]
    PHASE = 'exact_replay'
    replay = Replay()
    grade = pilot.experimental_canary(canary, chunk, replay)
    PHASE = 'replay_comparison'
    assert grade == recorded and replay.observed_turns == 8 and len(saved) == 1
    PHASE = 'path_projection'
    objects = [SimpleNamespace(**request) for request in requests]
    pairs = list(zip(objects, responses))
    path = canary._request_execution_path(objects, pairs)
    path['provider_output_truncations'] = 0
    core, _ = pilot._core_path_and_types(canary, pairs, path)
    expected_path = canary.extraction_canary_policy()['normal_execution_path']
    delta = {}
    for key in expected_path:
        if key == 'source_message_ids_seen':
            continue
        if core[key] != expected_path[key]:
            assert type(core[key]) in (int, bool) and type(expected_path[key]) in (int, bool)
            delta[key] = {'observed': core[key], 'expected': expected_path[key]}
    call_shapes = []
    for request, response in pairs:
        parsed = loads_exact_or_fenced(response)
        stage = ('omission' if 'OMISSION VERIFICATION PASS' in request.system else
                 'empty' if 'EMPTY VERIFICATION PASS' in request.system else 'primary')
        singleton = canary._request_execution_path([request], [(request, response)])
        singleton['provider_output_truncations'] = 0
        single_core, _ = pilot._core_path_and_types(canary, [(request, response)], singleton)
        call_shapes.append({'stage': stage,
            'json_object': type(parsed) is dict,
            'triples': len(parsed['triples']) if type(parsed) is dict and type(parsed.get('triples')) is list else None,
            'table_requested': single_core['table_claim_requests'] > 0,
            'prose_requested': single_core['prose_claim_requests'] > 0,
            'table_exact_emissions': single_core['table_claim_exact_context_emissions'],
            'prose_exact_emissions': single_core['prose_claim_exact_context_emissions']})
    extracted = saved[0]
    reason = extracted.failure_reason
    assert reason is None or reason in chunk._FAILURE_REASONS
    field_names = ('subject', 'predicate', 'object', 'polarity', 'source_message_id')
    claim_fields = []
    for expected in canary._CANARY_EXPECTED_CLAIMS:
        wanted = (expected[0], expected[2], expected[3], expected[5], expected[6])
        candidates = [triple for triple in extracted.triples
                      if triple.source_message_id == expected[6]]
        claim_fields.append({'same_source_triples': len(candidates),
            'field_matches': [{field: getattr(triple, field) == target
                              for field, target in zip(field_names, wanted)}
                             for triple in candidates]})
    return {'verified': True, 'exact_wire_requests_replayed': replay.observed_turns,
        'recorded_grade_reproduced': True, 'new_model_calls': 0,
        'extraction_failed': extracted.failed, 'extraction_failure_reason': reason,
        'final_triples': len(extracted.triples), 'final_markers': len(extracted.markers),
        'source_ids_match_expected': core['source_message_ids_seen'] == expected_path['source_message_ids_seen'],
        'claim_field_checks': claim_fields,
        'path_delta': delta, 'calls': call_shapes, 'raw_text_exported': False}


if __name__ == '__main__':
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            report = main()
        print(json.dumps(report, sort_keys=True))
    except BaseException:
        print(json.dumps({'verified': False, 'failure': 'offline_replay_mismatch',
                          'phase': PHASE, 'new_model_calls': 0, 'raw_text_exported': False}))
        raise SystemExit(1)
