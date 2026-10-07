"""Read-only private replay of every claim-task return against the fixed schedule."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

sys.dont_write_bytecode = True
HEX = re.compile(r'[0-9a-f]{64}\Z')


def _regular(path: Path, cap: int) -> bytes:
    state = path.lstat()
    if not stat.S_ISREG(state.st_mode) or state.st_size > cap or state.st_size < 0:
        raise ValueError('private_record_invalid')
    return path.read_bytes()


def replay(root: Path, receipt_sha: str, entry_sha: str) -> dict:
    entry_path = root / 'code/tools/diagnostics/luna_claim_task_run_v1.py'
    if not HEX.fullmatch(entry_sha) or hashlib.sha256(_regular(entry_path, 100_000)).hexdigest() != entry_sha:
        raise ValueError('entry_pin_invalid')
    code = root / 'code'
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(root / 'candidate'))
    spec = importlib.util.spec_from_file_location('pinned_claim_task_replay_entry', entry_path)
    if spec is None or spec.loader is None:
        raise ValueError('entry_load_invalid')
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    receipt, loaded, _, retained = entry.preflight(root, receipt_sha, True)
    host, core, old_core, cases, canary, chunk, semantic, transport, warm, concurrent = loaded
    result_path = root / 'run/private-result.json'
    raw_result = _regular(result_path, 300_000)
    result = json.loads(raw_result)
    core.validate_public_result(result)
    completed = result['diagnostic_completed']
    expected_derivation = []
    batches = core.derive_canary_batches(canary_module=canary, chunk_module=chunk,
        semantic_probe_module=old_core, candidate=root / 'candidate', retained=retained,
        record=expected_derivation.append)
    directory = root / 'run/private-journal'
    if not directory.is_dir() or directory.is_symlink():
        raise ValueError('journal_invalid')
    files = sorted(directory.iterdir())
    if not 10 <= len(files) <= 256:
        raise ValueError('journal_count_invalid')
    grouped = {}
    derivation = []
    for sequence, path in enumerate(files, 1):
        match = re.fullmatch(r'(\d{4})-(derivation|unit-\d{2})\.json', path.name)
        if match is None or int(match[1]) != sequence:
            raise ValueError('journal_sequence_invalid')
        item = json.loads(_regular(path, 2_000_000))
        if type(item) is not dict:
            raise ValueError('journal_shape_invalid')
        if match[2] == 'derivation':
            if grouped:
                raise ValueError('derivation_order_invalid')
            derivation.append(item)
        else:
            grouped.setdefault(match[2], []).append(item)
    if (derivation != json.loads(core._canonical(expected_derivation)) or
            set(grouped) != {f'unit-{i:02d}' for i in range(len(grouped))} or
            len(grouped) > core.MAX_NEW_TURNS or
            len(grouped) != result['attempted_units'] or
            len(grouped) not in {len(result['units']), len(result['units']) + 1}):
        raise ValueError('journal_derivation_or_schedule_invalid')
    derived = next(x for x in derivation if x.get('phase') == 'canary_batches_derived')
    if (derived.get('table_batch_sha256') != batches[0].batch_sha256 or
            derived.get('prose_batch_sha256') != batches[1].batch_sha256 or
            derived.get('ordinary_replays') != 8 or
            derived.get('synthetic_table_advancement') is not True):
        raise ValueError('derivation_binding_invalid')
    fixture = cases.cases()
    valid = malformed = 0
    cumulative_tokens = 0
    for ordinal, unit in enumerate(core.schedule()[:len(result['units'])]):
        events = grouped[f'unit-{ordinal:02d}']
        item = result['units'][ordinal]
        phases = [event.get('phase') for event in events]
        if item['outcome'] == 'valid':
            expected_phases = ['before_dispatch', 'response_returned',
                               'validated_result', 'unit_finished']
        else:
            expected_phases = ['before_dispatch', 'response_returned', 'unit_finished']
        if phases != expected_phases:
            raise ValueError('journal_phase_invalid')
        triples, sources, expected_hash = core._unit_input(unit, fixture, batches)
        request, batch = core.contract.build_arm_request(unit.arm, triples, sources)
        core.contract.validate_arm_request(unit.arm, request, batch)
        schema_hash = core._sha(core._canonical(core.contract.build_arm_output_schema(unit.arm, batch)))
        before = events[0]
        expected_before = {'phase': 'before_dispatch', 'ordinal': ordinal,
                      'kind': unit.kind, 'index': unit.index, 'arm': unit.arm,
                      'request': asdict(request), 'batch': batch.canonical_json,
                      'batch_sha256': batch.batch_sha256,
                      'output_schema_sha256': schema_hash}
        if before != json.loads(core._canonical(expected_before)):
            raise ValueError('request_or_batch_replay_mismatch')
        if expected_hash is not None and batch.batch_sha256 != expected_hash:
            raise ValueError('canary_batch_replay_mismatch')
        response_event = events[1]
        if set(response_event) != {'phase', 'response'} or type(response_event['response']) is not str:
            raise ValueError('response_record_invalid')
        try:
            parsed = core.contract.parse_arm_response(unit.arm, response_event['response'], batch)
        except core.contract.v2.GroundingContractError:
            if item['outcome'] != 'malformed' or item['result'] is not None:
                raise ValueError('malformed_replay_mismatch')
            malformed += 1
        else:
            if (item['outcome'] != 'valid' or
                    events[2] != json.loads(core._canonical(
                        {'phase': 'validated_result', 'result': asdict(parsed)})) or
                    item['result'] != core._projection(parsed)):
                raise ValueError('valid_replay_mismatch')
            valid += 1
        expected_item = {'ordinal': ordinal, 'kind': unit.kind, 'index': unit.index,
                         'arm': unit.arm, **core._expected_metadata(unit, fixture),
                         'batch_sha256': batch.batch_sha256, 'schema_sha256': schema_hash}
        if any(item.get(key) != value for key, value in expected_item.items()):
            raise ValueError('public_unit_replay_mismatch')
        finished = events[-1]
        cumulative_tokens += item['known_tokens'] or 0
        snapshot = finished['budget']
        questions = snapshot.get('questions')
        if (type(questions) is not dict or
                set(questions) != {f'unit-{i:02d}' for i in range(ordinal + 1)}):
            raise ValueError('question_prefix_invalid')
        for prior in range(ordinal + 1):
            question = questions[f'unit-{prior:02d}']
            public = result['units'][prior]
            if (type(question) is not dict or
                    set(question) != {'turns', 'known_tokens', 'in_flight',
                                      'usage_complete', 'stopped'} or
                    question['turns'] != 1 or
                    question['known_tokens'] != (public['known_tokens'] or 0) or
                    question['usage_complete'] != public['usage_complete'] or
                    type(question['stopped']) is not bool or
                    type(question['in_flight']) is not int or question['in_flight'] < 0):
                raise ValueError('question_accounting_replay_mismatch')
        expected_stop = (result['stop_code'] if not completed and
                         ordinal == len(result['units']) - 1 and
                         len(grouped) == len(result['units']) else None)
        if (set(finished) != {'phase', 'stop_code', 'budget', 'result'} or
                finished['stop_code'] != expected_stop or finished['result'] != item or
                finished['budget']['turns'] != ordinal + 1 or
                finished['budget']['known_tokens'] != cumulative_tokens or
                finished['budget']['usage_complete'] != all(
                    x['usage_complete'] for x in result['units'][:ordinal + 1]) or
                type(finished['budget']['reserved']) is not int or
                type(finished['budget']['in_flight']) is not int or
                finished['budget']['reserved'] < 0 or
                finished['budget']['in_flight'] < 0 or
                finished['budget']['questions'][f'unit-{ordinal:02d}']['turns'] != 1 or
                finished['budget']['questions'][f'unit-{ordinal:02d}']['known_tokens'] != item['known_tokens']):
            raise ValueError('accounting_replay_mismatch')
        if ((completed or ordinal < len(result['units']) - 1) and
                finished['budget']['stopped'] is not False):
            raise ValueError('intermediate_budget_stopped')
        if ordinal < len(result['units']) - 1 and (
                snapshot['reserved'] != 0 or snapshot['in_flight'] != 0):
            raise ValueError('intermediate_flight_invalid')
    returned = valid + malformed
    if (returned != len(result['units']) or malformed != result['malformed_units'] or
            (completed and (returned != 29 or len(grouped) != 29 or
                sum(x['known_tokens'] for x in result['units']) != result['paid_budget']['known_tokens'] or
                result['paid_budget']['turns'] != 29))):
        raise ValueError('campaign_replay_mismatch')
    if len(grouped) > len(result['units']):
        ordinal = len(result['units'])
        events = grouped[f'unit-{ordinal:02d}']
        phases = [event.get('phase') for event in events]
        if phases not in (['before_dispatch', 'unit_finished'],
                          ['before_dispatch', 'response_returned', 'unit_finished'],
                          ['before_dispatch', 'response_returned',
                           'validated_result', 'unit_finished']):
            raise ValueError('partial_unit_phase_invalid')
        unit = core.schedule()[ordinal]
        triples, sources, _ = core._unit_input(unit, fixture, batches)
        request, batch = core.contract.build_arm_request(unit.arm, triples, sources)
        schema_hash = core._sha(core._canonical(core.contract.build_arm_output_schema(unit.arm, batch)))
        expected_before = {'phase': 'before_dispatch', 'ordinal': ordinal,
            'kind': unit.kind, 'index': unit.index, 'arm': unit.arm,
            'request': asdict(request), 'batch': batch.canonical_json,
            'batch_sha256': batch.batch_sha256, 'output_schema_sha256': schema_hash}
        if events[0] != json.loads(core._canonical(expected_before)):
            raise ValueError('partial_request_mismatch')
        if 'response_returned' in phases:
            if set(events[1]) != {'phase', 'response'} or type(events[1]['response']) is not str:
                raise ValueError('partial_response_invalid')
            try:
                parsed = core.contract.parse_arm_response(unit.arm, events[1]['response'], batch)
            except core.contract.v2.GroundingContractError:
                if 'validated_result' in phases:
                    raise ValueError('partial_validated_malformed')
            else:
                if 'validated_result' in phases and events[2] != json.loads(core._canonical(
                        {'phase': 'validated_result', 'result': asdict(parsed)})):
                    raise ValueError('partial_validated_mismatch')
            returned += 1
        if (set(events[-1]) != {'phase', 'stop_code', 'budget', 'result'} or
                events[-1].get('result') is not None or
                events[-1].get('stop_code') != result['stop_code']):
            raise ValueError('partial_accounting_invalid')
        last_budget = events[-1]['budget']
        if (type(last_budget) is not dict or type(last_budget.get('questions')) is not dict or
                set(last_budget['questions']) != {f'unit-{i:02d}' for i in range(ordinal + 1)} or
                type(last_budget.get('stopped')) is not bool or
                type(last_budget.get('usage_complete')) is not bool or
                type(last_budget.get('known_tokens')) is not int or
                type(last_budget.get('turns')) is not int or
                type(last_budget.get('reserved')) is not int or
                type(last_budget.get('in_flight')) is not int):
            raise ValueError('partial_budget_shape_invalid')
        for prior in range(ordinal):
            question = last_budget['questions'][f'unit-{prior:02d}']
            public = result['units'][prior]
            if (type(question) is not dict or
                    set(question) != {'turns', 'known_tokens', 'in_flight',
                                      'usage_complete', 'stopped'} or
                    question != {'turns': 1,
                                 'known_tokens': public['known_tokens'] or 0,
                                 'in_flight': 0,
                                 'usage_complete': public['usage_complete'],
                                 'stopped': False}):
                raise ValueError('partial_prior_question_mismatch')
        partial = last_budget['questions'][f'unit-{ordinal:02d}']
        if (type(partial) is not dict or
                set(partial) != {'turns', 'known_tokens', 'in_flight',
                                 'usage_complete', 'stopped'} or
                type(partial['turns']) is not int or partial['turns'] not in (0, 1) or
                type(partial['known_tokens']) is not int or partial['known_tokens'] < 0 or
                type(partial['in_flight']) is not int or partial['in_flight'] not in (0, 1) or
                type(partial['usage_complete']) is not bool or
                type(partial['stopped']) is not bool or
                last_budget['turns'] != ordinal + partial['turns'] or
                last_budget['known_tokens'] != cumulative_tokens + partial['known_tokens'] or
                last_budget['in_flight'] != partial['in_flight'] or
                not 0 <= last_budget['reserved'] <= partial['in_flight'] or
                last_budget['usage_complete'] !=
                    (all(x['usage_complete'] for x in result['units']) and
                     partial['usage_complete'])):
            raise ValueError('partial_question_accounting_mismatch')
    if grouped:
        last_budget = grouped[f'unit-{len(grouped)-1:02d}'][-1]['budget']
        for key, value in result['paid_budget'].items():
            if last_budget.get(key) != value:
                raise ValueError('final_accounting_replay_mismatch')
    return {'schema': 'luna-claim-task-probe-replay-v1', 'verified': True,
            'receipt_sha256': receipt_sha,
            'private_result_sha256': hashlib.sha256(raw_result).hexdigest(),
            'replayed_units': len(result['units']), 'returned_responses': returned,
            'complete': completed, 'valid_units': valid, 'malformed_units': malformed,
            'new_model_calls': 0, 'semantic_accuracy_accepted': False,
            'full_lme_ready': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args(argv)
    def deny(*a, **k):
        raise RuntimeError('replay_external_action_forbidden')
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    subprocess.Popen = deny
    def audit(event, values):
        if event == 'open':
            mode, flags = values[1:3]
            if ((type(mode) is str and any(c in mode for c in 'wax+')) or
                    (type(flags) is int and flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC))):
                deny()
        if event in {'os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.symlink',
                     'os.link', 'os.chmod', 'os.chown', 'os.utime', 'os.truncate',
                     'os.system', 'os.fork', 'os.exec', 'os.posix_spawn',
                     'subprocess.Popen', 'socket.connect', 'socket.getaddrinfo'}:
            deny()
    sys.addaudithook(audit)
    try:
        print(json.dumps(replay(Path(args.root), args.receipt_sha256,
                                args.entry_sha256), sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps({'verified': False, 'new_model_calls': 0,
                          'reason': 'offline_replay_failed'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
