"""Root's private, read-only, zero-inference rehearsal of sealed staged inputs."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

sys.dont_write_bytecode = True


def verify(root, receipt_sha, entry_sha):
    source = root / 'code/tools/diagnostics/luna_staged_run_v1.py'
    if not stat.S_ISREG(source.lstat().st_mode) or hashlib.sha256(source.read_bytes()).hexdigest() != entry_sha:
        raise ValueError('entry_pin_invalid')
    spec = importlib.util.spec_from_file_location('root_staged_rehearsal_entry', source)
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    receipt, loaded, proof, retained = entry.preflight(root, receipt_sha)
    host, core, old, cases, canary, chunk, semantic, transport, warm, concurrent = loaded
    if (root / 'launch-attempt.json').exists() or (root / 'run').exists():
        raise ValueError('unlaunched_root_required')
    derivation = []
    batches = core.derive_canary_batches(canary_module=canary, chunk_module=chunk,
        semantic_probe_module=old, candidate=root / 'candidate', retained=retained,
        record=derivation.append)
    calls, closed = [], []

    class Synthetic:
        def __init__(self, key, cap, budget):
            self.key, self.budget = key, budget
            budget.register(key, warm.BudgetLimits(*cap))

        @property
        def observed_turns(self):
            return self.budget.snapshot()['questions'][self.key]['turns']

        @property
        def observed_tokens(self):
            return self.budget.snapshot()['questions'][self.key]['known_tokens']

        @property
        def usage_complete(self):
            return self.budget.snapshot()['questions'][self.key]['usage_complete']

        def complete_stage(self, request, batch, stage, recheck):
            assert recheck is False
            if stage == 'original':
                core.staged.validate_original_request(request, batch)
                rows = [dict(index=i, original=dict(state='not_established', support=None))
                        for i in range(len(batch.triples))]
                payload = dict(schema='source-grounding-staged-original-v1',
                    batch_sha256=batch.batch_sha256, complete=True, originals=rows)
                bound = batch
            elif stage == 'alternatives':
                core.staged.validate_alternatives_request(request, batch)
                bound = batch.classification_batch
                rows = [dict(index=i, alternatives={p: dict(state='not_established', support=None)
                    for p in core.v4.PREDICATE_ORDER if p != bound.triples[i].predicate})
                    for i in batch.negative_indices]
                payload = dict(schema='source-grounding-staged-alternatives-v1',
                    batch_sha256=bound.batch_sha256, complete=True,
                    original_response_sha256=batch.original_response_sha256, alternatives=rows)
            else:
                raise ValueError('unexpected_stage')
            calls.append((self.key, stage, bound.batch_sha256))
            self.budget.reserve(self.key)
            self.budget.before_turn(self.key, dict(auth='chatgpt', model=warm.base.MODEL,
                config_isolation_admitted=True, inference_enabled=False,
                quota_windows=[dict(remaining_percent=90)]))
            self.budget.settle(self.key, used=7, turn_started=True)
            return json.dumps(payload)

        def close(self):
            closed.append(self.key)

    class MemoryJournal:
        def record(self, *args):
            pass

    out = core.run_campaign(concurrent=concurrent, warm=warm, binary='offline-no-executable',
        cases_module=cases, canary_batches=batches, journal=MemoryJournal(), client_factory=Synthetic)
    core.validate_public_result(out)
    assert out['diagnostic_completed'] and len(closed) == len(set(closed)) == 8
    assert len(calls) == 16 and out['malformed_units'] == 0
    assert out['paid_budget'] == dict(turns=16, known_tokens=112, usage_complete=True, in_flight=0, reserved=0)
    assert not out['semantic_accuracy_accepted'] and not out['full_lme_ready']
    assert [b.triples[0].predicate for b in batches] == ['deploys_to', 'uses']
    assert len([e for e in derivation if e.get('phase') == 'ordinary_replay']) == 8
    for i in range(8):
        left, right = calls[2*i:2*i+2]
        assert left[0] == right[0] == f'unit-{i:02d}'
        assert left[1] == 'original' and right[1] == 'alternatives' and left[2] == right[2]
    assert not (root / 'run').exists()
    return dict(schema='luna-staged-root-rehearsal-v1', verified=True, receipt_sha256=receipt_sha,
        candidate_files=proof['candidate_files'], synthetic_judgments=16,
        synthetic_token_sentinels=112, retained_ordinary_replays=8,
        exact_bound_batch_pairs=8, original_prose_predicate_preserved=True,
        model_calls=0, files_written=0, semantic_model_accuracy_measured=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--receipt-sha256', required=True)
    parser.add_argument('--entry-sha256', required=True)
    args = parser.parse_args()

    def deny(*a, **k):
        raise RuntimeError('rehearsal_external_action_forbidden')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
    subprocess.Popen = deny

    def audit(event, args):
        if event == 'open':
            mode, flags = args[1:3]
            if (type(mode) is str and any(c in mode for c in 'wax+') or
                type(flags) is int and flags & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC)):
                deny()
        if event in {'os.remove', 'os.rename', 'os.rmdir', 'os.mkdir', 'os.symlink', 'os.link',
            'os.chmod', 'os.chown', 'os.utime', 'os.truncate', 'os.system', 'os.fork', 'os.exec',
            'os.posix_spawn', 'subprocess.Popen', 'socket.connect', 'socket.getaddrinfo'}:
            deny()
    sys.addaudithook(audit)
    try:
        print(json.dumps(verify(Path(args.root), args.receipt_sha256, args.entry_sha256), sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps(dict(verified=False, model_calls=0, reason='root_offline_rehearsal_failed')))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
