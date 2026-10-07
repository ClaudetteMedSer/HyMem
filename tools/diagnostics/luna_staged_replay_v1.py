"""Read-only, source-bound replay of the private staged diagnostic journal.

Only finite counters and hashes leave this module. Raw model responses and
provider exceptions are read locally and never included in its proof.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import subprocess
import sys

sys.dont_write_bytecode = True
HEX = re.compile(r"[0-9a-f]{64}\Z")
JOURNAL = re.compile(r"(\d{4})-(derivation|unit-\d{2})\.json\Z")
SCHEMA = "luna-staged-probe-replay-v1"


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _regular(path: Path, cap: int) -> bytes:
    state = path.lstat()
    _require(stat.S_ISREG(state.st_mode) and 0 <= state.st_size <= cap,
             "private_record_invalid")
    return path.read_bytes()


def _json(raw: bytes):
    return json.loads(raw)


def _same(actual, expected, reason: str) -> None:
    _require(json.dumps(actual, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False) ==
             json.dumps(expected, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False), reason)


def _snapshot(snapshot, prior, current, *, terminal=False):
    """Verify the entire settled question prefix and global sum."""
    _require(type(snapshot) is dict and set(snapshot) == {
        "turns", "known_tokens", "usage_complete", "in_flight", "reserved",
        "stopped", "stop_code", "first_failure", "timings", "known_tokens_scope",
        "token_cap_kind", "questions"} and
        snapshot["known_tokens_scope"] ==
            "completed_turns_only_failed_turn_usage_unknown" and
        snapshot["token_cap_kind"] == "stop_before_next_observed_usage" and
        type(snapshot["timings"]) is dict and
        set(snapshot["timings"]) == {"preflight_seconds", "model_seconds",
                                     "cleanup_seconds"} and
        all(type(v) in (int, float) and math.isfinite(v) and v >= 0
            for v in snapshot["timings"].values()) and
        (snapshot["first_failure"] is None or type(snapshot["first_failure"]) is dict),
        "budget_shape_invalid")
    questions = snapshot["questions"]
    _require(type(questions) is dict and set(questions) ==
             {f"unit-{i:02d}" for i in range(len(prior) + (current is not None))},
             "question_prefix_invalid")
    total_turns = total_tokens = 0
    for i, item in enumerate(prior):
        q = questions[f"unit-{i:02d}"]
        _same(q, {"turns": item["admitted_turns"],
                  "known_tokens": item["known_tokens"], "in_flight": 0,
                  "usage_complete": True, "stopped": False},
              "prior_question_invalid")
        total_turns += q["turns"]
        total_tokens += q["known_tokens"]
    if current is not None:
        q = questions[f"unit-{len(prior):02d}"]
        _require(type(q) is dict and set(q) == {"turns", "known_tokens",
            "in_flight", "usage_complete", "stopped"} and
            type(q["turns"]) is int and 0 <= q["turns"] <= 3 and
            type(q["known_tokens"]) is int and q["known_tokens"] >= 0 and
            type(q["in_flight"]) is int and q["in_flight"] in (0, 1) and
            type(q["usage_complete"]) is bool and type(q["stopped"]) is bool,
            "current_question_invalid")
        total_turns += q["turns"]
        total_tokens += q["known_tokens"]
    _require(type(snapshot["turns"]) is int and snapshot["turns"] == total_turns and
             type(snapshot["known_tokens"]) is int and
             snapshot["known_tokens"] == total_tokens and
             type(snapshot["in_flight"]) is int and
             snapshot["in_flight"] == (q["in_flight"] if current is not None else 0) and
             type(snapshot["reserved"]) is int and
             0 <= snapshot["reserved"] <= snapshot["in_flight"] and
             type(snapshot["usage_complete"]) is bool and
             snapshot["usage_complete"] == (snapshot["in_flight"] == 0 and
                all(x["usage_complete"] for x in questions.values())) and
             type(snapshot["stopped"]) is bool and
             snapshot["turns"] <= 29 and snapshot["known_tokens"] >= 0,
             "budget_reconciliation_invalid")
    _require((snapshot["stop_code"] is None and not snapshot["stopped"]) or
             (type(snapshot["stop_code"]) is str and snapshot["stopped"]),
             "budget_stop_invalid")
    if terminal:
        _require(snapshot["reserved"] == snapshot["in_flight"] == 0 and
                 snapshot["usage_complete"] and not snapshot["stopped"] and
                 snapshot["first_failure"] is None,
                 "completed_budget_invalid")
        _require(q == {"turns": current["admitted_turns"],
                       "known_tokens": current["known_tokens"],
                       "in_flight": 0, "usage_complete": True, "stopped": False},
                 "completed_question_invalid")
    return q if current is not None else None


def _read_journal(directory: Path, expected_derivation, core):
    state = directory.lstat()
    _require(stat.S_ISDIR(state.st_mode), "journal_directory_invalid")
    paths = sorted(directory.iterdir())
    _require(1 <= len(paths) <= 128, "journal_count_invalid")
    derivation, grouped = [], {}
    seen_unit = False
    last_key = None
    for sequence, path in enumerate(paths, 1):
        match = JOURNAL.fullmatch(path.name)
        _require(match is not None and int(match[1]) == sequence,
                 "journal_sequence_invalid")
        event = _json(_regular(path, 2_000_000))
        _require(type(event) is dict, "journal_event_invalid")
        key = match[2]
        if key == "derivation":
            _require(not seen_unit, "derivation_order_invalid")
            derivation.append(event)
        else:
            seen_unit = True
            _require(key == last_key or key == f"unit-{len(grouped):02d}",
                     "journal_chronology_invalid")
            grouped.setdefault(key, []).append(event)
            last_key = key
    _same(derivation, _json(core._canonical(expected_derivation)),
          "derivation_mismatch")
    _require(set(grouped) == {f"unit-{i:02d}" for i in range(len(grouped))} and
             len(grouped) <= len(core.schedule()), "journal_schedule_invalid")
    return grouped


def _expected_before(core, request, batch, stage, recheck):
    if stage == "original":
        core.staged.validate_original_request(request, batch)
        schema = core.staged.build_original_output_schema(batch)
        batch_hash, batch_json, prior = batch.batch_sha256, batch.canonical_json, None
    else:
        core.staged.validate_alternatives_request(request, batch)
        schema = core.staged.build_alternatives_output_schema(batch)
        batch_hash = batch.classification_batch.batch_sha256
        batch_json = batch.classification_batch.canonical_json
        prior = batch.original_response_sha256
    return {"phase": "before_dispatch", "stage": stage, "recheck": recheck,
        "request": asdict(request), "batch": batch_json,
        "batch_sha256": batch_hash, "prior_response_sha256": prior,
        "output_schema_sha256": core._sha(core._canonical(schema))}


class _Incomplete(BaseException):
    pass


def _evaluate(core, unit, triples, sources, events, *, complete):
    """The pure evaluator drives the exact requests; records supply only responses."""
    cursor = 0
    responses = stages = 0

    def invoke(request, batch, stage, recheck):
        nonlocal cursor, responses, stages
        _require(cursor < len(events), "missing_before_dispatch")
        expected = _expected_before(core, request, batch, stage, recheck)
        _same(events[cursor], _json(core._canonical(expected)),
              "stage_request_mismatch")
        cursor += 1
        stages += 1
        if cursor == len(events):
            raise _Incomplete()
        event = events[cursor]
        _require(type(event) is dict and set(event) ==
                 {"phase", "stage", "recheck", "response"} and
                 event["phase"] == "response_returned" and
                 event["stage"] == stage and event["recheck"] is recheck and
                 type(event["response"]) is str, "stage_response_invalid")
        cursor += 1
        responses += 1
        return event["response"]

    try:
        evaluated = core.evaluate_unit(unit, triples, sources, invoke)
    except _Incomplete:
        _require(not complete and cursor == len(events), "partial_stage_invalid")
        return None, stages, responses
    _require(cursor == len(events), "stage_sequence_invalid")
    return evaluated, stages, responses


def replay(root: Path, receipt_sha: str, entry_sha: str) -> dict:
    root = Path(root)
    _require(type(entry_sha) is str and HEX.fullmatch(entry_sha) is not None,
             "entry_pin_invalid")
    entry_path = root / "code/tools/diagnostics/luna_staged_run_v1.py"
    _require(hashlib.sha256(_regular(entry_path, 100_000)).hexdigest() == entry_sha,
             "entry_pin_invalid")
    spec = importlib.util.spec_from_file_location("pinned_staged_replay_entry", entry_path)
    _require(spec is not None and spec.loader is not None, "entry_load_invalid")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    receipt, loaded, _, retained = entry.preflight(root, receipt_sha, True)
    host, core, old_core, cases, canary, chunk, semantic, transport, warm, concurrent = loaded
    raw_result = _regular(root / "run/private-result.json", 300_000)
    result = _json(raw_result)
    core.validate_public_result(result)
    expected_derivation = []
    batches = core.derive_canary_batches(canary_module=canary, chunk_module=chunk,
        semantic_probe_module=old_core, candidate=root / "candidate", retained=retained,
        record=expected_derivation.append)
    grouped = _read_journal(root / "run/private-journal", expected_derivation, core)
    units = result["units"]
    _require(len(grouped) in (len(units), len(units) + 1) and
             (len(grouped) == result["attempted_units"] or
              (len(grouped) == result["attempted_units"] + 1 and
               result["attempted_units"] == len(units))),
             "journal_result_prefix_invalid")
    _require(not result["diagnostic_completed"] or
             len(units) == len(core.schedule()) == len(grouped),
             "incomplete_campaign")
    fixture = cases.cases()
    malformed = returned = stages = 0
    for ordinal, item in enumerate(units):
        unit = core.schedule()[ordinal]
        events = grouped[f"unit-{ordinal:02d}"]
        _require(len(events) >= 4 and events[-2].get("phase") == "evaluated" and
                 events[-1].get("phase") == "unit_finished", "unit_phase_invalid")
        triples, sources, expected_hash = core._unit_input(unit, fixture, batches)
        evaluated, nstage, nreturned = _evaluate(core, unit, triples, sources,
                                                 events[:-2], complete=True)
        _same(events[-2], {"phase": "evaluated", "evaluation": evaluated},
              "evaluation_mismatch")
        expected_item = {"ordinal": ordinal, "kind": unit.kind, "index": unit.index,
            "label": ("table_original" if unit.kind == "table_canary" else
                      "prose_original" if unit.kind == "prose_canary" else
                      fixture[unit.index].category),
            "original_predicates": [t.predicate for t in triples],
            "initial_batch_sha256": evaluated["stages"][0]["batch_sha256"],
            **evaluated, "expected_gold_match": core._gold(unit, fixture, evaluated)}
        for key, value in expected_item.items():
            _same(item.get(key), value, "public_evaluation_mismatch")
        _require(nstage == nreturned == item["admitted_turns"] and
                 evaluated["stages"][0]["batch_sha256"] ==
                 (expected_hash or evaluated["stages"][0]["batch_sha256"]),
                 "completed_stage_accounting_invalid")
        finished = events[-1]
        _require(type(finished) is dict and set(finished) ==
                 {"phase", "stop_code", "budget", "result"} and
                 finished["phase"] == "unit_finished", "finish_shape_invalid")
        _same(finished["result"], item, "finished_result_mismatch")
        expected_stop = (result["stop_code"] if ordinal == len(units) - 1 and
                         len(grouped) == len(units) and
                         not result["diagnostic_completed"] else None)
        _same(finished["stop_code"], expected_stop, "finish_stop_mismatch")
        _snapshot(finished["budget"], units[:ordinal], item, terminal=True)
        _require(finished["budget"]["questions"][f"unit-{ordinal:02d}"]["turns"] ==
                 nreturned, "completed_ledger_invalid")
        malformed += evaluated["outcome"] == "malformed"
        returned += nreturned
        stages += nstage
    if len(grouped) > len(units):
        ordinal = len(units)
        unit = core.schedule()[ordinal]
        events = grouped[f"unit-{ordinal:02d}"]
        _require(events and events[-1].get("phase") == "unit_finished" and
                 type(events[-1]) is dict and set(events[-1]) ==
                 {"phase", "stop_code", "budget", "result"} and
                 events[-1]["result"] is None and
                 events[-1]["stop_code"] == result["stop_code"],
                 "partial_finish_invalid")
        body = events[:-1]
        exception = None
        if body and body[-1].get("phase") == "private_exception":
            exception = body.pop()
            _require(type(exception) is dict and set(exception) ==
                     {"phase", "type", "detail"} and
                     type(exception["type"]) is str and
                     type(exception["detail"]) is str,
                     "private_exception_shape_invalid")
        recorded_evaluation = None
        if body and body[-1].get("phase") == "evaluated":
            recorded_evaluation = body.pop()
            _require(type(recorded_evaluation) is dict and
                     set(recorded_evaluation) == {"phase", "evaluation"},
                     "partial_evaluation_shape_invalid")
        triples, sources, expected_hash = core._unit_input(unit, fixture, batches)
        if body:
            evaluated, nstage, nreturned = _evaluate(core, unit, triples, sources,
                                                     body, complete=False)
            _require(1 <= nstage <= 3 and
                     (recorded_evaluation is None or evaluated is not None),
                     "partial_stage_invalid")
            if recorded_evaluation is not None:
                _same(recorded_evaluation,
                      {"phase": "evaluated", "evaluation": evaluated},
                      "partial_evaluation_mismatch")
            returned += nreturned
            stages += nstage
        else:
            nstage = nreturned = 0
            _require(recorded_evaluation is None, "partial_evaluation_without_stage")
        q = _snapshot(events[-1]["budget"], units, None if
                      result["attempted_units"] == len(units) else {}, terminal=False)
        if result["attempted_units"] == len(units):
            _require(not body and q is None, "unattempted_stage_invalid")
        else:
            _require(nreturned <= q["turns"] <= nstage and
                     q["in_flight"] <= 1 and
                     (q["usage_complete"] or result["stop_code"] is not None),
                     "partial_question_invalid")
        _require(result["stop_code"] is not None and
                 (exception is None or result["stop_code"] in
                  {"infrastructure_or_runtime_failure", "cleanup_failure",
                   "private_evidence_write_failure"}), "partial_stop_invalid")
    elif not result["diagnostic_completed"]:
        _require(units and result["stop_code"] is not None,
                 "unexplained_incomplete_campaign")
    if grouped:
        final_budget = grouped[f"unit-{len(grouped)-1:02d}"][-1]["budget"]
        for key, value in result["paid_budget"].items():
            _same(final_budget.get(key), value, "final_budget_mismatch")
        _same(warm.serialize_failure(final_budget["first_failure"]),
              result["first_failure"], "failure_metadata_mismatch")
    _require(malformed == result["malformed_units"] and
             returned <= core.MAX_NEW_TURNS and stages <= 3 * len(grouped),
             "campaign_counts_invalid")
    return {"schema": SCHEMA, "verified": True,
        "complete": result["diagnostic_completed"],
        "receipt_sha256": receipt_sha,
        "private_result_sha256": hashlib.sha256(raw_result).hexdigest(),
        "replayed_units": len(units), "returned_responses": returned,
        "replayed_stages": stages, "malformed_units": malformed,
        "new_model_calls": 0, "semantic_accuracy_accepted": False,
        "full_lme_ready": False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--receipt-sha256", required=True)
    parser.add_argument("--entry-sha256", required=True)
    args = parser.parse_args(argv)

    def deny(*_args, **_kwargs):
        raise RuntimeError("replay_external_action_forbidden")

    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    subprocess.Popen = deny

    def audit(event, values):
        if event == "open":
            mode, flags = values[1:3]
            if ((type(mode) is str and any(c in mode for c in "wax+")) or
                    (type(flags) is int and flags & (os.O_WRONLY | os.O_RDWR |
                     os.O_APPEND | os.O_CREAT | os.O_TRUNC))):
                deny()
        if event in {"os.remove", "os.rename", "os.rmdir", "os.mkdir", "os.symlink",
                     "os.link", "os.chmod", "os.chown", "os.utime", "os.truncate",
                     "os.system", "os.fork", "os.exec", "os.posix_spawn",
                     "subprocess.Popen", "socket.connect", "socket.getaddrinfo"}:
            deny()

    sys.addaudithook(audit)
    try:
        proof = replay(args.root, args.receipt_sha256, args.entry_sha256)
        print(json.dumps(proof, sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps({"schema": SCHEMA, "verified": False, "complete": False,
                          "new_model_calls": 0, "reason": "offline_replay_failed"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
