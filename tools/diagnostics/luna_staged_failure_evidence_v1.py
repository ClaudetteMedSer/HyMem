"""Finite, read-only evidence check for the two failed staged probe units.

Run on the private host. Source, request, response, quote and identifier text
never appears in the result, including on failure.
"""
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
SCHEMA = "luna-staged-failure-evidence-v1"


def _regular(path: Path, cap: int) -> bytes:
    state = path.lstat()
    if not stat.S_ISREG(state.st_mode) or state.st_size > cap:
        raise ValueError("private_file_invalid")
    return path.read_bytes()


def _entry(root: Path, entry_sha: str):
    path = root / "code/tools/diagnostics/luna_staged_run_v1.py"
    if hashlib.sha256(_regular(path, 100_000)).hexdigest() != entry_sha:
        raise ValueError("entry_pin_invalid")
    spec = importlib.util.spec_from_file_location("pinned_failure_evidence_entry", path)
    if spec is None or spec.loader is None:
        raise ValueError("entry_invalid")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    return entry


def _journal(root, core, replay, canary, chunk, old_core, retained):
    derivation = []
    batches = core.derive_canary_batches(canary_module=canary, chunk_module=chunk,
        semantic_probe_module=old_core, candidate=root / "candidate", retained=retained,
        record=derivation.append)
    grouped = replay._read_journal(root / "run/private-journal", derivation, core)
    return batches, grouped


def _first_stage(core, replay, fixture, batches, grouped, ordinal):
    unit = core.schedule()[ordinal]
    triples, sources, expected_hash = core._unit_input(unit, fixture, batches)
    request, batch = core.staged.build_original_request(triples, sources)
    expected = replay._expected_before(core, request, batch, "original", False)
    events = grouped[f"unit-{ordinal:02d}"]
    replay._same(events[0], json.loads(core._canonical(expected)), "first_stage_unbound")
    response = events[1]
    if (type(response) is not dict or set(response) !=
            {"phase", "stage", "recheck", "response"} or
            response["phase"] != "response_returned" or
            response["stage"] != "original" or response["recheck"] is not False or
            type(response["response"]) is not str or
            (expected_hash is not None and batch.batch_sha256 != expected_hash)):
        raise ValueError("first_response_unbound")
    return unit, batch, response["response"], events


def _support(evidence):
    indices = list(range(len(evidence)))
    check = {"state": "supported", "evidence_indices": indices}
    return {"state": "supported", "support": {"evidence": evidence,
        "checks": {"attribution_and_roles": check,
                   "relation_and_polarity": check}}}


def _prose(core, canary, batch, original_raw, events):
    staged = core.staged
    original = staged.parse_original_response(original_raw, batch)
    source, triple = batch.sources[0], batch.triples[0]
    left, right = canary._PROSE_BOUNDARY_LEFT, canary._PROSE_BOUNDARY_RIGHT
    matching = [c for c in source.contexts if left in c.content]
    context = matching[0] if len(matching) == 1 else None
    left_cue = context is not None
    right_owned = right in source.content
    prefix_covers_owned = bool(context and right_owned and
                               right in source.content[:context.owned_prefix_chars])
    parent_applicable = bool(context and context.applies_to_region is not None)
    parent_quote_available = False
    evidence = []
    if left_cue and right_owned:
        evidence = [
            {"source_message_id": triple.source_message_id,
             "region": "owned", "quote": right},
            {"source_message_id": triple.source_message_id,
             "region": context.region, "quote": left},
        ]
        if parent_applicable:
            parents = [c for c in source.contexts if c.region == context.applies_to_region]
            if len(parents) == 1:
                quote = parents[0].content[:context.applies_to_prefix_chars].strip()[:192]
                parent_quote_available = bool(quote)
                if quote:
                    evidence.append({"source_message_id": triple.source_message_id,
                        "region": parents[0].region, "quote": quote})
    actual_negative = original.states == ("not_established",)
    actual_alternatives_all_negative = False
    if len(events) >= 4 and events[2].get("phase") == "before_dispatch" and \
            events[3].get("phase") == "response_returned":
        alt = staged._load(events[3]["response"])
        rows = alt.get("alternatives")
        actual_alternatives_all_negative = (type(rows) is list and len(rows) == 1 and
            type(rows[0]) is dict and type(rows[0].get("alternatives")) is dict and
            set(rows[0]["alternatives"]) == set(core.v4.PREDICATE_ORDER) - {triple.predicate} and
            all(type(v) is dict and v.get("state") == "not_established"
                for v in rows[0]["alternatives"].values()))
    candidate_valid = candidate_accepted = False
    if (actual_negative and actual_alternatives_all_negative and left_cue and
            right_owned and prefix_covers_owned and
            (not parent_applicable or parent_quote_available)):
        try:
            _, alt_batch = staged.build_alternatives_request(batch, original_raw)
            payload = staged._load(events[3]["response"])
            payload["alternatives"][0]["alternatives"]["prefers"] = _support(evidence)
            alternate_raw = core._canonical(payload).decode("utf-8")
            review = staged.parse_staged_responses(batch, original_raw, alternate_raw)
            candidate_valid = (len(review.verdicts) == 1 and
                               review.verdicts[0].status == "replace_predicate" and
                               review.verdicts[0].predicate == "prefers")
            def invoke(request, current, stage, recheck):
                if stage == "alternatives":
                    staged.validate_alternatives_request(request, current)
                    if current.original_response_sha256 != alt_batch.original_response_sha256:
                        raise ValueError("candidate_prior_invalid")
                    return alternate_raw
                staged.validate_original_request(request, current)
                if not recheck:
                    if current.batch_sha256 != batch.batch_sha256:
                        raise ValueError("candidate_initial_invalid")
                    return original_raw
                response = {"schema": staged.ORIGINAL_SCHEMA,
                    "batch_sha256": current.batch_sha256, "complete": True,
                    "originals": [{"index": 0, "original": _support(evidence)}]}
                raw = core._canonical(response).decode("utf-8")
                staged.parse_original_response(raw, current)
                return raw
            evaluated = core.evaluate_unit(core.schedule()[7], batch.triples,
                                           batch.sources, invoke)
            candidate_accepted = (candidate_valid and
                evaluated["outcome"] == "accepted" and
                evaluated["final_predicates"] == ["prefers"] and
                core._gold(core.schedule()[7], (), evaluated) is True)
        except (ValueError, KeyError, IndexError, TypeError,
                staged.GroundingContractError, core.DiagnosticStop):
            candidate_valid = candidate_accepted = False
    return {"case_index": 7, "original_negative": actual_negative,
        "alternatives_all_negative": actual_alternatives_all_negative,
        "left_cue_in_bound_context": left_cue, "right_cue_in_owned": right_owned,
        "owned_quote_within_context_prefix": prefix_covers_owned,
        "parent_applicability_present": parent_applicable,
        "parent_quote_available": parent_quote_available,
        "mechanical_candidate_valid": candidate_valid,
        "mechanical_candidate_accepted": candidate_accepted}


def _role(core, batch, original_raw, events):
    original = core.staged.parse_original_response(original_raw, batch)
    payload = core.staged._load(original_raw)
    rows = payload["originals"]
    supported = []
    for index, row in enumerate(rows):
        assessment = row["original"]
        if assessment["state"] != "supported":
            continue
        triple = batch.triples[index]
        evidence = assessment["support"]["evidence"]
        joined = " ".join(item["quote"] for item in evidence).casefold()
        supported.append({"index": index, "evidence": [
            {"region": item["region"], "length": len(item["quote"])}
            for item in evidence],
            "all_evidence_maps_to_claim_source": all(
                item["source_message_id"] == triple.source_message_id for item in evidence),
            "subject_literal_in_quotes": triple.subject.casefold() in joined,
            "object_literal_in_quotes": triple.object.casefold() in joined})
    return {"case_index": 5, "claim_count": len(batch.triples),
        "original_states": list(original.states),
        "original_supported": supported,
        "alternatives_returned": len(events) >= 4 and
            events[2].get("phase") == "before_dispatch" and
            events[3].get("phase") == "response_returned"}


def verify(root: Path, receipt_sha: str, entry_sha: str, result_sha: str) -> dict:
    root = Path(root)
    entry = _entry(root, entry_sha)
    receipt, loaded, _, retained = entry.preflight(root, receipt_sha, postrun=True)
    host, core, old_core, cases, canary, chunk, _, transport, warm, concurrent = loaded
    raw = _regular(root / "run/private-result.json", 300_000)
    if hashlib.sha256(raw).hexdigest() != result_sha:
        raise ValueError("result_pin_invalid")
    result = json.loads(raw)
    core.validate_public_result(result)
    replay_path = root / "code/tools/diagnostics/luna_staged_replay_v1.py"
    if hashlib.sha256(_regular(replay_path, 100_000)).hexdigest() != \
            receipt["source_sha256"].get("tools/diagnostics/luna_staged_replay_v1.py"):
        raise ValueError("replay_pin_invalid")
    spec = importlib.util.spec_from_file_location("pinned_failure_evidence_replay", replay_path)
    if spec is None or spec.loader is None:
        raise ValueError("replay_invalid")
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)
    proof = replay.replay(root, receipt_sha, entry_sha)
    if not proof["verified"] or not proof["complete"] or \
            proof["private_result_sha256"] != result_sha:
        raise ValueError("replay_incomplete")
    batches, grouped = _journal(root, core, replay, canary, chunk,
                                old_core, retained)
    fixture = cases.cases()
    _, role_batch, role_raw, role_events = _first_stage(
        core, replay, fixture, batches, grouped, 5)
    _, prose_batch, prose_raw, prose_events = _first_stage(
        core, replay, fixture, batches, grouped, 7)
    if len(result["units"]) != 8 or result["units"][5]["index"] != 21 or \
            result["units"][7]["kind"] != "prose_canary":
        raise ValueError("schedule_invalid")
    return {"schema": SCHEMA, "verified": True, "new_model_calls": 0,
        "private_result_sha256": result_sha,
        "role_control": _role(core, role_batch, role_raw, role_events),
        "prose_canary": _prose(core, canary, prose_batch, prose_raw, prose_events)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("root", "receipt-sha256", "entry-sha256", "result-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    def deny(*_args, **_kwargs):
        raise RuntimeError("external_action_forbidden")
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = deny
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
        print(json.dumps(verify(Path(args.root), args.receipt_sha256,
                                args.entry_sha256, args.result_sha256), sort_keys=True))
        return 0
    except BaseException:
        print(json.dumps({"schema": SCHEMA, "verified": False,
                          "new_model_calls": 0, "reason": "finite_evidence_unverified"},
                         sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
