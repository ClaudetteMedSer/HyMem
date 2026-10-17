"""Read-only, source-guarded capacity probe for one stopped six-unit fact session.

Preview makes no provider calls. Probe requires the exact preview seals and
explicit call authorization. At most six cap-64/6144 completions are made.
The first rejected unit stops the sequence. No model result is published.
Only hashes, lengths, counts, and closed failure categories are printed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import logging
from pathlib import Path

from hymem.core import db as core_db
from hymem.dreaming import facts
from hymem.dreaming.lossless import validate_message_coverage_artifact
from hymem.extraction.llm import LLMOutputTruncatedError, LLMResponseError

if __package__:
    from . import fact_only_replay as replay
else:
    import fact_only_replay as replay


MAX_UNITS = 6
MAX_LOGICAL = MAX_UNITS
MAX_HTTP = 3 * MAX_LOGICAL
REVIEWED_CAP32_TARGET = (
    "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=3072|items=32|"
    "semantic=sha256:545261b726153f79c65448281e0f426d8320f37c814f0fa9f5c39fa49600ab9c"
)
REVIEWED_CAP64_TARGET = (
    "facts-lossless-v1|prompt=facts.v4|chars=12000|tokens=6144|items=64|"
    "semantic=sha256:bfd4c192babad097273e34d9373a01c3a6e2b3818fc00b0f62506ff87008387f"
)


def _units(conn, target32: str) -> tuple[str, list, str, list[dict]]:
    """Identify the first unpublished session in the operator's exact order."""
    first = conn.execute(
        "SELECT session_id FROM fact_extraction_outcomes "
        "WHERE prompt_version<>? ORDER BY session_id LIMIT 1", (target32,),
    ).fetchone()
    if first is None:
        raise replay.RepairStopped("no stale fact session remains")
    session_id = first["session_id"]
    rows = conn.execute(
        "SELECT * FROM fact_extraction_outcomes WHERE session_id=? "
        "AND prompt_version<>? ORDER BY " + facts._fact_replay_order_sql(),
        (session_id, target32),
    ).fetchall()
    if len(rows) != MAX_UNITS or facts.next_fact_outcome_for_replay(
        conn, session_id, target32,
    ) != rows[0]["slice_key"]:
        raise replay.RepairStopped("first stale session is not the reviewed six units")
    if not facts.fact_session_authority_is_valid(conn, session_id):
        raise replay.RepairStopped("existing fact authority is invalid")

    sealed = []
    public = []
    for row in rows:
        occurrences = facts.load_fact_outcome_source_manifest(
            conn, row["slice_key"], verify_result=True,
            _require_committed_chain=False,
        )
        if occurrences is None or len(occurrences) != row["source_manifest_count"]:
            raise replay.RepairStopped("fact source manifest is invalid")
        proofs = [validate_message_coverage_artifact(
            conn, message_id=source.message_id,
            chunk_id=source.coverage_chunk_id,
            coverage_version=source.coverage_version,
        ) for source in occurrences]
        rendered = facts._render_fact_slice(
            proofs,
            before_partial_message_id=row["cursor_before_partial_message_id"],
            before_offset=int(row["cursor_before_offset"]),
            after_partial_message_id=row["cursor_after_partial_message_id"],
            after_offset=int(row["cursor_after_offset"]),
        )
        if facts.fact_input_hash(rendered) != row["input_hash"]:
            raise replay.RepairStopped("fact source differs from stored input hash")
        sealed.append([
            row["slice_key"], row["input_hash"], row["source_manifest_hash"],
            row["source_manifest_count"], row["generation"],
            row["cursor_before_message_id"],
            row["cursor_before_partial_message_id"], row["cursor_before_offset"],
            row["cursor_after_message_id"],
            row["cursor_after_partial_message_id"], row["cursor_after_offset"],
        ])
        public.append({
            "slice_key_sha256": hashlib.sha256(
                row["slice_key"].encode("utf-8")
            ).hexdigest(),
            "input_hash": row["input_hash"],
            "source_manifest_hash": row["source_manifest_hash"],
            "source_count": len(occurrences),
            "rendered_chars": len(rendered),
            "rendered_utf8_bytes": len(rendered.encode("utf-8")),
        })
    packed = json.dumps(sealed, ensure_ascii=True, separators=(",", ":")).encode()
    return (
        facts.fact_diagnostic_session_key(session_id), rows,
        hashlib.sha256(packed).hexdigest(), public,
    )


def _probe_units(conn, rows, llm, cfg32, *, target32: str) -> tuple[list[dict], str | None]:
    """Classify each exact unit at one larger capacity, stopping on rejection."""
    cfg64 = replace(
        cfg32, dream_max_facts_per_session=64, facts_extraction_max_tokens=6144,
    )
    request_token_cap = facts.build_facts_request("synthetic test", cfg64).max_tokens
    if request_token_cap != 6144:
        raise replay.RepairStopped("fact-specific request token cap is not 6144")
    results = []
    start_http = int(llm.request_attempts)
    logical = 0
    for index, row in enumerate(rows, 1):
        if facts.next_fact_outcome_for_replay(
            conn, row["session_id"], target32,
        ) != rows[0]["slice_key"]:
            return results, "stale fact order changed during probe"
        if logical >= MAX_LOGICAL or int(llm.request_attempts) - start_http + 3 > MAX_HTTP:
            return results, "provider call ceiling reached"
        logical += 1
        before_http = int(llm.request_attempts)
        before_prompt = int(getattr(llm, "prompt_tokens", 0))
        before_completion = int(getattr(llm, "completion_tokens", 0))
        try:
            extraction = facts.reextract_fact_outcome(
                conn, row["slice_key"], llm, cfg64,
                _require_committed_chain=False,
            )
        except LLMOutputTruncatedError:
            accepted, reason, count = False, "output_truncated", None
        except LLMResponseError:
            accepted, reason, count = False, "response_rejected", None
        except Exception:
            accepted, reason, count = False, "provider_or_source_error", None
        else:
            accepted = not extraction.parse_failed
            reason = extraction.failure_reason if extraction.parse_failed else None
            count = len(extraction.items) if accepted else None
        http = int(llm.request_attempts) - before_http
        usage_available = bool(getattr(llm, "token_usage_available", False))
        results.append({
            "unit": index, "cap": 64, "max_tokens": request_token_cap,
            "accepted": accepted, "reason": reason, "items": count,
            "http_attempts": http,
            "prompt_tokens": (
                int(llm.prompt_tokens) - before_prompt if usage_available else None
            ),
            "completion_tokens": (
                int(llm.completion_tokens) - before_completion
                if usage_available else None
            ),
        })
        if http < 1 or http > 3 or int(llm.request_attempts) - start_http > MAX_HTTP:
            return results, "provider HTTP attempt ceiling exceeded"
        if not accepted:
            return results, "cap 64 did not validate the complete unit"
    return results, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--server-process-env", action="store_true")
    parser.add_argument("--expect-state-sha256")
    parser.add_argument("--expect-target32")
    parser.add_argument("--expect-target64")
    parser.add_argument("--expect-session-key")
    parser.add_argument("--expect-units-sha256")
    parser.add_argument("--expect-stale", type=int)
    args = parser.parse_args()
    if args.probe != args.allow_provider_calls:
        parser.error("provider calls require both --probe and --allow-provider-calls")
    if args.probe and any(value is None for value in (
        args.expect_state_sha256, args.expect_target32, args.expect_target64,
        args.expect_session_key, args.expect_units_sha256, args.expect_stale,
    )):
        parser.error("probe requires exact state, producer, session, unit and count guards")
    db_path = args.db.resolve(strict=True)
    live = replay._is_live_store(db_path)
    if live and db_path != replay.LIVE_CONTAINER_DB:
        parser.error("live probe must run inside the Hermes1 container")
    if live and not args.server_process_env:
        parser.error("live probe requires --server-process-env")

    previous_logging_disable = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    llm = None
    try:
        if args.server_process_env:
            replay._load_server_process_env()
        cfg32 = replay._configuration(db_path, 32)
        llm = replay._client_from_env()
        target32 = REVIEWED_CAP32_TARGET
        cfg64 = replace(
            cfg32, dream_max_facts_per_session=64,
            facts_extraction_max_tokens=6144,
        )
        target64 = facts.facts_config_version(cfg64, client=llm)
        if target64 != REVIEWED_CAP64_TARGET:
            raise replay.RepairStopped("cap-64 producer generation differs from reviewed cutover")
        if facts.build_facts_request("synthetic test", cfg64).max_tokens != 6144:
            raise replay.RepairStopped("fact-specific request token cap is not 6144")
        with core_db.read_snapshot(db_path) as conn:
            state = replay._state(conn, target32)
            if state["schema"] != 64:
                raise replay.RepairStopped("unexpected HyMem schema")
            if conn.execute(
                "SELECT 1 FROM run_lock WHERE name='dreaming'"
            ).fetchone() is not None:
                raise replay.RepairStopped("dream lock is held")
            session_key, rows, units_sha256, units = _units(conn, target32)
            output = {
                "state_sha256": state["state_sha256"],
                "target32": target32,
                "target64": target64,
                "stale": state["stale"],
                "outcomes": state["outcomes"],
                "sessions": state["sessions"],
                "pending_fresh": state["pending_fresh"],
                "quarantined": state["quarantined"],
                "session_key": session_key,
                "units_sha256": units_sha256,
                "units": units,
            }
            if not args.probe:
                print(json.dumps(output, sort_keys=True))
                return 0
            if (args.expect_state_sha256 != state["state_sha256"]
                    or args.expect_target32 != target32
                    or args.expect_target64 != target64
                    or args.expect_session_key != session_key
                    or args.expect_units_sha256 != units_sha256
                    or args.expect_stale != state["stale"]):
                raise replay.RepairStopped("guarded fact state changed")
            if (state["stale"] != 77 or state["outcomes"] != 128
                    or state["sessions"] != 49
                    or state["pending_fresh"] != 1
                    or state["quarantined"] != 0):
                raise replay.RepairStopped("fact replay stop baseline changed")
            probes, stopped = _probe_units(
                conn, rows, llm, cfg32, target32=target32,
            )
            print(json.dumps({
                "state_sha256": state["state_sha256"],
                "units_sha256": units_sha256,
                "probes": probes,
                "logical_calls": len(probes),
                "http_attempts": sum(item["http_attempts"] for item in probes),
                "stopped": stopped,
            }, sort_keys=True))
            return 2 if stopped else 0
    except Exception as exc:
        print(json.dumps({
            "stopped": type(exc).__name__,
            "reason": str(exc) if isinstance(exc, replay.RepairStopped)
            else "unexpected failure; no source or provider text emitted",
        }, sort_keys=True))
        return 2
    finally:
        logging.disable(previous_logging_disable)
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
