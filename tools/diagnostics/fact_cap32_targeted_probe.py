"""Read-only, single-unit cap-32 probe for a guarded stopped fact replay.

The default preview performs no provider request. Probe transmits exactly the
authoritative source of the first stale cap-16 fact outcome to the configured
provider, at most once (the transport may retry up to three HTTP attempts).
It never publishes facts or moves cursors. Output contains digests and counts,
never source or provider text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from hymem import bootstrap
from hymem.core import db as core_db
from hymem.dreaming import facts as fact_mod

if __package__:
    from . import fact_only_replay as replay
else:
    import fact_only_replay as replay


def _close_client_safely(client) -> None:
    """Never let cleanup print a provider exception or reply fragment."""
    try:
        client.close()
    except Exception:
        pass


def _first_stale(conn, target16: str):
    row = conn.execute(
        "SELECT * FROM fact_extraction_outcomes WHERE prompt_version<>? "
        "ORDER BY session_id," + fact_mod._fact_replay_order_sql() + " LIMIT 1",
        (target16,),
    ).fetchone()
    if row is None:
        raise replay.RepairStopped("no stale cap-16 fact outcome remains")
    if fact_mod.next_fact_outcome_for_replay(
        conn, row["session_id"], target16,
    ) != row["slice_key"]:
        raise replay.RepairStopped("first stale fact outcome changed")
    if not fact_mod.fact_session_authority_is_valid(conn, row["session_id"]):
        raise replay.RepairStopped("fact session authority is invalid")
    occurrences = fact_mod.load_fact_outcome_source_manifest(
        conn, row["slice_key"], verify_result=True,
        _require_committed_chain=False,
    )
    if occurrences is None or len(occurrences) != row["source_manifest_count"]:
        raise replay.RepairStopped("fact source manifest is invalid")
    return row, occurrences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--server-process-env", action="store_true")
    parser.add_argument("--expect-state-sha256")
    parser.add_argument("--expect-slice-key-sha256")
    parser.add_argument("--expect-input-hash")
    parser.add_argument("--expect-manifest-hash")
    args = parser.parse_args()
    if args.probe and not args.allow_provider_calls:
        parser.error("probe requires --allow-provider-calls")
    if args.allow_provider_calls and not args.probe:
        parser.error("--allow-provider-calls requires --probe")
    db_path = args.db.resolve(strict=True)
    live = replay._is_live_store(db_path)
    if live and db_path != replay.LIVE_CONTAINER_DB:
        parser.error("live probe must run inside the Hermes1 container")
    if live and not args.server_process_env:
        parser.error("live probe requires --server-process-env")
    if args.probe and any(value is None for value in (
        args.expect_state_sha256, args.expect_slice_key_sha256,
        args.expect_input_hash, args.expect_manifest_hash,
    )):
        parser.error("probe requires exact state, slice, input, and manifest guards")

    # Third-party exception strings and standard fact warnings can contain
    # untrusted provider or session values. Only bounded categories are output.
    logging.disable(logging.CRITICAL)
    llm = None
    try:
        if args.server_process_env:
            replay._load_server_process_env()
        if args.probe and bootstrap._validated_facts_cap_override(
            bootstrap.resolve_env().dream_max_facts_per_session_raw
        ) != 16:
            raise replay.RepairStopped("running Hermes cap is not 16")
        llm = replay._client_from_env()
        cfg16 = replay._configuration(db_path, 16)
        cfg32 = replay._configuration(db_path, 32)
        target16 = fact_mod.facts_config_version(cfg16, client=llm)
        target32 = fact_mod.facts_config_version(cfg32, client=llm)
        with core_db.read_snapshot(db_path) as conn:
            state = replay._state(conn, target16)
            if state["schema"] != 64:
                raise replay.RepairStopped("unexpected HyMem schema")
            if conn.execute(
                "SELECT 1 FROM run_lock WHERE name='dreaming'"
            ).fetchone() is not None:
                raise replay.RepairStopped("dream lock is held")
            row, occurrences = _first_stale(conn, target16)
            key_digest = hashlib.sha256(row["slice_key"].encode()).hexdigest()
            public = {
                "state_sha256": state["state_sha256"],
                "stale_cap16": state["stale"],
                "slice_key_sha256": key_digest,
                "input_hash": row["input_hash"],
                "source_manifest_hash": row["source_manifest_hash"],
                "source_manifest_count": len(occurrences),
                "old_active_facts": conn.execute(
                    "SELECT COUNT(*) FROM narrative_facts WHERE "
                    "source_outcome_key=? AND lifecycle_status='active' "
                    "AND invalid_at IS NULL", (row["slice_key"],),
                ).fetchone()[0],
                "target32": target32,
            }
            if not args.probe:
                print(json.dumps(public, sort_keys=True))
                return 0
            if (
                args.expect_state_sha256 != public["state_sha256"]
                or args.expect_slice_key_sha256 != key_digest
                or args.expect_input_hash != row["input_hash"]
                or args.expect_manifest_hash != row["source_manifest_hash"]
            ):
                raise replay.RepairStopped("guarded fact source changed")
            before_http = llm.request_attempts
            # This calls the provider once for the exact historical unit. The
            # client has a fixed maximum of three transport attempts.
            result = fact_mod.reextract_fact_outcome(
                conn, row["slice_key"], llm, cfg32,
                _require_committed_chain=False,
            )
            attempts = llm.request_attempts - before_http
            if attempts < 1 or attempts > 3:
                raise replay.RepairStopped("provider exceeded probe HTTP ceiling")
            public.update({
                "cap": 32,
                "accepted": not result.parse_failed,
                "reason": result.failure_reason if result.parse_failed else None,
                "items": len(result.items) if not result.parse_failed else None,
                "http_attempts": attempts,
                "prompt_tokens": llm.prompt_tokens if llm.token_usage_available else None,
                "completion_tokens": (
                    llm.completion_tokens if llm.token_usage_available else None
                ),
            })
            print(json.dumps(public, sort_keys=True))
            return 0
    except Exception as exc:
        print(json.dumps({
            "stopped": type(exc).__name__,
            "reason": str(exc) if isinstance(exc, replay.RepairStopped)
            else "unexpected failure; no source or provider text emitted",
        }, sort_keys=True))
        return 2
    finally:
        if llm is not None:
            _close_client_safely(llm)


if __name__ == "__main__":
    raise SystemExit(main())
