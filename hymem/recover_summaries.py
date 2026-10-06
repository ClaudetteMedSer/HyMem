"""Inspect or explicitly repair rolling summaries without reindexing items.

The default is a read-only health check with no provider calls. --apply spends
a bounded number of calls and keeps partial recovery private until complete.
This command requires an existing current-schema store; it never runs the full dream loop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from hymem.core import db


def _safe_recovery_report(value):
    counts = {'calls', 'provider_attempts', 'advanced', 'published', 'held', 'exhausted', 'remaining'}
    if (type(value) is not dict or set(value) != counts | {'provider_attempts_exact'}
            or any(type(value[key]) is not int or not 0 <= value[key] <= 2**63-1 for key in counts)
            or type(value['provider_attempts_exact']) is not bool):
        return None
    return dict(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='explicitly authorize bounded LLM calls and summary writes')
    parser.add_argument('--max-calls', type=int, default=1)
    parser.add_argument('--max-attempts', type=int, default=3, help='maximum rejected replies per exact recovery slice')
    parser.add_argument('--max-chars', type=int, default=8000)
    parser.add_argument('--max-tokens', type=int, default=3072)
    parser.add_argument('--timeout-seconds', type=float, default=120.0)
    parser.add_argument('--session-id')
    args = parser.parse_args(argv)
    if (not 1 <= args.max_calls <= 100 or not 1 <= args.max_attempts <= 100
            or not 1 <= args.max_chars <= 1000000 or not 1 <= args.max_tokens <= 32768
            or not math.isfinite(args.timeout_seconds) or not 0 < args.timeout_seconds <= 3600
            or (args.session_id is not None and (not args.session_id.strip()
                or len(args.session_id) > 1024 or '\x00' in args.session_id))):
        parser.error('invalid recovery bounds')
    result = {'mode': 'apply' if args.apply else 'inspect', 'status': 'unverified'}
    hy = None
    primary = None
    try:
        from hymem.bootstrap import resolve_env, build_from_env
        from hymem.dreaming.status import durable_summary_status
        cfg = resolve_env()
        # A missing/old database must not be silently created or migrated by
        # an inspection or a repair command.
        with db.read_snapshot(cfg.root / 'hymem.sqlite') as conn:
            if db.schema_version(conn) != db.EXPECTED_SCHEMA_VERSION:
                raise RuntimeError('current_schema_required')
            result['health_before'] = durable_summary_status(conn)
        if result['health_before']['malformed_summaries']:
            result['status'] = 'malformed'
        elif args.apply:
            hy = build_from_env()
            result['recovery'] = _safe_recovery_report(hy.recover_summaries(
                max_calls=args.max_calls, max_attempts=args.max_attempts,
                max_chars=args.max_chars, max_tokens=args.max_tokens,
                timeout_seconds=args.timeout_seconds, session_id=args.session_id,
            ))
            if result['recovery'] is None:
                raise RuntimeError('invalid_recovery_accounting')
            with db.read_snapshot(cfg.root / 'hymem.sqlite') as conn:
                result['health_after'] = durable_summary_status(conn)
            result['status'] = 'complete' if result['health_after']['summary_healthy'] else 'degraded'
        else:
            result['status'] = 'complete' if result['health_before']['summary_healthy'] else 'degraded'
    except BaseException as exc:
        # No exception text, source, session identifiers, endpoint or credentials
        # enter this operator receipt.
        primary = exc
        if not isinstance(exc, Exception):
            raise
        result['status'] = 'error'
        result['error_type'] = type(exc).__name__
        receipt = _safe_recovery_report(getattr(exc, 'summary_recovery_report', None))
        if receipt is not None:
            result['recovery'] = receipt
        elif args.apply:
            result['recovery_accounting'] = 'unavailable'
    finally:
        if hy is not None:
            # The bootstrap owner also closes its provider transports. Do not
            # close only HyMem's database handle and leak the HTTP client.
            from hymem.bootstrap import shutdown_instance
            try:
                shutdown_instance(hy)
            except BaseException as exc:
                if primary is not None and not isinstance(primary, Exception):
                    primary.add_note(f'summary recovery cleanup failed: {type(exc).__name__}')
                elif not isinstance(exc, Exception):
                    # An already-reported ordinary failure is not an active
                    # exception that may suppress cancellation during cleanup.
                    raise
                elif primary is not None:
                    primary.add_note(f'summary recovery cleanup failed: {type(exc).__name__}')
                else:
                    result['status'] = 'error'
                    result['error_type'] = type(exc).__name__
    print(json.dumps(result, sort_keys=True))
    return 0 if result['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
