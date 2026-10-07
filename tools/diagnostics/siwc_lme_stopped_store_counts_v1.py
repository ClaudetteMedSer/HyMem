"""One-use, counts-only inspection of the stopped o83ipyar diagnostic.

The remote program executes only the pinned metadata reader, then reads four
fixed private summaries and four immutable SQLite stores.  Failure output is
constant; no private exception, row, detail, or host stderr is returned.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import subprocess
import time


SCHEMA = "siwc-lme-stopped-store-counts-v1"
ROOT = "/home/atta/.hymem-siwc-lme-diagnostic-preflight-o83ipyar"
RECEIPT_SHA = "16443899d6a5dccd99c77c01b2c8fe58623a89fb4403cf06dd4b652ad09a7391"
READER = Path(__file__).with_name("siwc_lme_diagnostic_progress_v9.py")
READER_SHA = "a2c3fca37352c1604df661cbbf477109707866874d41d7d1edee7960e2f10fe6"
INDEXING_SCHEMA = "hymem-lme-indexing-summary-v6"
REASONS = (
    "branch_incomplete", "call_failure", "contract_failure", "grounding_failure",
    "incomplete_response", "input_contract_failure", "internal_validation_failure",
    "item_validation_failure", "output_limit_exceeded", "parse_failure",
    "resource_limit", "response_conflict", "shape_failure",
    "source_coverage_failure", "unspecified_failure", "other",
)
PENDING = (
    "pending_source_materialization", "pending_chunks", "pending_digests",
    "pending_profiles", "pending_facts", "pending_aggregation",
    "pending_chunk_embeddings", "pending_message_embeddings", "pending_edge_embeddings",
    "pending_episode_embeddings", "pending_fact_embeddings",
)
MALFORMED = (
    "malformed_source_materialization", "malformed_digests", "malformed_profiles",
    "malformed_facts", "malformed_summaries",
)
QUARANTINED = (
    "quarantined_chunks", "quarantined_digests", "quarantined_profiles",
    "quarantined_facts", "quarantined_facts_malformed",
)
REPORT_COUNTS = (
    "sessions_processed", "chunks_seen", "chunks_processed",
    "chunk_extraction_failures", "chunk_extraction_completion_calls",
    "chunk_extraction_provider_attempts", "coverage_integrity_failures",
    "triples_extracted", "markers_extracted", "rules_extracted",
    "chunks_embedded", "chunks_embedded_from_cache", "messages_embedded",
    "messages_embedded_from_cache", "edges_embedded", "edges_embedded_from_cache",
    "episodes_embedded", "episodes_embedded_from_cache",
    "aggregation_nodes_built", "aggregation_nodes_reused", "aggregation_input_episodes",
    "digest_failures", "episodes_created", "facts_extracted", "fact_failures",
    "facts_embedded", "facts_embedded_from_cache", "profile_items_extracted",
    "digest_quarantined", "profile_failures",
    "aggregation_fusion_failures", "aggregation_build_exceptions",
)
REPORT_OPTIONAL_COUNTS = (
    "aggregation_level0_missed", "aggregation_leaf_changed",
    "aggregation_predicted_rebuild", "aggregation_keying_residual",
    "aggregation_rebuilt_level0", "aggregation_rebuilt_rollup",
    "aggregation_rebuilt_root", "aggregation_leaf_added",
    "aggregation_leaf_removed", "aggregation_facts_rekey",
)
REPORT_FLAGS = (
    "budget_exhausted", "skipped_locked", "extraction_provider_attempt_budget_exhausted",
)
MAX_COUNT = 2_147_483_647
MAX_ROWS = 100_000
MAX_OUTPUT_BYTES = 32_768


REMOTE = r'''
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import time
from pathlib import Path
from urllib.parse import quote

reader = {'__name__': 'pinned_progress', '__file__': '<pinned-progress>'}
exec(compile(READER_SOURCE, '<pinned-progress>', 'exec'), reader)
root = Path(ROOT)

def count(value):
    if type(value) is not int or not 0 <= value <= MAX_COUNT:
        raise ValueError('count_invalid')
    return value

def counts(value, keys):
    if type(value) is not dict or not set(keys).issubset(value):
        raise ValueError('counts_invalid')
    return {key: count(value[key]) for key in keys}

def last_status(value):
    if value is None:
        return None
    if type(value) is not dict or value.get('phase1_backlog_status') != 'current_producer' \
            or value.get('pending_chunks_authoritative') is not True \
            or type(value.get('phase1_generation_key')) is not str \
            or re.fullmatch(r'hymem-phase1-generation-v1:[0-9a-f]{64}',
                            value['phase1_generation_key']) is None:
        raise ValueError('status_invalid')
    terminal = value.get('terminal_loss')
    coverage = value.get('coverage_integrity')
    if type(terminal) is not dict or type(coverage) is not dict:
        raise ValueError('status_invalid')
    return {'pending': counts(value.get('pending'), PENDING),
        'malformed': counts(value.get('malformed'), MALFORMED),
        'quarantined': counts(value.get('quarantined'), QUARANTINED),
        'terminal_loss_chunks': count(terminal.get('chunks')),
        'coverage_integrity_failures': count(coverage.get('failures'))}

def indexing(value):
    if type(value) is not dict or value.get('schema') != INDEXING_SCHEMA \
            or value.get('outcome') != 'failure' \
            or value.get('complete') is not False or value.get('healthy') is not False \
            or type(value.get('failure')) is not dict \
            or value['failure'].get('code') != 'timeout_during_cycle' \
            or value.get('max_cycles') != 100 or value.get('timeout_s') != 10800:
        raise ValueError('indexing_invalid')
    cycles = count(value.get('cycles'))
    if cycles > 100:
        raise ValueError('cycles_invalid')
    elapsed = value.get('elapsed_s')
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) \
            or not 10800 <= elapsed <= 1_000_000:
        raise ValueError('elapsed_invalid')
    reports = value.get('reports')
    if type(reports) is not list or len(reports) != cycles:
        raise ValueError('reports_invalid')
    totals = {key: 0 for key in REPORT_COUNTS + REPORT_OPTIONAL_COUNTS}
    flagged = {key: 0 for key in REPORT_FLAGS}
    first = None
    last = None
    for report in reports:
        item = counts(report, REPORT_COUNTS)
        for key in REPORT_OPTIONAL_COUNTS:
            raw = report.get(key)
            item[key] = None if raw is None else count(raw)
        if any(type(report.get(key)) is not bool for key in REPORT_FLAGS):
            raise ValueError('report_flags_invalid')
        for key in REPORT_COUNTS:
            totals[key] = count(totals[key] + item[key])
        for key in REPORT_OPTIONAL_COUNTS:
            if item[key] is None:
                totals[key] = None
            elif totals[key] is not None:
                totals[key] = count(totals[key] + item[key])
        for key in REPORT_FLAGS:
            flagged[key] = count(flagged[key] + int(report[key]))
        if first is None:
            first = item
        last = item
    cleanup = value.get('cleanup_errors')
    if type(cleanup) is not list or cleanup:
        raise ValueError('indexing_cleanup_invalid')
    return {'cycles_completed': cycles,
        'completed_report_first': first, 'completed_report_last': last,
        'completed_report_totals': totals, 'completed_report_flagged_cycles': flagged,
        'last_completed_status': last_status(value.get('final_status')),
        'status_may_predate_interrupted_cycle': True}

def gated():
    report = reader['inspect'](root, RECEIPT_SHA)
    if (type(report) is not dict or report.get('schema') != 'siwc-lme-diagnostic-progress-v9'
            or report.get('status') != 'terminal_incomplete_or_unclean'
            or report.get('runtime_cleanup_verified') is not True
            or report.get('completed_diagnostic_and_clean') is not False
            or report.get('selected_denominator') != 4
            or report.get('scored_count') != 0 or report.get('failed_count') != 4
            or report.get('correct_count') != 0
            or report.get('canary_structural_valid') is not True
            or report.get('canary_model_gold_match') is not False
            or report.get('strict_indexing_healthy_for_all') is not None
            or report.get('question_failure_codes') !=
            ['indexing_failure:timeout_during_cycle'] * 4
            or report.get('known_turns') != 3838
            or report.get('known_tokens') != 12801578
            or report.get('usage_complete') is not True
            or report.get('campaign_stop') != 'question_failure'
            or report.get('budget_stop_code') != 'question_failure'
            or report.get('first_failure') is not None
            or report.get('owner_failure') is not None
            or report.get('resource_fault') is not None
            or type(report.get('resource_observation')) is not dict
            or any(type(report['resource_observation'].get(key)) is not int
                   or report['resource_observation'][key] != expected
                   for key, expected in (('denials', 0), ('current', 2),
                                         ('peak', 18), ('limit', 256)))):
        raise ValueError('terminal_gate_invalid')
    observations = report.get('siwc_observations')
    slots = ('canary.ordinary', 'canary.structured') + tuple(
        'question.%d.%s' % (index, route) for index in range(4)
        for route in ('ordinary', 'structured'))
    if type(observations) is not dict or set(observations) != set(slots):
        raise ValueError('observation_gate_invalid')
    ledgers = ((11,46706), (948,3151114), (962,3222237),
               (970,3259542), (947,3121979))
    calls = tokens = 0
    for ledger, expected in zip(('canary', 'question.0', 'question.1',
                                  'question.2', 'question.3'), ledgers):
        pair_calls = 0
        for route in ('ordinary', 'structured'):
            item = observations[ledger + '.' + route]
            if type(item) is not dict or item.get('status') != 'observed':
                raise ValueError('observation_gate_invalid')
            summary = item.get('summary')
            if (type(summary) is not dict or summary.get('usage_complete') is not True
                    or summary.get('timing_saturated') is not False
                    or summary.get('first_failure') is not None
                    or summary.get('last_failure_code') is not None
                    or summary.get('failures') != 0
                    or type(summary.get('calls')) is not int
                    or not 0 <= summary['calls'] <= expected[0]
                    or summary.get('successes') != summary['calls']
                    or summary.get('internal_http_attempts') != summary['calls']
                    or type(summary.get('known_tokens')) is not int
                    or summary['known_tokens'] != expected[1]):
                raise ValueError('observation_gate_invalid')
            pair_calls += summary['calls']
        if pair_calls != expected[0]:
            raise ValueError('observation_gate_invalid')
        calls += pair_calls
        tokens += expected[1]
    if calls != 3838 or tokens != 12801578:
        raise ValueError('observation_gate_invalid')
    receipt = reader['receipt'](root, RECEIPT_SHA)
    checkpoint = reader['read'](root / 'run' / 'diagnostic-checkpoint.json', root, 2_000_000)
    ledger, completed, _, _, check = reader['checkpoint'](checkpoint, receipt)
    if (ledger is None or completed != 0 or ledger.get('expected') != 4
            or ledger.get('failed') != 4 or ledger.get('missing') != 0):
        raise ValueError('checkpoint_gate_invalid')
    entries = check['entries']
    ids = check['expected_ids']
    if type(ids) is not list or len(ids) != 4:
        raise ValueError('checkpoint_ids_invalid')
    for ident in ids:
        entry = entries.get(ident)
        if type(entry) is not dict or entry.get('status') != 'failed' \
                or entry.get('failure') != 'indexing_failure:timeout_during_cycle':
            raise ValueError('checkpoint_entry_invalid')

def pathstat(path, directory=False, absent=False):
    try:
        info = path.lstat()
    except FileNotFoundError:
        if absent:
            return None
        raise
    kind = stat.S_ISDIR if directory else stat.S_ISREG
    if not kind(info.st_mode) or info.st_uid != reader['ROOT_UID'] \
            or info.st_mode & 0o022 or (not directory and info.st_nlink != 1):
        raise ValueError('unsafe_path')
    return info

def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)

def digest(path, expected):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        opened = os.fstat(fd)
        if identity(opened) != identity(expected) or not 0 < opened.st_size <= 512*1024*1024:
            raise ValueError('database_changed')
        hashed = hashlib.sha256()
        remaining = opened.st_size
        while remaining:
            data = os.read(fd, min(remaining, 1024*1024))
            if not data:
                raise ValueError('database_changed')
            hashed.update(data)
            remaining -= len(data)
        if os.read(fd, 1) or identity(os.fstat(fd)) != identity(expected):
            raise ValueError('database_changed')
        return hashed.digest()
    finally:
        os.close(fd)

def private_json(path, cap):
    before = pathstat(path)
    if not 0 < before.st_size <= cap:
        raise ValueError('private_size_invalid')
    original = digest(path, before)
    value = reader['read'](path, root, cap)
    if identity(pathstat(path)) != identity(before) or digest(path, before) != original:
        raise ValueError('private_changed')
    return value

def sidecars(path):
    for suffix in ('-wal', '-journal', '-shm'):
        info = pathstat(Path(str(path) + suffix), absent=True)
        if info is not None and info.st_size:
            raise ValueError('sqlite_sidecar_nonempty')

def retry_counts(path):
    before = pathstat(path)
    if not 0 < before.st_size <= 512*1024*1024:
        raise ValueError('database_size_invalid')
    sidecars(path)
    original = digest(path, before)
    if identity(pathstat(path)) != identity(before):
        raise ValueError('database_changed')
    query = ("SELECT CASE WHEN typeof(attempts)='integer' AND attempts=1 THEN '1' "
             "WHEN typeof(attempts)='integer' AND attempts=2 THEN '2' "
             "WHEN typeof(attempts)='integer' AND attempts=3 THEN '3' "
             "WHEN typeof(attempts)='integer' AND attempts>3 THEN '>3' ELSE 'invalid' END, "
             "CASE " + ' '.join("WHEN typeof(last_failure_reason)='text' AND "
             "last_failure_reason COLLATE BINARY = '%s' THEN '%s'" % (reason, reason)
             for reason in REASONS[:-1]) + " ELSE 'other' END, COUNT(*) "
             "FROM chunk_extraction_attempts GROUP BY 1,2")
    start = time.monotonic()
    steps = [0]
    connection = sqlite3.connect('file:' + quote(str(path)) + '?mode=ro&immutable=1',
                                 uri=True, timeout=1)
    try:
        if hasattr(connection, 'enable_load_extension'):
            connection.enable_load_extension(False)
        connection.execute('PRAGMA query_only=ON')
        def authorize(action, arg1, arg2, database, trigger):
            if action == sqlite3.SQLITE_SELECT:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ and database == 'main' and (arg1,arg2) in {
                    ('chunk_extraction_attempts','attempts'),
                    ('chunk_extraction_attempts','last_failure_reason')}:
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_FUNCTION and arg2 in {'typeof','count'}:
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        connection.set_authorizer(authorize)
        def progress():
            steps[0] += 1000
            return int(steps[0] > 20_000_000 or time.monotonic()-start > 15)
        connection.set_progress_handler(progress, 1000)
        buckets = {'1': {}, '2': {}, '3': {}, '>3': {}}
        total = 0
        for attempt, reason, amount in connection.execute(query):
            if attempt not in buckets or reason not in REASONS \
                    or type(amount) is not int or not 0 < amount <= MAX_ROWS \
                    or reason in buckets[attempt]:
                raise ValueError('retry_bucket_invalid')
            total += amount
            if total > MAX_ROWS:
                raise ValueError('retry_row_bound')
            buckets[attempt][reason] = amount
    finally:
        connection.close()
    if identity(pathstat(path)) != identity(before) or digest(path, before) != original:
        raise ValueError('database_changed')
    sidecars(path)
    return {'surviving_held_rows': total, 'by_attempt_and_last_reason': buckets}

gated()
result = {'schema': SCHEMA, 'source_receipt_terminal_cleanup_verified': True,
          'retry_rows_are_consecutive_unsucceeded_not_history': True, 'questions': {}}
run = root / 'run'
pathstat(run, directory=True)
for index in range(4):
    label = 'q-%04d' % index
    directory = run / label
    pathstat(directory, directory=True)
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError('directory_escape')
    private = directory / 'private-indexing.json'
    summary = private_json(private, 262144)
    result['questions'][label] = {'indexing': indexing(summary),
                                  'retry': retry_counts(directory / 'hymem.sqlite')}
encoded = json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False)
if len(encoded.encode('utf-8')) > MAX_OUTPUT_BYTES:
    raise ValueError('projection_too_large')
print(encoded)
'''


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field")
        result[key] = value
    return result


def _count(value: object, maximum: int = MAX_COUNT) -> bool:
    return type(value) is int and 0 <= value <= maximum


def _counts(value: object, keys: tuple[str, ...]) -> bool:
    return type(value) is dict and set(value) == set(keys) and all(
        _count(value[key]) for key in keys)


def _report_counts(value: object) -> bool:
    return (type(value) is dict and set(value) == set(REPORT_COUNTS + REPORT_OPTIONAL_COUNTS)
            and all(_count(value[key]) for key in REPORT_COUNTS)
            and all(value[key] is None or _count(value[key])
                    for key in REPORT_OPTIONAL_COUNTS))


def _valid(value: object) -> dict | None:
    if (type(value) is not dict or set(value) != {
            "schema", "source_receipt_terminal_cleanup_verified",
            "retry_rows_are_consecutive_unsucceeded_not_history", "questions"}
            or value["schema"] != SCHEMA
            or value["source_receipt_terminal_cleanup_verified"] is not True
            or value["retry_rows_are_consecutive_unsucceeded_not_history"] is not True
            or type(value["questions"]) is not dict
            or set(value["questions"]) != {f"q-{i:04d}" for i in range(4)}):
        return None
    for item in value["questions"].values():
        if type(item) is not dict or set(item) != {"indexing", "retry"}:
            return None
        index = item["indexing"]
        if (type(index) is not dict or set(index) != {
                "cycles_completed", "completed_report_first",
                "completed_report_last", "completed_report_totals",
                "completed_report_flagged_cycles", "last_completed_status",
                "status_may_predate_interrupted_cycle"}
                or not _count(index["cycles_completed"], 100)
                or index["status_may_predate_interrupted_cycle"] is not True
                or not _report_counts(index["completed_report_totals"])
                or not _counts(index["completed_report_flagged_cycles"], REPORT_FLAGS)):
            return None
        for key in ("completed_report_first", "completed_report_last"):
            if (index[key] is None) != (index["cycles_completed"] == 0):
                return None
            if index[key] is not None and not _report_counts(index[key]):
                return None
        if index["cycles_completed"] == 0 and (any(index["completed_report_totals"].values())
                or any(index["completed_report_flagged_cycles"].values())):
            return None
        if index["cycles_completed"]:
            if any(index[which][key] > index["completed_report_totals"][key]
                   for which in ("completed_report_first", "completed_report_last")
                   for key in REPORT_COUNTS):
                return None
            if any(index["completed_report_totals"][key] is not None
                   and index[which][key] is not None
                   and index[which][key] > index["completed_report_totals"][key]
                   for which in ("completed_report_first", "completed_report_last")
                   for key in REPORT_OPTIONAL_COUNTS):
                return None
        if any(value > index["cycles_completed"] for value in
               index["completed_report_flagged_cycles"].values()):
            return None
        status = index["last_completed_status"]
        if status is not None and (type(status) is not dict or set(status) != {
                "pending", "malformed", "quarantined", "terminal_loss_chunks",
                "coverage_integrity_failures"}
                or not _counts(status["pending"], PENDING)
                or not _counts(status["malformed"], MALFORMED)
                or not _counts(status["quarantined"], QUARANTINED)
                or not _count(status["terminal_loss_chunks"])
                or not _count(status["coverage_integrity_failures"])):
            return None
        retry = item["retry"]
        if (type(retry) is not dict or set(retry) != {
                "surviving_held_rows", "by_attempt_and_last_reason"}
                or not _count(retry["surviving_held_rows"], MAX_ROWS)
                or type(retry["by_attempt_and_last_reason"]) is not dict
                or set(retry["by_attempt_and_last_reason"]) != {"1", "2", "3", ">3"}):
            return None
        total = 0
        for bucket in retry["by_attempt_and_last_reason"].values():
            if type(bucket) is not dict or not set(bucket).issubset(REASONS) \
                    or any(not _count(number, MAX_ROWS) or number == 0
                           for number in bucket.values()):
                return None
            total += sum(bucket.values())
        if total != retry["surviving_held_rows"]:
            return None
    return value


def _payload() -> str:
    source = READER.read_bytes()
    if hashlib.sha256(source).hexdigest() != READER_SHA:
        raise ValueError("source_pin_mismatch")
    constants = {name: globals()[name] for name in (
        "SCHEMA", "ROOT", "RECEIPT_SHA", "INDEXING_SCHEMA", "REASONS", "PENDING",
        "MALFORMED", "QUARANTINED", "REPORT_COUNTS", "REPORT_OPTIONAL_COUNTS",
        "REPORT_FLAGS", "MAX_COUNT",
        "MAX_ROWS", "MAX_OUTPUT_BYTES")}
    body = "\n".join("    " + line if line else "" for line in REMOTE.splitlines())
    return ("READER_SOURCE=" + repr(source.decode("utf-8")) + "\n" +
            "\n".join(f"{key}={value!r}" for key, value in constants.items()) +
            "\ntry:\n" + body + "\nexcept BaseException:\n"
            "    print(" + repr(json.dumps({"schema": SCHEMA,
                "status": "inspection_unavailable"}, separators=(",", ":"))) + ")\n")


def _invoke(payload: str) -> bytes:
    """Bound both time and bytes; host stderr is discarded without inspection."""
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o",
               "ConnectionAttempts=1", "afrodite", "/usr/bin/python3 -I -B -"]
    deadline = time.monotonic() + 90
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL) as proc:
        assert proc.stdin is not None and proc.stdout is not None
        data = memoryview(payload.encode("utf-8"))
        offset = 0
        output = bytearray()
        selector = selectors.DefaultSelector()
        try:
            os.set_blocking(proc.stdin.fileno(), False)
            os.set_blocking(proc.stdout.fileno(), False)
            selector.register(proc.stdin, selectors.EVENT_WRITE)
            selector.register(proc.stdout, selectors.EVENT_READ)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("inspection_timeout")
                for key, _ in selector.select(remaining):
                    if key.fileobj is proc.stdin:
                        try:
                            written = os.write(proc.stdin.fileno(), data[offset:offset + 65536])
                        except BrokenPipeError:
                            written = 0
                            offset = len(data)
                        offset += written
                        if offset == len(data):
                            selector.unregister(proc.stdin)
                            proc.stdin.close()
                    else:
                        block = os.read(proc.stdout.fileno(), MAX_OUTPUT_BYTES + 1 - len(output))
                        if not block:
                            selector.unregister(proc.stdout)
                            proc.stdout.close()
                        else:
                            output.extend(block)
                            if len(output) > MAX_OUTPUT_BYTES:
                                raise ValueError("inspection_output_oversize")
            if proc.wait(timeout=max(0, deadline - time.monotonic())) != 0:
                raise ValueError("inspection_unavailable")
            return bytes(output)
        except BaseException:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
            raise
        finally:
            selector.close()


def main() -> int:
    try:
        raw = _invoke(_payload())
        decoded = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite")))
        safe = _valid(decoded)
        if safe is None:
            raise ValueError("projection_invalid")
        print(json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except (OSError, UnicodeError, subprocess.SubprocessError, ValueError, TypeError):
        print(json.dumps({"schema": SCHEMA, "status": "inspection_unavailable"},
                         sort_keys=True, separators=(",", ":")))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
