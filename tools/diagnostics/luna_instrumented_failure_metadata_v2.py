"""Bounded final-status counters for the already stopped afit9i7d pilot.

Extends the frozen v1 projection after its terminal/source/cleanup gate. The
remote side reads only the same fixed per-question private-indexing files and
never imports candidate code or exports private rows, reason maps, or hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from tools.diagnostics import luna_instrumented_failure_metadata_v1 as v1


SCHEMA = "luna-instrumented-failure-metadata-v2"
V1_SHA = "a892dbd759ab0ebd5bfacf81eee0b7e5e00f5de530cf1cc340453f235d164826"

# Exact keys from the frozen bundle's benchmarks/lme_protocol.py (d39f8409)
# and hymem/dreaming/status.py (562814ce). No keys come from private JSON.
PENDING = (
    "pending_source_materialization", "pending_chunks", "pending_digests",
    "pending_profiles", "pending_facts", "pending_aggregation",
    "pending_chunk_embeddings", "pending_message_embeddings",
    "pending_edge_embeddings", "pending_episode_embeddings",
    "pending_fact_embeddings",
)
MALFORMED = (
    "malformed_source_materialization", "malformed_digests",
    "malformed_profiles", "malformed_facts", "malformed_summaries",
)
QUARANTINED = (
    "quarantined_chunks", "quarantined_digests", "quarantined_profiles",
    "quarantined_facts", "quarantined_facts_malformed",
)
SUMMARY_COUNTS = (
    "summary_degraded_sessions", "summary_missing_sessions",
    "malformed_summaries",
)
MAX_COUNT = 2_147_483_647

EXTENSION = r'''
def _count(value):
    if type(value) is not int or not 0 <= value <= MAX_COUNT:
        raise ValueError('final_count_invalid')
    return value

def _counts(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError('final_counts_shape_invalid')
    return {key: _count(value[key]) for key in keys}

def _final(value):
    if value is None:
        return None
    if type(value) is not dict:
        raise ValueError('final_status_invalid')
    pending = _counts(value.get('pending'), PENDING)
    malformed = _counts(value.get('malformed'), MALFORMED)
    quarantined = _counts(value.get('quarantined'), QUARANTINED)
    terminal = value.get('terminal_loss')
    coverage = value.get('coverage_integrity')
    summary = value.get('summary_health')
    if type(terminal) is not dict or type(coverage) is not dict:
        raise ValueError('final_status_invalid')
    if type(summary) is not dict or set(summary) != set(SUMMARY_COUNTS) | {'summary_healthy'}:
        raise ValueError('summary_health_invalid')
    health = {key: _count(summary[key]) for key in SUMMARY_COUNTS}
    flag = summary['summary_healthy']
    if type(flag) is not bool or flag is not (
            health['summary_degraded_sessions'] == 0 and health['malformed_summaries'] == 0):
        raise ValueError('summary_health_invalid')
    if (health['summary_missing_sessions'] > health['summary_degraded_sessions']
            or malformed['malformed_summaries'] != health['malformed_summaries']):
        raise ValueError('summary_health_invalid')
    health['summary_healthy'] = flag
    return {'pending': pending, 'malformed': malformed,
        'quarantined': quarantined, 'terminal_loss': {'chunks': _count(terminal.get('chunks'))},
        'coverage_integrity': {'failures': _count(coverage.get('failures'))},
        'summary_health': health}

def _extend(root, namespace, report, base):
    # v1 _project already enforced terminal/source/checkpoint/cleanup gates.
    if (type(base) is not dict or base.get('schema') != V1_SCHEMA
            or base.get('terminal_and_cleanup_verified') is not True
            or base.get('source_receipt_verified') is not True):
        raise ValueError('v1_gate_invalid')
    base['schema'] = SCHEMA
    for index in range(4):
        label = 'q-%04d' % index
        item = base['questions'][label]['indexing']
        if not item['present']:
            item['final_status'] = None
            continue
        path = root/'run'/label/'private-indexing.json'
        value = namespace['_read'](path, root, 262144)
        if V1_FIELDS(value) != {key: data for key, data in item.items()
                              if key != 'present'}:
            raise ValueError('private_metadata_changed')
        item['final_status'] = _final(value.get('final_status') if type(value) is dict else None)
    return base
'''

REMOTE = r'''
namespace = {'__name__': 'reviewed_reader', '__file__': '<reviewed-reader>'}
exec(compile(SOURCE, '<reviewed-reader>', 'exec'), namespace)
root = namespace['Path'](ROOT)
report = namespace['inspect'](root, RECEIPT_SHA)
scope = {'INDEXING_CODES': INDEXING_CODES, 'EXCEPTION_TYPES': EXCEPTION_TYPES,
         'RECEIPT_SHA': RECEIPT_SHA, 'SCHEMA': V1_SCHEMA}
exec(compile(V1_PROJECTION, '<frozen-v1-projection>', 'exec'), scope)
base = scope['_project'](root, namespace, report)
extension = {'PENDING': PENDING, 'MALFORMED': MALFORMED,
             'QUARANTINED': QUARANTINED, 'SUMMARY_COUNTS': SUMMARY_COUNTS,
             'MAX_COUNT': MAX_COUNT, 'V1_SCHEMA': V1_SCHEMA, 'SCHEMA': SCHEMA,
             'V1_FIELDS': scope['_fields']}
exec(compile(EXTENSION, '<bounded-final-projection>', 'exec'), extension)
print(json.dumps(extension['_extend'](root, namespace, report, base),
                 sort_keys=True, separators=(',', ':'), allow_nan=False))
'''


def _validated_final(value: object) -> bool:
    if value is None:
        return True
    if type(value) is not dict or set(value) != {
            "pending", "malformed", "quarantined", "terminal_loss",
            "coverage_integrity", "summary_health"}:
        return False
    for key, names in (("pending", PENDING), ("malformed", MALFORMED),
                       ("quarantined", QUARANTINED),
                       ("terminal_loss", ("chunks",)),
                       ("coverage_integrity", ("failures",))):
        counts = value[key]
        if (type(counts) is not dict or set(counts) != set(names)
                or any(type(counts[name]) is not int or not 0 <= counts[name] <= MAX_COUNT
                       for name in names)):
            return False
    summary = value["summary_health"]
    if (type(summary) is not dict
            or set(summary) != set(SUMMARY_COUNTS) | {"summary_healthy"}
            or any(type(summary[name]) is not int or not 0 <= summary[name] <= MAX_COUNT
                   for name in SUMMARY_COUNTS)
            or type(summary["summary_healthy"]) is not bool):
        return False
    return (summary["summary_missing_sessions"] <= summary["summary_degraded_sessions"]
        and summary["summary_healthy"] is (
            summary["summary_degraded_sessions"] == 0
            and summary["malformed_summaries"] == 0)
        and value["malformed"]["malformed_summaries"] == summary["malformed_summaries"])


def _validated(value: object) -> dict | None:
    if type(value) is not dict or value.get("schema") != SCHEMA:
        return None
    prior = {**value, "schema": v1.SCHEMA}
    if type(prior.get("questions")) is not dict:
        return None
    prior["questions"] = {}
    for label, question in value["questions"].items():
        if type(question) is not dict or set(question) != {
                "checkpoint_failure_code", "private_row_present", "indexing",
                "diagnostic_indexing"}:
            return None
        item = question["indexing"]
        if (type(item) is not dict or "final_status" not in item
                or (item.get("present") is False and item["final_status"] is not None)
                or not _validated_final(item["final_status"])):
            return None
        prior["questions"][label] = {**question,
            "indexing": {key: val for key, val in item.items() if key != "final_status"}}
    if v1._validated(prior) is None:
        return None
    return value


def main() -> int:
    try:
        if hashlib.sha256(Path(v1.__file__).read_bytes()).hexdigest() != V1_SHA:
            raise ValueError("v1_pin_mismatch")
        source = v1.READER.read_bytes()
        if hashlib.sha256(source).hexdigest() != v1.READER_SHA:
            raise ValueError("reader_pin_mismatch")
        payload = ("import json\nSOURCE=" + repr(source.decode("utf-8")) + "\n"
            + "ROOT=" + repr(v1.ROOT) + "\nRECEIPT_SHA=" + repr(v1.RECEIPT_SHA)
            + "\nV1_SCHEMA=" + repr(v1.SCHEMA) + "\nSCHEMA=" + repr(SCHEMA)
            + "\nINDEXING_CODES=" + repr(v1.INDEXING_CODES)
            + "\nEXCEPTION_TYPES=" + repr(v1.EXCEPTION_TYPES)
            + "\nV1_PROJECTION=" + repr(v1.PROJECTION)
            + "\nPENDING=" + repr(PENDING) + "\nMALFORMED=" + repr(MALFORMED)
            + "\nQUARANTINED=" + repr(QUARANTINED)
            + "\nSUMMARY_COUNTS=" + repr(SUMMARY_COUNTS)
            + "\nMAX_COUNT=" + repr(MAX_COUNT)
            + "\nEXTENSION=" + repr(EXTENSION) + "\n" + REMOTE)
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o",
            "ConnectTimeout=10", "-o", "ConnectionAttempts=1", "afrodite",
            "/usr/bin/python3 -I -B -"], input=payload, text=True,
            capture_output=True, timeout=30)
        if result.returncode != 0 or len(result.stdout) > 32768:
            raise ValueError("remote_unavailable")
        decoded = json.loads(result.stdout, parse_constant=lambda _: (_ for _ in ()).throw(
            ValueError("nonfinite_metadata")))
        safe = _validated(decoded)
        if safe is None:
            raise ValueError("projection_invalid")
        print(json.dumps(safe, sort_keys=True, separators=(",", ":"), allow_nan=False))
        return 0
    except (OSError, subprocess.SubprocessError, ValueError, TypeError):
        print(json.dumps({"schema": SCHEMA, "status": "metadata_unavailable"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
