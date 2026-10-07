"""Root's read-only fault localization for the first diagnostic LME pilot.

Run locally: validates and sends only the accepted reader plus this bounded
metadata projection to Afrodite. Does not read logs, stores or private rows.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

READER_SHA = "3ceb2466ce0a3848e42e2264f1061f7905fd13a92deca84af812f0d79860480a"
READER = Path(__file__).with_name("luna_lme_diagnostic_progress_v1.py")
REMOTE = r'''
namespace = {'__name__': 'reviewed_reader', '__file__': '<reviewed-reader>'}
exec(compile(SOURCE, '<reviewed-reader>', 'exec'), namespace)
root = namespace['Path']('/home/atta/.hymem-lme-diagnostic-preflight-oi233cee')
receipt_sha = '67ce8fec27a8848612c9ae99f3cbd9cf42da41c31762ed7970eca25604548c2c'
allowed = {
 'root_invalid','root_permission_invalid','receipt_sha_invalid','receipt_invalid',
 'receipt_identity_invalid','inventory_drift','inventory_shape_invalid',
 'inventory_entries_invalid','inventory_map_drift','candidate_missing',
 'candidate_symlink','candidate_special_file','candidate_source_drift',
 'pinned_code_drift','checkpoint_invalid','checkpoint_ids_invalid',
 'checkpoint_identity_invalid','checkpoint_row_identity_invalid',
 'checkpoint_limits_invalid','checkpoint_entries_invalid','checkpoint_entry_invalid',
 'checkpoint_projection_invalid','checkpoint_projection_inconsistent',
 'checkpoint_counts_invalid','terminal_identity_invalid','terminal_score_counts_invalid',
 'terminal_score_mismatch','terminal_accuracy_invalid','terminal_partial_accuracy_invalid',
 'terminal_canary_invalid','terminal_budget_invalid','metadata_file_invalid',
 'metadata_size_invalid','duplicate_metadata_field','nonfinite_metadata',
 'terminal_before_checkpoint_complete','terminal_without_checkpoint',
 'launch_marker_invalid','launch_command_result_invalid','unlaunched_state_drift',
}
out = {'schema': 'luna-diagnostic-failure-metadata-root-v1'}
rpc_methods = {'initialize','config/read','model/list','account/read',
 'account/rateLimits/read','thread/start','turn/start','thread/unsubscribe','startup','turn/events'}
event_methods = {'thread/started','thread/status/changed','thread/closed','turn/started',
 'turn/completed','turn/failed','item/started','item/completed','item/agentMessage/delta',
 'item/reasoning/textDelta','thread/tokenUsage/updated','account/updated','account/rateLimits/updated',
 'remoteControl/status/changed','warning','invalid_method','error','model/rerouted',
 'model/verification','model/safetyBuffering/updated','turn/plan/updated','turn/diff/updated'}
stop_codes = {None,'RuntimeError','ValueError','KeyboardInterrupt','finite_other',
 'final_accounting_or_dataset_failure','incomplete_questions','campaign_failure',
 'question_failure','usage_unknown','worker_runtime_failure','quota_exhausted',
 'auth_failure','fixed_other','quota_floor','quota_unverified','transport_failure'}
stop_codes |= {family + ':' + method for family in ('process_exit','protocol_failure','rpc_failure')
               for method in rpc_methods}
stop_codes |= {'unexpected_notification:' + method for method in event_methods}
stop_codes |= {'unexpected_notification:' + method.replace('/','_') for method in event_methods}
try:
    namespace['_root'](root)
    receipt = namespace['_receipt'](root, receipt_sha)
    out['source_receipt_verified'] = True
except Exception as exc:
    out['source_receipt_verified'] = False
    out['validation_error'] = str(exc) if type(exc) is ValueError and str(exc) in allowed else 'unclassified'
    print(json.dumps(out, sort_keys=True))
    raise SystemExit(1)
try:
    namespace['inspect'](root, receipt_sha)
    out['reader_validated'] = True
except Exception as exc:
    out['reader_validated'] = False
    out['validation_error'] = str(exc) if type(exc) is ValueError and str(exc) in allowed else 'unclassified'
for label, filename, cap in (
    ('checkpoint', 'diagnostic-checkpoint.json', 2000000),
    ('terminal', 'diagnostic-result.json', 128000),
):
    path = root / 'run' / filename
    out[label + '_exists'] = path.is_file()
    if not path.is_file():
        continue
    out[label + '_bytes'] = path.stat().st_size
    value = namespace['_read'](path, root, cap)
    out[label + '_object'] = type(value) is dict
    if type(value) is not dict:
        continue
    if label == 'checkpoint':
        out['checkpoint_status'] = value.get('status') if value.get('status') in ('running','complete') else 'other'
        entries = value.get('entries')
        if type(entries) is dict:
            out['entry_count'] = len(entries)
            out['completed_entries'] = sum(type(v) is dict and v.get('status') == 'completed' for v in entries.values())
            out['failed_entries'] = sum(type(v) is dict and v.get('status') == 'failed' for v in entries.values())
    else:
        out['campaign_stop'] = value.get('campaign_stop') if value.get('campaign_stop') in stop_codes else 'other'
        canary = value.get('canary')
        out['canary_present'] = type(canary) is dict
        if type(canary) is dict:
            out['canary_flags'] = {k: canary.get(k) for k in (
                'structural_valid','model_gold_match','semantic_failure_proved')
                if type(canary.get(k)) is bool}
            out['canary_numbers'] = {k: canary.get(k) for k in (
                'completion_calls','known_tokens','triples','episodes')
                if type(canary.get(k)) is int and canary[k] >= 0}
            out['canary_quality_reason'] = canary.get('quality_failure_reason') if canary.get('quality_failure_reason') in (
                None, 'branch_incomplete','parse_failure','item_validation_failure',
                'response_conflict','grounding_failure','truncated') else 'other'
        budget = value.get('budget')
        if type(budget) is dict:
            out['budget_numbers'] = {k: budget.get(k) for k in (
                'turns','reserved','known_tokens','in_flight')
                if type(budget.get(k)) is int and budget[k] >= 0}
            out['budget_flags'] = {k: budget.get(k) for k in ('usage_complete','stopped')
                if type(budget.get(k)) is bool}
            out['budget_stop'] = budget.get('stop_code') if budget.get('stop_code') in stop_codes else 'other'
            failure = budget.get('first_failure')
            if type(failure) is dict:
                first = {'code': failure.get('code') if failure.get('code') in stop_codes else 'other',
                    'phase': failure.get('phase') if failure.get('phase') in (
                        'startup','preflight','run','unsubscribe','rotation_cleanup','cleanup') else 'other',
                    'rpc': failure.get('rpc') if failure.get('rpc') in rpc_methods else None}
                for key in ('process_index','request_index','retired_count','queue_count','known_tokens'):
                    item = failure.get(key)
                    if type(item) is int and 0 <= item <= 1000000000:
                        first[key] = item
                for key in ('turn_admitted','known_usage','usage_complete'):
                    if type(failure.get(key)) is bool:
                        first[key] = failure[key]
                family = failure.get('event_family')
                if family in {method.replace('/','_') for method in event_methods} | {'unknown','invalid_method'}:
                    first['event_family'] = family
                rpc_error = failure.get('rpc_error')
                if type(rpc_error) is dict:
                    first['rpc_error_category'] = rpc_error.get('category') if rpc_error.get('category') in (
                        'parse_error','invalid_request','method_not_found','invalid_params','internal_error','other_numeric','invalid') else 'other'
                    number = rpc_error.get('code')
                    if type(number) is int and -2147483648 <= number <= 2147483647:
                        first['rpc_error_code'] = number
                app_error = failure.get('app_server_error')
                if type(app_error) is dict:
                    first['app_error_identity'] = app_error.get('identity') if app_error.get('identity') in (
                        'invalid','unbound','mismatch','matched') else 'other'
                    first['app_error_class'] = app_error.get('error_class') if app_error.get('error_class') in (
                        'contextWindowExceeded','sessionBudgetExceeded','usageLimitExceeded','rateLimitExceeded',
                        'flexUnavailable','serverOverloaded','cyberPolicy','misalignmentPolicyViolation',
                        'internalServerError','unauthorized','badRequest','threadRollbackFailed',
                        'sandboxError','other','httpConnectionFailed','responseStreamConnectionFailed',
                        'responseStreamDisconnected','responseTooManyFailedAttempts','activeTurnNotSteerable',
                        'invalid','unspecified') else 'other'
                out['first_failure'] = first
group = namespace['CGROUP_ROOT'] / receipt['expected_cgroup'].lstrip('/')
out['cgroup_exists'] = group.exists()
out['cgroup_empty'] = namespace['_cgroup_empty'](group) if group.exists() else True
print(json.dumps(out, sort_keys=True))
'''


def main() -> int:
    source = READER.read_bytes()
    if hashlib.sha256(source).hexdigest() != READER_SHA:
        raise SystemExit("local_reader_pin_mismatch")
    payload = "import json\nSOURCE = " + repr(source.decode()) + "\n" + REMOTE
    result = subprocess.run([
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o",
        "ConnectionAttempts=1", "afrodite", "/usr/bin/python3 -I -B -",
    ], input=payload, text=True, capture_output=True, timeout=30)
    try:
        metadata = json.loads(result.stdout)
        if type(metadata) is not dict or metadata.get("schema") != "luna-diagnostic-failure-metadata-root-v1":
            raise ValueError("metadata_shape")
        print(json.dumps(metadata, sort_keys=True))
    except (ValueError, TypeError):
        print(json.dumps({"schema": "luna-diagnostic-failure-metadata-root-v1",
                          "status": "metadata_unavailable", "returncode": result.returncode}))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
