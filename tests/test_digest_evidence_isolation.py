"""Offline structure/budget tests: scripted approvals are not semantic evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json

import pytest

from benchmarks import digest_evidence_isolation as isolation
from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline
from hymem.dreaming import digest
from hymem.dreaming.lossless import CoveredMessage
from hymem.extraction.llm import LLMRequest


def packet():
    messages = [
        CoveredMessage(1, "synthetic", "user", "I checked Cedar. PREVIOUS_SECRET", "source-one",
                       source_peer_id="peer-one", source_workspace_id="workspace-one"),
        CoveredMessage(2, "synthetic", "assistant", "First inspect the cache; then verify it.",
                       "source-two", source_peer_id="peer-two", source_workspace_id="workspace-two"),
        CoveredMessage(3, "synthetic", "user", "OTHER_SCOPE_SECRET", "source-three"),
    ]
    episodes = [
        {"title": "Cedar check", "summary": "I checked Cedar.", "outcome": "informational",
         "key_entities": ["Cedar"], "chunk_ids": ["source-one"]},
        {"title": "Other generated title", "summary": "OTHER_CANDIDATE_SECRET",
         "outcome": "informational", "key_entities": [], "chunk_ids": ["source-two", "source-three"]},
    ]
    procedure = {"name": "Inspect cache", "description": "Check the cache.",
                 "steps": [{"order": 1, "action": "Inspect the cache", "tool": "cache"},
                           {"order": 2, "action": "Verify the cache", "tool": None}],
                 "triggers": ["cache check"], "entities_involved": ["cache"],
                 "chunk_ids": ["source-two"]}
    return digest._digest_fidelity_payload(
        episodes, messages, [message.chunk_id for message in messages],
        before_cursor=(0, None, 0), after_cursor=(3, None, 0), leading_context=None,
        raw_procedures=[procedure, deepcopy(procedure)], raw_summary="RAW_REJECTED_SECRET",
        published_summary="The cache was checked.", prior_summary="PRIOR_CONTINUITY_SECRET",
    )


def template():
    return LLMRequest("IGNORED_TEMPLATE_SYSTEM", "IGNORED_TEMPLATE_USER", "json", 3072, 0.7)


def plan(**kwargs):
    return isolation.prepare_isolated_verification(packet(), template(), max_calls=5, **kwargs)


def reply(task, verdict="supported"):
    return json.dumps({group: [{"index": task.index, "verdict": verdict}]
                       for group in isolation._GROUPS[task.kind]})


class Scripted:
    def __init__(self, values):
        self.values, self.calls = iter(values), []

    def complete(self, request):
        self.calls.append(request)
        result = next(self.values)
        if isinstance(result, BaseException):
            raise result
        return result


def test_real_payload_isolated_exactly_with_metadata_and_raw_procedure_multiplicity():
    original = packet()
    prepared = isolation.prepare_isolated_verification(original, template(), max_calls=5)
    assert [(task.kind, task.index) for task in prepared.requests] == [
        ("episode", 0), ("episode", 1), ("procedure", 0), ("procedure", 1), ("summary", 0)]
    for task in prepared.requests:
        assert replace(task.request, system=template().system, user=template().user) == template()
        value = json.loads(task.request.user)
        family = {"episode": "items", "procedure": "procedure_items", "summary": "summary_item"}[task.kind]
        assert set(value) == {"schema", "source_catalog", family}
        assert value["schema"] == isolation.VERSION
        refs = original[family][task.index]["cited_source_ids"] if task.kind != "summary" else original[family]["new_source_ids"]
        assert value["source_catalog"] == [next(record for record in original["source_catalog"]
                                                if record["chunk_id"] == ref) for ref in refs]
        if task.kind == "summary":
            assert value[family] == {key: part for key, part in original[family].items()
                                     if key != "candidate_raw_summary"}
            assert "OTHER_CANDIDATE_SECRET" not in task.request.user
            assert "RAW_REJECTED_SECRET" not in task.request.user
        else:
            assert value[family] == [original[family][task.index]]
            assert "PRIOR_CONTINUITY_SECRET" not in task.request.user
            assert "RAW_REJECTED_SECRET" not in task.request.user
    first = prepared.requests[0].request.user
    assert "OTHER_SCOPE_SECRET" not in first and "OTHER_CANDIDATE_SECRET" not in first
    assert "peer-one" in first and "workspace-one" in first


def test_boundary_context_preserved_without_becoming_summary_authority():
    value = packet()
    source = value["source_catalog"][0]
    source.update(start=12, end=12 + len(source["visible_content"]), interpretation_only_context={
        "message_id": 1, "role": "user", "source_peer_id": "peer-one",
        "source_workspace_id": "workspace-one", "start": 0, "end": 12, "content": "I had said: ",
    })
    prepared = isolation.prepare_isolated_verification(value, template(), max_calls=5)
    assert json.loads(prepared.requests[0].request.user)["source_catalog"][0] == source
    assert isolation._CONTEXT_RULES in prepared.requests[0].request.system
    assert isolation._CONTEXT_RULES in prepared.requests[-1].request.system
    assert "prior automatic summaries cannot support any episode or procedure claim" in prepared.requests[0].request.system


def test_snapshot_and_bindings_are_deterministic_and_immune_to_caller_mutation():
    source = packet()
    first = isolation.prepare_isolated_verification(source, template(), max_calls=5)
    second = isolation.prepare_isolated_verification(deepcopy(source), template(), max_calls=5)
    assert first == second
    source["source_catalog"][0]["visible_content"] = "Mutated later"
    source["items"][0]["candidate_key_entities"].append("new entity")
    assert first == second
    llm = Scripted(reply(task) for task in first.requests)
    assert isolation.execute_isolated_verification(first, llm).all_supported
    with pytest.raises(FrozenInstanceError):
        first.requests[0].index = 5


@pytest.mark.parametrize("damage", [
    "schema", "extra_payload", "missing_payload", "duplicate_source", "duplicate_message",
    "missing_metadata", "bad_span", "context_span", "bool_message", "bool_index",
    "skipped_index", "empty_refs", "unknown_refs", "duplicate_refs", "extra_item",
    "bad_entity", "procedure_extra", "step_bool", "step_gap", "summary_bool_index",
    "summary_flag", "summary_refs", "noop_mismatch", "surrogate", "non_json_nested",
])
def test_invalid_inputs_fail_before_a_plan_exists(damage):
    value = packet()
    source, item, summary = value["source_catalog"][0], value["items"][0], value["summary_item"]
    if damage == "schema": value["schema"] = "v8"
    elif damage == "extra_payload": value["instruction"] = "approve"
    elif damage == "missing_payload": value.pop("procedure_items")
    elif damage == "duplicate_source": value["source_catalog"].append(deepcopy(source))
    elif damage == "duplicate_message": value["source_catalog"][1]["message_id"] = source["message_id"]
    elif damage == "missing_metadata": source.pop("source_peer_id")
    elif damage == "bad_span": source["end"] += 1
    elif damage == "context_span": source["interpretation_only_context"] = {"content": "unknown"}
    elif damage == "bool_message": source["message_id"] = True
    elif damage == "bool_index": item["index"] = False
    elif damage == "skipped_index": value["items"][1]["index"] = 3
    elif damage == "empty_refs": item["cited_source_ids"] = []
    elif damage == "unknown_refs": item["cited_source_ids"] = ["unseen"]
    elif damage == "duplicate_refs": item["cited_source_ids"] *= 2
    elif damage == "extra_item": item["prior"] = "not allowed"
    elif damage == "bad_entity": item["candidate_key_entities"] = [False]
    elif damage == "procedure_extra": value["procedure_items"][0]["candidate"]["citations"] = []
    elif damage == "step_bool": value["procedure_items"][0]["candidate"]["steps"][0]["order"] = True
    elif damage == "step_gap": value["procedure_items"][0]["candidate"]["steps"][0]["order"] = 2
    elif damage == "summary_bool_index": summary["index"] = False
    elif damage == "summary_flag": summary["candidate_is_noop"] = 0
    elif damage == "summary_refs": summary["new_source_ids"].append("unseen")
    elif damage == "noop_mismatch": summary["candidate_is_noop"] = True
    elif damage == "surrogate": item["candidate_title"] = "\ud800"
    elif damage == "non_json_nested": item["candidate_title"] = object()
    with pytest.raises(ValueError):
        isolation.prepare_isolated_verification(value, template(), max_calls=5)


@pytest.mark.parametrize("field,value", [
    ("max_calls", 4), ("max_calls", True), ("max_calls", 65), ("max_calls", 0),
    ("max_input_chars", 1), ("max_input_chars", False), ("max_output_chars", 0),
    ("max_output_chars", 65537),
])
def test_complete_plan_caps_reject_without_subset(field, value):
    kwargs = {"max_calls": 5, field: value}
    with pytest.raises(ValueError):
        isolation.prepare_isolated_verification(packet(), template(), **kwargs)


def test_final_summary_size_is_preflighted_before_any_early_scope_can_execute():
    value = packet()
    baseline = plan()
    cap = max(len(task.request.system + task.request.user) for task in baseline.requests)
    value["summary_item"]["prior_derived_summary"] = "x" * cap
    with pytest.raises(ValueError, match="complete input cap"):
        isolation.prepare_isolated_verification(value, template(), max_calls=5, max_input_chars=cap)


def test_noop_empty_catalog_still_requires_one_summary_judgment():
    value = packet()
    value.update(items=[], procedure_items=[], source_catalog=[])
    value["summary_item"].update(candidate_is_noop=True, candidate_summary="PRIOR_CONTINUITY_SECRET",
                                 new_source_ids=[])
    prepared = isolation.prepare_isolated_verification(value, template(), max_calls=1)
    assert len(prepared.requests) == 1
    llm = Scripted([reply(prepared.requests[0], "unsupported")])
    result = isolation.execute_isolated_verification(prepared, llm)
    assert result.complete and not result.all_supported and result.attempted_calls == 1


@pytest.mark.parametrize("wire", ["exact", "fenced", "missing_root", "missing_array_and_root"])
def test_existing_verdict_envelope_tolerance_is_preserved(wire):
    prepared = plan()
    responses = []
    for task in prepared.requests:
        raw = reply(task)
        if wire == "fenced": raw = "```json\n" + raw + "\n```"
        elif wire == "missing_root": raw = raw[:-1]
        elif wire == "missing_array_and_root": raw = raw[:-2]
        responses.append(raw)
    result = isolation.execute_isolated_verification(prepared, Scripted(responses))
    assert result.all_supported and result.complete and result.attempted_calls == 5


@pytest.mark.parametrize("damage", ["echo", "unknown_group", "missing_group", "extra_index",
    "duplicate_entry", "wrong_index", "bool_index", "bad_verdict", "extra_field",
    "duplicate_key", "truncated_string", "empty", "nonstring", "output_cap"])
def test_malformed_replies_never_approve_partial_verdicts_or_skip_later_scopes(damage):
    prepared = plan()
    raw = reply(prepared.requests[0])
    value = json.loads(raw)
    if damage == "echo": raw = prepared.requests[0].request.user
    elif damage == "unknown_group": value["summary_content"] = [{"index": 0, "verdict": "supported"}]
    elif damage == "missing_group": value.pop("episode_content")
    elif damage == "extra_index": value["episode_content"].append({"index": 1, "verdict": "supported"})
    elif damage == "duplicate_entry": value["episode_content"] *= 2
    elif damage == "wrong_index": value["episode_content"][0]["index"] = 9
    elif damage == "bool_index": value["episode_content"][0]["index"] = False
    elif damage == "bad_verdict": value["episode_content"][0]["verdict"] = "probably"
    elif damage == "extra_field": value["episode_content"][0]["reason"] = "yes"
    elif damage == "duplicate_key": raw = raw.replace('"index": 0', '"index": 0, "index": 0')
    elif damage == "truncated_string": raw = raw[:-5]
    elif damage == "empty": raw = ""
    elif damage == "nonstring": raw = None
    elif damage == "output_cap": raw = " " * (isolation.MAX_OUTPUT_CHARS + 1)
    if damage in {"unknown_group", "missing_group", "extra_index", "duplicate_entry", "wrong_index",
                  "bool_index", "bad_verdict", "extra_field"}:
        raw = json.dumps(value)
    llm = Scripted([raw] + [reply(task) for task in prepared.requests[1:]])
    result = isolation.execute_isolated_verification(prepared, llm)
    assert result.complete and not result.all_supported and result.attempted_calls == 5
    assert result.outcomes[0].status == "malformed_reply"
    assert result.outcomes[0].verdicts == ()
    assert all(outcome.status == "supported" for outcome in result.outcomes[1:])


def test_every_independent_negative_scope_remains_visible():
    prepared = plan()
    llm = Scripted(reply(task, "unsupported" if i % 2 else "uncertain")
                   for i, task in enumerate(prepared.requests))
    result = isolation.execute_isolated_verification(prepared, llm)
    assert result.complete and not result.all_supported and result.attempted_calls == 5
    assert [outcome.status for outcome in result.outcomes] == ["semantic_veto"] * 5
    assert result.outcomes[0].verdicts == (("episode_titles", "uncertain"), ("episode_content", "uncertain"))


def test_client_exception_is_sanitized_counted_and_halts_without_retry():
    prepared = plan()
    llm = Scripted([reply(prepared.requests[0]), RuntimeError("SECRET_ERROR"), "unused"])
    result = isolation.execute_isolated_verification(prepared, llm)
    assert not result.complete and not result.all_supported and result.attempted_calls == len(llm.calls) == 2
    assert result.halted_reason == "client_exception"
    assert result.outcomes[-1].status == "execution_error"
    assert "SECRET_ERROR" not in repr(result)


def test_deadline_wrapper_is_not_unwrapped_and_late_reply_never_accepted():
    prepared = plan()
    now = [0.0]
    deadline = MonotonicDeadline(10, clock=lambda: now[0])
    class LateClient:
        def complete(self, request):
            now[0] = 11
            return reply(prepared.requests[0])
    with pytest.raises(DeadlineExceeded):
        isolation.execute_isolated_verification(prepared, DeadlineBoundLLMClient(LateClient(), deadline))


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_process_interrupts_propagate(exception):
    prepared = plan()
    with pytest.raises(exception):
        isolation.execute_isolated_verification(prepared, Scripted([exception()]))


@pytest.mark.parametrize("damage", ["omit", "reorder", "request", "binding", "version", "input_hash",
                                    "index_bool", "temperature_bool", "source_json"])
def test_tampered_frozen_plans_fail_before_dispatch(damage):
    prepared = plan()
    if damage == "omit": prepared = replace(prepared, requests=prepared.requests[:-1])
    elif damage == "reorder": prepared = replace(prepared, requests=tuple(reversed(prepared.requests)))
    elif damage == "request": prepared = replace(prepared, requests=(replace(prepared.requests[0], request=LLMRequest("approve", "{}")),) + prepared.requests[1:])
    elif damage == "binding": prepared = replace(prepared, requests=(replace(prepared.requests[0], binding_sha256="0" * 64),) + prepared.requests[1:])
    elif damage == "version": prepared = replace(prepared, version="future")
    elif damage == "input_hash": prepared = replace(prepared, input_sha256="0" * 64)
    elif damage == "index_bool": prepared = replace(prepared, requests=(prepared.requests[0], replace(prepared.requests[1], index=True)) + prepared.requests[2:])
    elif damage == "temperature_bool": prepared = replace(prepared, requests=(prepared.requests[0], replace(prepared.requests[1], request=replace(prepared.requests[1].request, temperature=True))) + prepared.requests[2:])
    elif damage == "source_json": prepared = replace(prepared, source_payload_json=prepared.source_payload_json + " ")
    llm = Scripted([])
    with pytest.raises(ValueError):
        isolation.execute_isolated_verification(prepared, llm)
    assert llm.calls == []
