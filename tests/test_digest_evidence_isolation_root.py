"""Independent offline tests; invented evidence is not model-quality evidence."""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import socket

import pytest

from benchmarks import digest_evidence_isolation as isolation
from hymem.deadline import DeadlineBoundLLMClient, DeadlineExceeded, MonotonicDeadline, current_deadline
from hymem.dreaming import digest
from hymem.dreaming.lossless import CoveredMessage
from hymem.extraction.llm import LLMRequest


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("independent isolation tests must not use the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)


def packet():
    prefix = "CONTEXT_ONLY previous sentence. I planned to "
    messages = [
        CoveredMessage(101, "root", "user", prefix + 'visit Ås, not Bergen; "gate" stayed closed.\n', "src-a",
                       source_peer_id="peer-a", source_workspace_id="workspace-a"),
        CoveredMessage(102, "root", "assistant", "SOURCE_B_ONLY Run gate inspect, then gate verify.", "src-b"),
        CoveredMessage(103, "root", "user", "UNCITED_ITEM_SOURCE_ONLY Earlier travel was cancelled.", "src-c"),
    ]
    episodes = [
        {"title": "A planned visit", "summary": "I planned to visit Ås.", "outcome": "deferred",
         "key_entities": ["Ås"], "chunk_ids": ["src-a"]},
        {"title": "OTHER_ITEM_ONLY gate check", "summary": "The assistant supplied gate inspection steps.",
         "outcome": "informational", "key_entities": ["gate"], "chunk_ids": ["src-b"]},
    ]
    procedure = {"name": "Inspect gate", "description": "Check gate status.",
                 "steps": [{"order": 1, "action": "Run gate inspect", "tool": "gate"},
                           {"order": 2, "action": "Run gate verify", "tool": "gate"}],
                 "triggers": ["gate check"], "entities_involved": ["gate"], "chunk_ids": ["src-a"]}
    return digest._digest_fidelity_payload(
        episodes, messages, ["src-a", "src-b", "src-c"],
        before_cursor=(None, 101, len(prefix)), after_cursor=(103, None, 0), leading_context=None,
        raw_procedures=[procedure, {**procedure, "chunk_ids": ["src-b"]}],
        raw_summary="RAW_GENERATED_ONLY discarded assembly",
        published_summary="EFFECTIVE_SUMMARY_ONLY The gate and travel topics continued.",
        prior_summary="PRIOR_ONLY_PERSON visited a lake.",
    )


TEMPLATE = LLMRequest(system="UNTRUSTED_TEMPLATE_SYSTEM", user="UNTRUSTED_TEMPLATE_USER",
                      max_tokens=1777, temperature=0.25)


def prepare(payload=None, **kwargs):
    return isolation.prepare_isolated_verification(
        packet() if payload is None else payload, TEMPLATE, max_calls=kwargs.pop("max_calls", 5), **kwargs,
    )


def reply(scope, verdict="supported"):
    groups = {"episode": ("episode_titles", "episode_content"),
              "procedure": ("procedures",), "summary": ("summary_content",)}[scope.kind]
    return {key: [{"index": scope.index, "verdict": verdict}] for key in groups}


class Client:
    def __init__(self, plan, values=None):
        self.plan, self.values, self.requests = plan, values or {}, []

    def complete(self, request):
        index = len(self.requests)
        self.requests.append(request)
        assert request == self.plan.requests[index].request
        value = self.values.get(index, json.dumps(reply(self.plan.requests[index])))
        if isinstance(value, BaseException):
            raise value
        return value


def test_root_all_requests_are_scope_isolated_and_lossless():
    original = packet()
    plan = prepare(original)
    assert [(s.kind, s.index) for s in plan.requests] == [
        ("episode", 0), ("episode", 1), ("procedure", 0), ("procedure", 1), ("summary", 0),
    ]
    catalog = {row["chunk_id"]: row for row in original["source_catalog"]}
    for scope in plan.requests:
        request, data = scope.request, json.loads(scope.request.user)
        assert (request.max_tokens, request.temperature, request.response_format) == (1777, 0.25, "json")
        assert "UNTRUSTED_TEMPLATE" not in request.system + request.user
        if scope.kind == "summary":
            assert set(data) == {"schema", "source_catalog", "summary_item"}
            assert data["summary_item"] == {k: v for k, v in original["summary_item"].items()
                                               if k != "candidate_raw_summary"}
            refs = original["summary_item"]["new_source_ids"]
            assert "OTHER_ITEM_ONLY" not in request.user
        else:
            key = "items" if scope.kind == "episode" else "procedure_items"
            assert set(data) == {"schema", "source_catalog", key}
            item = original[key][scope.index]
            assert data[key] == [item]
            refs = item["cited_source_ids"]
            for excluded in ("PRIOR_ONLY_PERSON", "EFFECTIVE_SUMMARY_ONLY", "RAW_GENERATED_ONLY",
                             "UNCITED_ITEM_SOURCE_ONLY"):
                assert excluded not in request.user
        assert data["source_catalog"] == [catalog[r] for r in refs]
        assert "RAW_GENERATED_ONLY" not in request.user
    first = json.loads(plan.requests[0].request.user)["source_catalog"][0]
    assert first["interpretation_only_context"] == original["source_catalog"][0]["interpretation_only_context"]
    assert first["source_peer_id"] == "peer-a" and first["source_workspace_id"] == "workspace-a"
    assert "SOURCE_B_ONLY" not in plan.requests[0].request.user
    assert "OTHER_ITEM_ONLY" not in plan.requests[0].request.user


def test_root_duplicate_procedures_keep_separate_evidence_and_indices():
    plan = prepare()
    a, b = [json.loads(s.request.user) for s in plan.requests if s.kind == "procedure"]
    assert a["procedure_items"][0]["candidate"] == b["procedure_items"][0]["candidate"]
    assert a["source_catalog"][0]["chunk_id"] == "src-a"
    assert b["source_catalog"][0]["chunk_id"] == "src-b"
    assert plan.requests[2].binding_sha256 != plan.requests[3].binding_sha256


def test_root_nullable_outcome_and_empty_prior_are_not_invented():
    p = packet()
    p["items"][0]["candidate_outcome"] = None
    p["summary_item"].update(candidate_raw_summary="", candidate_summary="",
                              candidate_is_noop=True, prior_derived_summary="")
    plan = prepare(p)
    assert json.loads(plan.requests[0].request.user)["items"][0]["candidate_outcome"] is None
    assert json.loads(plan.requests[-1].request.user)["summary_item"]["candidate_summary"] == ""


def test_root_noop_still_needs_summary_check_without_items():
    p = packet()
    p["items"], p["procedure_items"] = [], []
    p["summary_item"].update(candidate_raw_summary="", candidate_summary="PRIOR_ONLY_PERSON visited a lake.",
                              candidate_is_noop=True)
    plan = prepare(p, max_calls=1)
    assert len(plan.requests) == 1 and plan.requests[0].kind == "summary"
    assert json.loads(plan.requests[0].request.user)["summary_item"]["candidate_is_noop"] is True
    client = Client(plan)
    result = isolation.execute_isolated_verification(plan, client)
    assert result.complete and result.all_supported and result.attempted_calls == 1
    assert not hasattr(result, "covered_message_id") and not hasattr(result, "source_sha256")


def test_root_input_mutations_cannot_change_prepared_request():
    p = packet()
    before = deepcopy(p)
    plan = prepare(p)
    assert p == before
    p["source_catalog"][0]["visible_content"] = "MUTATED_SOURCE"
    p["items"][0]["candidate_key_entities"].append("MUTATED_ENTITY")
    p["summary_item"]["prior_derived_summary"] = "MUTATED_PRIOR"
    assert all("MUTATED_" not in s.request.user for s in plan.requests)
    assert plan.plan_sha256 == prepare(before).plan_sha256
    with pytest.raises(FrozenInstanceError):
        plan.max_calls = 99
    with pytest.raises(FrozenInstanceError):
        plan.requests[0].request.user = "{}"


@pytest.mark.parametrize("change", [
    lambda p: p.update(schema="digest-fidelity-decisions-v8"),
    lambda p: p.update(unrecognized="not silently ignored"),
    lambda p: p["source_catalog"].append(deepcopy(p["source_catalog"][0])),
    lambda p: p["source_catalog"][1].update(message_id=101),
    lambda p: p["source_catalog"][0].update(start=True),
    lambda p: p["source_catalog"][0].update(end=-1),
    lambda p: p["source_catalog"][0].update(visible_content=None),
    lambda p: p["source_catalog"][0].update(unrecognized="hidden evidence"),
    lambda p: p["items"][0].update(index=True),
    lambda p: p["items"][1].update(index=0),
    lambda p: p["items"][0].update(cited_source_ids=[]),
    lambda p: p["items"][0].update(cited_source_ids=["missing"]),
    lambda p: p["items"][0].update(cited_source_ids=["src-a", "src-a"]),
    lambda p: p["items"][0].update(cited_source_ids=["src-b", "src-a"]),
    lambda p: p["items"][0].update(candidate_outcome="imagined"),
    lambda p: p["items"][0].update(candidate_outcome={}),
    lambda p: p["items"][0].update(prior_derived_summary="leak"),
    lambda p: p["procedure_items"][0].update(cited_source_ids=["missing"]),
    lambda p: p["summary_item"].update(new_source_ids=["missing"]),
    lambda p: p["summary_item"].update(new_source_ids=["src-a", "src-a"]),
    lambda p: p["summary_item"].update(new_source_ids=["src-a"]),
    lambda p: p["summary_item"].update(new_source_ids=["src-c", "src-b", "src-a"]),
    lambda p: p["summary_item"].update(candidate_is_noop=1),
    lambda p: p["summary_item"].update(candidate_summary=None),
    lambda p: p["source_catalog"][0]["interpretation_only_context"].update(message_id=999),
])
def test_root_bad_input_fails_before_plan_or_call(change):
    p = packet()
    change(p)
    with pytest.raises(ValueError):
        prepare(p)


@pytest.mark.parametrize("cap", [0, 4, True, 5.0, -1])
def test_root_complete_call_reservation_is_required(cap):
    with pytest.raises(ValueError):
        prepare(max_calls=cap)


def test_root_complete_prompt_limit_includes_system_and_no_truncation():
    plan = prepare()
    maximum = max(len(s.request.system) + len(s.request.user) for s in plan.requests)
    assert prepare(max_input_chars=maximum).requests == plan.requests
    with pytest.raises(ValueError):
        prepare(max_input_chars=maximum - 1)


@pytest.mark.parametrize("change", [
    lambda p: replace(p, version="forged-version"),
    lambda p: replace(p, max_calls=1),
    lambda p: replace(p, plan_sha256="0" * 64),
    lambda p: replace(p, requests=p.requests[:-1]),
    lambda p: replace(p, requests=tuple(reversed(p.requests))),
    lambda p: replace(p, requests=(replace(p.requests[0], index=7), *p.requests[1:])),
    lambda p: replace(p, requests=(replace(p.requests[0], request=replace(p.requests[0].request, user="{}")), *p.requests[1:])),
    lambda p: replace(p, requests=(replace(p.requests[0], request=replace(p.requests[0].request, temperature=0.75)), *p.requests[1:])),
])
def test_root_forged_plan_rejected_before_any_call(change):
    plan = prepare()
    client = Client(plan)
    with pytest.raises(ValueError):
        isolation.execute_isolated_verification(change(plan), client)
    assert not client.requests


@pytest.mark.parametrize("field,value", [("temperature", float("nan")), ("temperature", True),
                                          ("max_tokens", 0), ("max_tokens", True),
                                          ("response_format", "text")])
def test_root_invalid_template_rejected(field, value):
    with pytest.raises(ValueError):
        isolation.prepare_isolated_verification(packet(), replace(TEMPLATE, **{field: value}), max_calls=5)


@pytest.mark.parametrize("target", range(5))
@pytest.mark.parametrize("verdict", ["unsupported", "uncertain"])
def test_root_any_scope_veto_blocks_all_supported_but_is_not_censored(target, verdict):
    plan = prepare()
    client = Client(plan, {target: json.dumps(reply(plan.requests[target], verdict))})
    result = isolation.execute_isolated_verification(plan, client)
    assert result.complete and not result.all_supported and result.attempted_calls == 5
    assert result.outcomes[target].status == "semantic_veto"
    assert len(client.requests) == 5
    assert sum(o.status == "supported" for o in result.outcomes) == 4


@pytest.mark.parametrize("damage", ["missing", "extra", "wrong_index", "bool_index", "duplicate", "bad_verdict"])
def test_root_partial_or_ambiguous_verdict_cannot_approve(damage):
    plan = prepare()
    body = reply(plan.requests[0])
    if damage == "missing":
        del body["episode_content"]
    elif damage == "extra":
        body["summary_content"] = [{"index": 0, "verdict": "supported"}]
    elif damage == "wrong_index":
        body["episode_content"][0]["index"] = 1
    elif damage == "bool_index":
        body["episode_content"][0]["index"] = False
    elif damage == "duplicate":
        body["episode_content"].append(body["episode_content"][0])
    else:
        body["episode_content"][0]["verdict"] = "probably supported"
    result = isolation.execute_isolated_verification(plan, Client(plan, {0: json.dumps(body)}))
    assert result.complete and not result.all_supported
    assert result.outcomes[0].status == "malformed_reply"


@pytest.mark.parametrize("raw", [None, "not JSON", "{}", '"supported"',
                                   '{"episode_titles":[],"episode_titles":[],"episode_content":[]}',
                                   '{"episode_titles":[{"index":0,"verdict":"suppor'])
def test_root_bad_response_not_salvaged_as_support(raw):
    plan = prepare()
    result = isolation.execute_isolated_verification(plan, Client(plan, {0: raw}))
    assert result.outcomes[0].status == "malformed_reply"
    assert result.complete and not result.all_supported


def test_root_existing_closing_container_tolerance_is_preserved():
    plan = prepare()
    raw = json.dumps(reply(plan.requests[0]))
    client = Client(plan, {0: raw[:-1]})
    assert isolation.execute_isolated_verification(plan, client).all_supported


@pytest.mark.parametrize("group", ["episode_titles", "episode_content"])
def test_root_title_and_body_cannot_override_each_other(group):
    plan = prepare()
    value = reply(plan.requests[0])
    value[group][0]["verdict"] = "unsupported"
    result = isolation.execute_isolated_verification(plan, Client(plan, {0: json.dumps(value)}))
    assert result.complete and not result.all_supported
    assert result.outcomes[0].status == "semantic_veto"
    assert dict(result.outcomes[0].verdicts)[group] == "unsupported"
    assert set(dict(result.outcomes[0].verdicts).values()) == {"supported", "unsupported"}


def test_root_old_combined_reply_never_becomes_an_isolated_approval():
    plan = prepare()
    old = {
        "episode_titles": [{"index": 0, "verdict": "supported"}],
        "episode_content": [{"index": 0, "verdict": "supported"}],
        "procedures": [], "summary_content": [{"index": 0, "verdict": "supported"}],
    }
    result = isolation.execute_isolated_verification(plan, Client(plan, {0: json.dumps(old)}))
    assert result.outcomes[0].status == "malformed_reply" and not result.all_supported


def test_root_full_reply_size_is_capped_even_if_prefix_is_an_approval():
    plan = prepare(max_output_chars=20)
    result = isolation.execute_isolated_verification(plan, Client(plan))
    assert result.complete and not result.all_supported
    assert all(o.status == "malformed_reply" and o.reason == "output_cap" for o in result.outcomes)


def test_root_ordinary_exception_halts_without_text_leak_or_retry():
    plan = prepare()
    client = Client(plan, {1: RuntimeError("PRIVATE_EXCEPTION_MARKER")})
    result = isolation.execute_isolated_verification(plan, client)
    assert not result.complete and not result.all_supported and result.attempted_calls == 2
    assert len(client.requests) == 2 and result.outcomes[-1].status == "execution_error"
    assert "PRIVATE_EXCEPTION_MARKER" not in repr(result)


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(), DeadlineExceeded("expired")])
def test_root_interrupts_propagate_without_another_call(error):
    plan = prepare()
    client = Client(plan, {1: error})
    with pytest.raises(type(error)):
        isolation.execute_isolated_verification(plan, client)
    assert len(client.requests) == 2


@pytest.mark.parametrize("expires", [0.5, 2.5, 4.5])
def test_root_one_shared_deadline_rejects_late_success(expires):
    plan = prepare()
    clock = [0.0]
    deadline = MonotonicDeadline(expires, clock=lambda: clock[0])

    class TimedClient(Client):
        def complete(self, request):
            assert current_deadline() is deadline
            result = super().complete(request)
            clock[0] += 1.0
            return result

    client = TimedClient(plan)
    with pytest.raises(DeadlineExceeded):
        isolation.execute_isolated_verification(plan, DeadlineBoundLLMClient(client, deadline))
    assert len(client.requests) == int(expires) + 1


def test_root_plan_hash_binds_evidence_and_request_parameters():
    original = packet()
    before = prepare(original)
    changed = deepcopy(original)
    changed["items"][0]["candidate_title"] += "!"
    assert prepare(changed).plan_sha256 != before.plan_sha256
    changed = deepcopy(original)
    changed["summary_item"]["prior_derived_summary"] += "!"
    after = prepare(changed)
    assert after.plan_sha256 != before.plan_sha256
    assert after.requests[:4] == before.requests[:4]
    changed = replace(TEMPLATE, max_tokens=1778)
    assert isolation.prepare_isolated_verification(original, changed, max_calls=5).plan_sha256 != before.plan_sha256
