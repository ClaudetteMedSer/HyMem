"""F1 verifier wiring/fail-closed controls, not real-provider entailment proof."""
from contextlib import closing
from dataclasses import asdict, replace
import json
import re

import pytest

from hymem import HyMem
from hymem.deadline import DeadlineExceeded
from hymem.dreaming import digest, summary
from hymem.dreaming.lossless import CoveredMessage
from hymem.dreaming.semantic_generation import semantic_generation_suffix
from hymem.extraction.contract import extraction_contract_identity
from hymem.extraction.llm import StubLLMClient
from hymem.extraction.producer import canonical_module_sha256
from tests.digest_verification_fixtures import resolve_fidelity_sources, synthetic_format_approval
from tests.test_digest_summary_contract import _config, _extract, _seed


def _verdict(*values):
    return json.dumps({
        "episode_titles": [{"index": index, "verdict": value}
                           for index, value in enumerate(values)],
        "episode_content": [{"index": index, "verdict": "supported"}
                            for index in range(len(values))],
        "procedures": [],
        "summary_content": [{"index": 0, "verdict": "supported"}],
    })


class _Client:
    def __init__(self, verdict, *, titles=("Availability clarified",), body=None,
                 summary_text="The preview's availability was clarified."):
        self.verdict = verdict
        self.titles = titles
        self.body = body or "The preview is available through Cedar, not Birch."
        self.summary_text = summary_text
        self.calls = []
        self.primary = None

    def complete(self, request):
        self.calls.append(request)
        format_result = synthetic_format_approval(request)
        if format_result is not None:
            return format_result
        if request.system == digest._DIGEST_FIDELITY_SYSTEM:
            if isinstance(self.verdict, BaseException):
                raise self.verdict
            return self.verdict
        if request.system.startswith("You compact one rolling conversation summary"):
            return json.dumps({"summary": "The preview's availability was clarified."})
        assert request.system.startswith(("You analyze one conversation session",
                                          "You re-read one conversation session"))
        citations = re.findall(r"\[chunk ([^\]]+)\]", request.user)
        self.primary = {
            "episodes": [{
                "title": title, "summary": self.body, "outcome": "informational",
                "key_entities": ["Cedar", "Birch"], "chunk_ids": citations[:1],
            } for title in self.titles],
            "summary": self.summary_text, "procedures": [],
        }
        return json.dumps(self.primary)


@pytest.mark.parametrize("granular", [False, True])
@pytest.mark.parametrize("title,body,source,verdict", [
    ("Atlas is Cedar-only", "Atlas is available through Cedar, not Birch.",
     "Atlas is available through Cedar, not Birch.", "unsupported"),
    ("Cache always works", "Cache may work if enabled.",
     "Cache may work if enabled; offline use is unverified.", "unsupported"),
    ("Caching is enabled", "Caching is not enabled.",
     "Caching is not enabled.", "unsupported"),
    # A weaker generated body cannot veto an explicitly supported strong title.
    ("Atlas is Cedar-only", "Atlas is available through Cedar.",
     "Atlas is exclusively available through Cedar.", "supported"),
    ("Preview availability discussed", "The preview's availability is unknown.",
     "The preview's availability is unknown.", "uncertain"),
])
def test_title_gate_uses_source_and_keeps_strong_supported_titles(
    cfg, granular, title, body, source, verdict,
):
    client = _Client(_verdict(verdict), titles=(title,), body=body)
    with closing(HyMem(_config(cfg, granular), llm=client)) as hy:
        last_id = _seed(hy, source)
        result = _extract(hy, client, granular=granular, prior_summary="PRIOR_NOT_EVIDENCE")
        assert len(client.calls) == (3 if verdict == "supported" else 2)
        primary, verification = client.calls[:2]
        assert replace(verification, system=primary.system, user=primary.user) == primary
        payload = json.loads(verification.user)
        assert payload["schema"] == digest.DIGEST_FIDELITY_VERIFICATION_VERSION
        item = payload["items"][0]
        assert item["candidate_title"] == title and item["candidate_body"] == body
        assert resolve_fidelity_sources(payload, item["cited_source_ids"])[0]["visible_content"] == source
        assert "PRIOR_NOT_EVIDENCE" not in json.dumps(payload["items"])
        assert payload["summary_item"]["prior_derived_summary"] == "PRIOR_NOT_EVIDENCE"
        if verdict == "supported":
            assert not result.parse_failed and result.caught_up
            assert result.covered_message_id == last_id
            assert result.episodes.items == client.primary["episodes"]
        else:
            assert result.parse_failed and not result.caught_up
            assert result.failure_reason == "episode_title_" + verdict
            assert result.failure_stage == "fidelity_verification"
            assert result.source_sha256 is result.covered_message_id is result.summary is None
            assert result.episodes.items == result.procedures.items == []
            assert not digest.digest_failure_requires_input_shrink(result.failure_reason, result.failure_stage)
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0
        assert hy.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0] == 0
        assert tuple(hy.conn.execute(
            "SELECT digest_cursor_message_id,auto_summary FROM sessions WHERE id='bounded-summary'",
        ).fetchone()) == (None, None)


@pytest.mark.parametrize("summary_text,expected_calls", [("A valid rolling summary.", 2), ("long " * 110, 3)])
@pytest.mark.parametrize("second_verdict", ["supported", "unsupported", "uncertain"])
def test_single_batch_runs_after_final_assembly_and_never_rewrites_items(cfg, summary_text, expected_calls, second_verdict):
    client = _Client(_verdict("supported", second_verdict),
                     titles=("First title", "Second title"), summary_text=summary_text)
    with closing(HyMem(_config(cfg, True), llm=client)) as hy:
        _seed(hy)
        result = _extract(hy, client, granular=True)
        assert len(client.calls) == expected_calls + (second_verdict == "supported")
        assert sum(call.system == digest._DIGEST_FIDELITY_SYSTEM for call in client.calls) == 1
        if second_verdict == "supported":
            assert not result.parse_failed
            assert result.episodes.items == client.primary["episodes"]
        else:
            assert result.parse_failed and result.failure_stage == "fidelity_verification"
            assert result.episodes.items == [] and result.covered_message_id is None
            assert result.episode_input_items == result.episode_rejected_items == 2
        assert [item["index"] for item in json.loads(client.calls[-1].user)["items"]] == [0, 1]


@pytest.mark.parametrize("summary_text,expected_calls", [("A valid rolling summary.", 2), ("long " * 110, 3)])
def test_empty_episode_candidates_still_verify_final_summary(cfg, summary_text, expected_calls):
    client = _Client(_verdict(), titles=(), summary_text=summary_text)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy)
        result = _extract(hy, client, granular=False)
        assert not result.parse_failed and len(client.calls) == expected_calls + 1
        assert sum(call.system == digest._DIGEST_FIDELITY_SYSTEM for call in client.calls) == 1


@pytest.mark.parametrize("raw", ["not JSON", "[]", '{"episodes":[],"summary":42,"procedures":[]}'])
def test_failed_primary_never_reaches_verifier(cfg, raw):
    client = StubLLMClient(default=raw)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy)
        result = _extract(hy, client, granular=False)
        assert result.parse_failed and result.failure_stage == "primary"
        assert len(client.calls) == 1


_BAD_VERDICTS = [
    None, 1, [], "", "not JSON", "[]", "{}", '{"episode_titles":{}}',
    '{"episode_titles":[]}', '{"episode_titles":null}',
    '{"episode_titles":[{"index":0,"verdict":"supported"}],"extra":0}',
    '{"episode_titles":[{"index":0,"verdict":"supported","extra":0}]}',
    '{"episode_titles":[{"index":0,"verdict":"unsupported","verdict":"supported"}]}',
    '{"episode_titles":[],"episode_titles":[{"index":0,"verdict":"supported"}]}',
    '{"episode_titles":[{"index":0,"verdict":"supported"},{"index":0,"verdict":"supported"}]}',
    'prefix {"episode_titles":[{"index":0,"verdict":"supported"}]}',
    '{"episode_titles":[{"index":0,"verdict":"supported"}]} suffix',
] + [json.dumps({"episode_titles": [{"index": index, "verdict": verdict}]})
     for index, verdict in [(True, "supported"), (False, "supported"), (0.0, "supported"),
                            (-1, "supported"), (1, "supported"), ("0", "supported"),
                            (0, True), (0, []), (0, "SUPPORTED"), (0, "maybe")]]


@pytest.mark.parametrize("raw", _BAD_VERDICTS)
def test_verdict_schema_rejects_ambiguous_incomplete_or_nonexact_coverage(raw):
    assert digest._validate_digest_fidelity_response(raw, 1) is not None


def test_complete_reordered_verdicts_are_allowed_but_mixed_batch_is_never_salvaged():
    response = json.loads(_verdict("supported", "supported"))
    response["episode_titles"].reverse()
    raw = json.dumps(response, separators=(",", ":"))
    assert digest._validate_digest_fidelity_response(raw, 2) is None
    assert digest._validate_digest_fidelity_response("```json\n" + raw + "\n```", 2) is None
    assert digest._validate_digest_fidelity_response(_verdict("supported", "unsupported"), 2) == "episode_title_unsupported"
    assert digest._validate_digest_fidelity_response(_verdict("supported", "uncertain"), 2) == "episode_title_uncertain"
    assert digest._validate_digest_fidelity_response(raw.replace('"index":1', '"index":0'), 2) == "fidelity_shape_failure"


def test_per_item_evidence_uses_exact_offsets_and_separates_interpretation_context():
    prefix = "INDEPENDENT OLD FACT " + "x" * 60
    first = CoveredMessage(7, "session", "user", prefix + "NEW FIRST TAIL", "a", source_peer_id="p", source_workspace_id="w")
    second = CoveredMessage(8, "session", "assistant", "NEW SECOND HEAD UNSEEN SECRET SUFFIX", "b")
    uncited = CoveredMessage(9, "session", "tool", "UNCITED PRIVATE MESSAGE", "c")
    episodes = [{"title": "A", "summary": "Body A", "chunk_ids": ["a"],
                 "outcome": None, "key_entities": []},
                {"title": "B", "summary": "Body B", "chunk_ids": ["b"],
                 "outcome": None, "key_entities": []}]
    before = [asdict(row) for row in (first, second, uncited)]
    payload = digest._digest_fidelity_payload(
        episodes, [first, second, uncited], ["a", "b"],
        before_cursor=(6, 7, len(prefix)), after_cursor=(7, 8, 15), leading_context=None,
    )
    left, right = [resolve_fidelity_sources(payload, item["cited_source_ids"]) for item in payload["items"]]
    assert len(left) == len(right) == 1
    assert left[0]["visible_content"] == "NEW FIRST TAIL"
    assert left[0]["interpretation_only_context"]["content"] == prefix[-48:]
    assert left[0]["source_peer_id"] == "p" and left[0]["source_workspace_id"] == "w"
    assert right[0]["visible_content"] == second.content[:15]
    assert right[0]["interpretation_only_context"] is None
    rendered = digest._encode_digest_fidelity_payload(payload)
    assert "INDEPENDENT OLD FACT" not in rendered and "UNSEEN SECRET SUFFIX" not in rendered
    assert "UNCITED PRIVATE MESSAGE" not in rendered
    assert before == [asdict(row) for row in (first, second, uncited)]


def test_prior_message_context_retains_its_own_attribution_and_is_not_visible_evidence():
    previous = CoveredMessage(1, "session", "user", "Question boundary", "old", source_peer_id="old-peer")
    current = CoveredMessage(2, "session", "assistant", "Visible answer", "new", source_peer_id="new-peer")
    payload = digest._digest_fidelity_payload(
        [{"title": "Answer", "summary": "Candidate", "chunk_ids": ["new"],
          "outcome": None, "key_entities": []}],
        [current], ["new"], before_cursor=(1, None, 0), after_cursor=(2, None, 0), leading_context=previous,
    )
    source = resolve_fidelity_sources(payload, payload["items"][0]["cited_source_ids"])[0]
    assert source["visible_content"] == current.content and source["source_peer_id"] == "new-peer"
    assert source["interpretation_only_context"]["source_peer_id"] == "old-peer"
    assert source["interpretation_only_context"]["content"] == previous.content


@pytest.mark.parametrize("error", [RuntimeError("private transport error"), DeadlineExceeded("expired"), KeyboardInterrupt()])
def test_verification_errors_keep_stage_and_cancellation_identity(cfg, error):
    client = _Client(error)
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy)
        if isinstance(error, Exception):
            with pytest.raises(digest.DigestCompletionError) as caught:
                _extract(hy, client, granular=False)
            assert caught.value.failure_stage == "fidelity_verification"
            assert caught.value.__cause__ is error
            assert "private" not in str(caught.value)
        else:
            with pytest.raises(type(error)) as caught:
                _extract(hy, client, granular=False)
            assert caught.value is error
        assert len(client.calls) == 2
        assert hy.conn.execute("SELECT COUNT(*) FROM digest_staging").fetchone()[0] == 0


def test_resource_caps_fail_closed_without_truncating_or_spending_another_call(cfg, monkeypatch):
    client = _Client(_verdict("supported"))
    with closing(HyMem(_config(cfg, False), llm=client)) as hy:
        _seed(hy)
        monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", 1)
        result = _extract(hy, client, granular=False)
        assert result.failure_reason == "fidelity_input_cap" and len(client.calls) == 1
        assert result.failure_stage == "fidelity_verification" and result.covered_message_id is None
    maximum = digest._DIGEST_FIDELITY_MAX_OUTPUT_CHARS
    assert digest._validate_digest_fidelity_response(" " * (maximum + 1), 1) == "fidelity_output_cap"


def test_resource_bounds_are_exact_unicode_character_limits(monkeypatch):
    payload = {"content": "🧭\n\"literal\""}
    rendered = digest._encode_digest_fidelity_payload(payload)
    exact = len(digest._DIGEST_FIDELITY_SYSTEM) + len(rendered)
    monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", exact)
    assert digest._encode_digest_fidelity_payload(payload) == rendered
    monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", exact - 1)
    assert digest._encode_digest_fidelity_payload(payload) is None
    response = _verdict("supported")
    monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_OUTPUT_CHARS", len(response))
    assert digest._validate_digest_fidelity_response(response, 1) is None
    assert digest._validate_digest_fidelity_response(response + " ", 1) == "fidelity_output_cap"


@pytest.mark.parametrize("field", ["policy", "prompt", "parser", "input_cap"])
def test_verification_generation_is_bound_only_to_digest(monkeypatch, field):
    client = StubLLMClient()
    before = {tier: semantic_generation_suffix(tier, client) for tier in ("digest", "facts", "profile")}
    phase1 = extraction_contract_identity("v20")
    standalone = canonical_module_sha256(summary)
    if field == "policy":
        monkeypatch.setattr(digest, "DIGEST_FIDELITY_VERIFICATION_VERSION", "changed")
    elif field == "prompt":
        monkeypatch.setattr(digest, "_DIGEST_FIDELITY_SYSTEM", "changed")
    elif field == "input_cap":
        monkeypatch.setattr(digest, "_DIGEST_FIDELITY_MAX_INPUT_CHARS", 123)
    else:
        original = digest._validate_digest_fidelity_response
        monkeypatch.setattr(digest, "_validate_digest_fidelity_response", lambda raw, count: original(raw, count))
    after = {tier: semantic_generation_suffix(tier, client) for tier in before}
    assert after["digest"] != before["digest"]
    assert after["facts"] == before["facts"] and after["profile"] == before["profile"]
    assert extraction_contract_identity("v20") == phase1
    assert canonical_module_sha256(summary) == standalone
