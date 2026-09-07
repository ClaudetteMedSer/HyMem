from __future__ import annotations

import json
from dataclasses import replace

import pytest

from benchmarks.extraction_canary import (
    extraction_canary_policy,
    skipped_extraction_canary,
    validate_extraction_canary_config_binding,
    validate_extraction_canary_report,
)
from benchmarks.strictness import BenchmarkIntegrityError
from hymem import HyMem
from hymem.core import db as core_db
from hymem.dreaming import phase1
from hymem.dreaming.chunks import Chunk, persist_chunks
from hymem.dreaming.lossless import materialize_message_coverage
from hymem.extraction import chunk as chunk_module
from hymem.extraction import contract
from hymem.extraction import prompts
from hymem.extraction import triples
from hymem.extraction.llm import StubLLMClient


def _complete_empty() -> str:
    return json.dumps({"triples": [], "markers": [], "complete": True})


def _claim_response(message_id: int, object_: str) -> str:
    return json.dumps({
        "triples": [{
            "subject": "database",
            "subject_type": "database",
            "predicate": "uses",
            "object": object_,
            "object_type": "technology",
            "polarity": 1,
            "source_message_id": message_id,
        }],
        "markers": [],
        "complete": True,
    })


def _extract_and_persist_claim(
    hy: HyMem, chunk: Chunk, *, object_: str,
) -> str:
    key = contract.extraction_cache_key(hy.config.prompt_version)
    llm = StubLLMClient(
        fixtures={"ALREADY ACCEPTED RESULT (": _complete_empty()},
        default=_claim_response(chunk.start_message_id, object_),
    )
    extraction = phase1.extract_chunk_results(
        hy.conn,
        chunk,
        llm,
        prompt_version=hy.config.prompt_version,
    )
    assert extraction is not None and extraction.failed is False
    with core_db.transaction(hy.conn):
        phase1.persist_chunk_results(
            hy.conn,
            chunk,
            extraction,
            prompt_version=hy.config.prompt_version,
            cfg=hy.config,
        )
    return key


def _seed_chunk(hy: HyMem) -> Chunk:
    hy.conn.execute("INSERT INTO sessions(id) VALUES ('contract-cache')")
    cursor = hy.conn.execute(
        "INSERT INTO messages(session_id,role,content) VALUES (?,?,?)",
        ("contract-cache", "user", "I prefer PostgreSQL."),
    )
    message_id = int(cursor.lastrowid)
    chunk = Chunk(
        id="contract-cache-chunk",
        session_id="contract-cache",
        start_message_id=message_id,
        end_message_id=message_id,
        salience_reason="long_user_turn",
        text="user: I prefer PostgreSQL.",
        source_message_ids=(message_id,),
    )
    with core_db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, chunk.session_id)
        persist_chunks(hy.conn, [chunk])
    return chunk


def test_contract_identity_is_deterministic_and_bound_into_config(cfg):
    first = contract.extraction_contract_identity(cfg.prompt_version)
    second = contract.extraction_contract_identity(cfg.prompt_version)

    assert first == second
    assert first.startswith("hymem-extraction-contract-sha256-v1:")
    assert cfg.extraction_contract == contract.extraction_contract_binding(
        cfg.prompt_version
    )
    assert "/Users/" not in json.dumps(cfg.extraction_contract)


def test_contract_identity_tracks_prompt_bytes_version_validator_and_recovery(
    monkeypatch,
):
    baseline = contract.extraction_contract_identity("v20")

    monkeypatch.setattr(
        prompts,
        "_CHUNK_EMPTY_VERIFICATION_SUFFIX",
        prompts._CHUNK_EMPTY_VERIFICATION_SUFFIX + " changed",
    )
    assert contract.extraction_contract_identity("v20") != baseline
    monkeypatch.undo()

    monkeypatch.setattr(
        chunk_module,
        "CHUNK_EXTRACTION_USER_TEMPLATE",
        chunk_module.CHUNK_EXTRACTION_USER_TEMPLATE + " changed",
    )
    assert contract.extraction_contract_identity("v20") != baseline
    monkeypatch.undo()

    assert contract.extraction_contract_identity("v21") != baseline

    monkeypatch.setattr(
        triples, "_VALID_TYPES", triples._VALID_TYPES | {"tampered_type"}
    )
    assert contract.extraction_contract_identity("v20") != baseline
    monkeypatch.undo()

    monkeypatch.setattr(
        triples,
        "ALLOWED_PREDICATES",
        (*triples.ALLOWED_PREDICATES, "tampered_predicate"),
    )
    assert contract.extraction_contract_identity("v20") != baseline
    monkeypatch.undo()

    monkeypatch.setattr(
        chunk_module,
        "CLEAN_EMPTY_RECOVERY_POLICY_VERSION",
        "tampered-recovery",
    )
    assert contract.extraction_contract_identity("v20") != baseline


def test_source_split_policy_changes_derived_cache_key_without_prompt_bump(
    monkeypatch,
):
    current = contract.extraction_cache_key("v20")
    assert chunk_module.SOURCE_RECORD_SPLIT_POLICY_VERSION == (
        "hymem-source-semantic-split-v10"
    )
    assert chunk_module.SOURCE_FRAGMENT_CONTEXT_VERSION == (
        "hymem-canonical-markdown-table-fragment-context-v2"
    )
    assert chunk_module.SOURCE_BOUNDARY_CONTEXT_VERSION == (
        "hymem-adjacent-prose-boundary-context-v1"
    )

    monkeypatch.setattr(
        chunk_module,
        "SOURCE_RECORD_SPLIT_POLICY_VERSION",
        "hymem-source-semantic-split-v8",
    )
    prior_policy = contract.extraction_cache_key("v20")

    assert prior_policy != current
    assert contract.ACTIVE_EXTRACTION_PROMPT_VERSION == "v20"

    monkeypatch.undo()
    monkeypatch.setattr(
        chunk_module,
        "SOURCE_BOUNDARY_CONTEXT_VERSION",
        "hymem-adjacent-prose-boundary-context-v0",
    )
    assert contract.extraction_cache_key("v20") != current


def test_json_ceiling_classifier_is_bound_into_cache_and_canary_policy(cfg):
    from benchmarks.extraction_canary import extraction_canary_policy
    from hymem.extraction import jsonio

    current = contract.extraction_cache_key(cfg.prompt_version)
    policy = extraction_canary_policy(prompt_version=cfg.prompt_version)

    assert contract._contract_components(cfg.prompt_version)["recovery_policy"][
        "json_ceiling_cut"
    ] == jsonio.JSON_CEILING_CUT_POLICY_VERSION
    assert policy["extraction_contract"] == contract.extraction_contract_binding(
        cfg.prompt_version
    )

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            jsonio,
            "JSON_CEILING_CUT_POLICY_VERSION",
            "hymem-json-ceiling-cut-grammar-v0",
        )
        assert contract.extraction_cache_key(cfg.prompt_version) != current
        assert extraction_canary_policy(
            prompt_version=cfg.prompt_version
        ) != policy


def test_executable_normalization_ignores_comments_docstrings_and_layout():
    first = '''\
"""module docs"""
def value(a: int) -> int:
    """function docs"""
    # comment
    return a + 1
'''
    formatting_only = '''\
"""different docs"""

def value( a:int )->int:
    # another comment
    return (a+1)
'''
    executable_change = formatting_only.replace("a+1", "a+2")

    assert contract._normalized_python_source(first) == (
        contract._normalized_python_source(formatting_only)
    )
    assert contract._normalized_python_source(first) != (
        contract._normalized_python_source(executable_change)
    )

    type_only_first = """\
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from package_a import TypeA
value = 1
"""
    type_only_second = type_only_first.replace(
        "from package_a import TypeA", "from package_b import TypeB"
    )
    assert contract._normalized_python_source(type_only_first) == (
        contract._normalized_python_source(type_only_second)
    )


def test_unrelated_prompt_family_does_not_invalidate_extraction_contract(
    monkeypatch,
):
    baseline = contract.extraction_contract_identity("v20")
    monkeypatch.setattr(
        prompts, "EPISODE_SYSTEM", prompts.EPISODE_SYSTEM + " unrelated change"
    )
    assert contract.extraction_contract_identity("v20") == baseline


def test_active_imported_chunk_helper_rebind_fails_closed(monkeypatch):
    monkeypatch.setattr(
        chunk_module, "normalize_combined_marker_item", lambda _item: ({}, []),
    )
    with pytest.raises(RuntimeError, match="helper integrity"):
        contract.extraction_cache_key("v20")

    first = '''\
CHUNK = "extract this"
EPISODE = "episode version one"
def build_chunk():
    return CHUNK
'''
    unrelated_change = first.replace(
        "episode version one", "episode version two"
    )
    extraction_change = first.replace("extract this", "extract that")
    roots = ("build_chunk",)
    assert contract._normalized_module_slice(first, roots) == (
        contract._normalized_module_slice(unrelated_change, roots)
    )
    assert contract._normalized_module_slice(first, roots) != (
        contract._normalized_module_slice(extraction_change, roots)
    )


def test_canary_policy_and_zero_work_report_bind_effective_config(cfg):
    policy = extraction_canary_policy(prompt_version=cfg.prompt_version)
    assert validate_extraction_canary_config_binding(policy, cfg) == policy
    report = skipped_extraction_canary(
        "no_dream", prompt_version=cfg.prompt_version
    )
    assert validate_extraction_canary_report(
        report,
        expected_mode="no_dream",
        expected_prompt_version=cfg.prompt_version,
    ) == report

    missing = dict(policy)
    missing.pop("extraction_contract")
    with pytest.raises(BenchmarkIntegrityError, match="contract|policy"):
        validate_extraction_canary_config_binding(missing, cfg)

    tampered = json.loads(json.dumps(report))
    tampered["extraction_contract"]["identity"] = (
        "hymem-extraction-contract-sha256-v1:" + "0" * 64
    )
    with pytest.raises(BenchmarkIntegrityError, match="policy"):
        validate_extraction_canary_report(
            tampered,
            expected_mode="no_dream",
            expected_prompt_version=cfg.prompt_version,
        )


def test_legacy_bare_prompt_cache_row_is_stale_and_current_write_is_namespaced(cfg):
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        hy.conn.execute(
            "INSERT INTO processed_chunks(chunk_id,prompt_version) VALUES (?,?)",
            (chunk.id, cfg.prompt_version),
        )
        llm = StubLLMClient(default=_complete_empty())

        extraction = phase1.extract_chunk_results(
            hy.conn, chunk, llm, prompt_version=cfg.prompt_version
        )

        assert extraction is not None
        assert len(llm.calls) == 2
        with core_db.transaction(hy.conn):
            phase1.persist_chunk_results(
                hy.conn,
                chunk,
                extraction,
                prompt_version=cfg.prompt_version,
                cfg=cfg,
            )
        keys = {
            row[0]
            for row in hy.conn.execute(
                "SELECT prompt_version FROM processed_chunks WHERE chunk_id=?",
                (chunk.id,),
            )
        }
        assert cfg.prompt_version in keys
        assert contract.extraction_cache_key(cfg.prompt_version) in keys
    finally:
        hy.close()


def test_contract_drift_misses_prior_namespaced_cache(monkeypatch, cfg):
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        old_key = contract.extraction_cache_key(cfg.prompt_version)
        hy.conn.execute(
            "INSERT INTO processed_chunks(chunk_id,prompt_version) VALUES (?,?)",
            (chunk.id, old_key),
        )
        monkeypatch.setattr(
            triples, "_VALID_TYPES", triples._VALID_TYPES | {"new_type"}
        )
        new_key = contract.extraction_cache_key(cfg.prompt_version)
        assert new_key != old_key

        llm = StubLLMClient(default=_complete_empty())
        assert phase1.extract_chunk_results(
            hy.conn, chunk, llm, prompt_version=cfg.prompt_version
        ) is not None
        assert len(llm.calls) == 2
        with pytest.raises(ValueError, match="another contract"):
            contract.extraction_cache_key(old_key)
    finally:
        hy.close()


def test_contract_drift_replaces_same_public_generation_authority(
    monkeypatch, cfg,
):
    hy = HyMem(cfg)
    try:
        chunk = _seed_chunk(hy)
        old_key = _extract_and_persist_claim(
            hy, chunk, object_="PostgreSQL"
        )

        monkeypatch.setattr(
            chunk_module,
            "CLEAN_EMPTY_RECOVERY_POLICY_VERSION",
            "contract-drift-for-test",
        )
        new_key = contract.extraction_cache_key(cfg.prompt_version)
        assert new_key != old_key

        _extract_and_persist_claim(hy, chunk, object_="SQLite")

        outcome = hy.conn.execute(
            "SELECT prompt_version FROM kg_claim_extraction_outcomes "
            "WHERE chunk_id=?",
            (chunk.id,),
        ).fetchone()
        assert outcome["prompt_version"] == new_key
        observation_versions = {
            row[0] for row in hy.conn.execute(
                "SELECT DISTINCT prompt_version FROM kg_claim_observations "
                "WHERE chunk_id=?",
                (chunk.id,),
            )
        }
        assert observation_versions == {new_key}
        current_evidence_versions = {
            row[0] for row in hy.conn.execute(
                "SELECT DISTINCT extraction_prompt_version FROM kg_evidence "
                "WHERE chunk_id=? AND is_current=1",
                (chunk.id,),
            )
        }
        assert current_evidence_versions == {new_key}
        processed_versions = {
            row[0] for row in hy.conn.execute(
                "SELECT prompt_version FROM processed_chunks WHERE chunk_id=?",
                (chunk.id,),
            )
        }
        assert processed_versions == {old_key, new_key}
    finally:
        hy.close()


def test_tampered_effective_config_fails_before_phase1_provider_call(cfg):
    broken = replace(
        cfg,
        profile_extraction_enabled=False,
        facts_extraction_enabled=False,
        aggregation_nodes_enabled=False,
    )
    broken.extraction_contract["identity"] = (
        "hymem-extraction-contract-sha256-v1:" + "0" * 64
    )
    llm = StubLLMClient(default=_complete_empty())
    hy = HyMem(broken, llm=llm)
    try:
        hy.log_message("config-drift", "user", "I prefer PostgreSQL.")
        hy.close_session("config-drift")
        with pytest.raises(ValueError, match="contract"):
            hy.dream()
        assert llm.calls == []
    finally:
        hy.close()


def test_extraction_contract_uses_frozen_loaded_identity_helper(monkeypatch):
    from hymem.extraction import producer

    before = contract.extraction_cache_key()

    def replaced_helper(*_args, **_kwargs):
        raise AssertionError("contract followed a replaced identity helper")

    monkeypatch.setattr(
        producer, "canonical_module_slice_sha256", replaced_helper,
    )
    assert contract.extraction_cache_key() == before
