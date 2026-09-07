"""Exercise the physical store path through real MSC/LoCoMo callers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import locomo_adapter as locomo, msc_adapter as msc
from benchmarks.archive_evidence import validate_scoped_indexing
from benchmarks.strictness import BenchmarkIntegrityError
from hymem import HyMemConfig
from hymem.extraction.llm import StubLLMClient
from tests.test_locomo_checkpoint_resume import _conversation
from tests.test_msc_checkpoint_resume import _example


def _args(**overrides):
    return SimpleNamespace(**{
        "api_key": "offline-only", "sim": False, "no_dream": False,
        "dream_per_session": False, "hymem_model": msc._HYMEM_MODEL,
        "hymem_base_url": msc._DEEPSEEK_BASE_URL, "hymem_thinking": "auto",
        "embeddings": False, "rules_extraction": None, "graph_multihop": False,
        "facts": None, "facts_extraction": None, "indexing_max_cycles": 10,
        "indexing_timeout_s": 30.0, "top_k": 10, "dump_context": False,
        "dump_topk": False,
        "keep_db": False, "db_dir": None, "fresh": False,
        "message_fts_top_k": None, "rerank_top_k": None, "fts_top_k": None,
        "graph_top_k": None, "diag_only": False, "user_speaker": "a",
        "answerable_clause": False,
        "answer_model": msc._ANSWER_MODEL, "answer_base_url": msc._DEEPSEEK_BASE_URL,
        "judge_model": msc._JUDGE_MODEL, **overrides,
    })


@pytest.fixture
def offline_providers(monkeypatch):
    """Only provider boundaries are deterministic; all store work is real."""
    from hymem.contrib import openai_client
    import longmemeval_adapter as lme

    clients = []

    def memory_provider(**_kwargs):
        client = StubLLMClient(fixtures={
            "typed user-profile facts": '{"items":[]}',
            "You analyze one conversation session": (
                '{"episodes":[],"summary":"","procedures":[]}'
            ),
            "single pass": '{"triples":[],"markers":[],"complete":true}',
        }, default="[]")
        clients.append(client)
        return client

    monkeypatch.setattr(openai_client, "OpenAICompatibleClient", memory_provider)
    monkeypatch.setattr(lme, "answer_question", lambda *_a, **_k: "Two.")
    monkeypatch.setattr(lme, "judge_scored", lambda *_a, **_k: (True, "yes"))
    return clients


@pytest.fixture
def observed_temporary_stores(monkeypatch, tmp_path):
    # Observe the real mkdtemp call and place its real directory inside this
    # test's owned root. No adapter/indexing/receipt/cleanup operation is mocked.
    original = msc.tempfile.mkdtemp
    roots = []

    def tracked(*args, **kwargs):
        kwargs.setdefault("dir", str(tmp_path))
        result = original(*args, **kwargs)
        if kwargs.get("prefix", "").startswith(("msc_", "locomo_")):
            roots.append(Path(result))
        return result

    monkeypatch.setattr(msc.tempfile, "mkdtemp", tracked)
    return roots


@pytest.mark.parametrize("caller", ["recall", "recurrence"])
@pytest.mark.parametrize("keep", [False, True])
def test_real_msc_callers_publish_the_store_they_opened(
    offline_providers, observed_temporary_stores, caller, keep,
):
    args = _args(keep_db=keep)
    if caller == "recall":
        row = msc.run_recall(_example("q1"), args, None, None)
        assert row["correct"] is True
        indexing = row["indexing"]
    else:
        _, indexing = msc.run_recurrence_dump(_example("q1"), args)
    assert validate_scoped_indexing(indexing, scope_id="msc:q1", config=vars(args))
    assert indexing["store_build_receipt"]["status"] == "published"
    assert offline_providers and offline_providers[0].calls
    assert len(observed_temporary_stores) == 1
    root = observed_temporary_stores[0]
    assert root.exists() is keep
    if keep:
        db_path = HyMemConfig(root=root).db_path
        assert db_path.is_file() and db_path.name == "hymem.sqlite"
        assert not (root / "m.sqlite").exists()
        receipt = json.loads((root / msc.STORE_BUILD_RECEIPT_NAME).read_text())
        assert receipt["material_state"] == msc.compute_material_store_state(db_path)


@pytest.mark.parametrize("caller", ["recall", "recurrence"])
@pytest.mark.parametrize("sim", [False, True])
def test_real_skipped_msc_modes_keep_one_physical_database(
    offline_providers, observed_temporary_stores, caller, sim,
):
    args = _args(sim=sim, no_dream=True, keep_db=True)
    if caller == "recall":
        indexing = msc.run_recall(_example("q1"), args, None, None)["indexing"]
    else:
        _, indexing = msc.run_recurrence_dump(_example("q1"), args)
    root = observed_temporary_stores[0]
    assert HyMemConfig(root=root).db_path.is_file()
    assert not (root / "m.sqlite").exists()
    assert indexing["store_build_receipt"]["status"] == "not_published_non_comparable"
    assert not (root / msc.STORE_BUILD_RECEIPT_NAME).exists()


@pytest.mark.parametrize("existing", [False, True])
def test_adapter_rejects_a_different_requested_database_before_side_effects(
    tmp_path, monkeypatch, existing,
):
    from hymem.contrib import openai_client

    wrong = tmp_path / "custom.sqlite"
    if existing:
        wrong.write_bytes(b"caller-owned database must remain untouched")
    before = wrong.read_bytes() if existing else None
    calls = []
    monkeypatch.setattr(openai_client, "OpenAICompatibleClient", lambda **kw: calls.append(kw))
    adapter = msc.MSCAdapter(wrong, api_key="offline-only")
    with pytest.raises(BenchmarkIntegrityError, match="database path"):
        adapter.open()
    assert calls == [] and adapter.hy is None
    assert not HyMemConfig(root=tmp_path).db_path.exists()
    assert (wrong.read_bytes() if existing else None) == before
    adapter.close()


def test_material_attestation_rejects_a_post_open_path_swap(tmp_path):
    first = msc.MSCAdapter(tmp_path / "first" / "hymem.sqlite", sim=True).open()
    second = msc.MSCAdapter(tmp_path / "second" / "hymem.sqlite", sim=True).open()
    try:
        first.hy.log_message("first", "user", "first physical store")
        second.hy.log_message("second", "user", "different physical store")
        first.db_path = second.db_path
        with pytest.raises(BenchmarkIntegrityError, match="database path"):
            first.material_store_state()
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("caller", ["recall", "recurrence"])
def test_real_msc_callers_clean_up_after_provider_creation_failure(
    monkeypatch, observed_temporary_stores, caller,
):
    from hymem.contrib import openai_client

    def reject_provider(**_kwargs):
        raise RuntimeError("offline provider failure")

    monkeypatch.setattr(openai_client, "OpenAICompatibleClient", reject_provider)
    run = msc.run_recall if caller == "recall" else msc.run_recurrence_dump
    extra = (None, None) if caller == "recall" else ()
    with pytest.raises(RuntimeError, match="offline provider failure"):
        run(_example("q1"), _args(), *extra)
    assert len(observed_temporary_stores) == 1
    assert not observed_temporary_stores[0].exists()


def test_real_locomo_reuses_its_same_verified_physical_database(
    tmp_path, offline_providers,
):
    args = _args(db_dir=tmp_path / "persistent")
    conv = _conversation("conv-one", "q1")
    reader = SimpleNamespace(chat=lambda *_a, **_k: "Two.", extra_body={})
    first = locomo.evaluate_conversation(conv, args, reader, None)
    root = locomo.resolve_locomo_store_root(args.db_dir, conv["id"])
    db_path = HyMemConfig(root=root).db_path
    assert db_path.is_file()
    assert first[0]["indexing"]["store_build_receipt"]["status"] == "published"
    assert offline_providers[0].calls
    second = locomo.evaluate_conversation(conv, args, reader, None)
    assert second[0]["indexing"]["store_build_receipt"]["status"] == "validated"
    assert second[0]["correct"] is True
    assert offline_providers[1].calls == []
    assert db_path.is_file()
    assert first[0]["indexing"]["store_build_receipt"]["material_state_sha256"] == (
        second[0]["indexing"]["store_build_receipt"]["material_state_sha256"]
    )
