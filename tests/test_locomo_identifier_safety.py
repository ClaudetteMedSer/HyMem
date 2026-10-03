"""LoCoMo dataset identifiers must never become unchecked store paths."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import locomo_adapter as locomo
from benchmarks.strictness import BenchmarkIntegrityError


def _raw_record(sample_id: object = "conv-26", *, qids: list[object] | None = None):
    questions = []
    for index, qid in enumerate(qids if qids is not None else [None]):
        question = {
            "question": f"Question {index}?",
            "answer": "answer",
            "category": 1,
            "evidence": ["D1:1"],
        }
        if qid is not None:
            question["question_id"] = qid
        questions.append(question)
    return {
        "sample_id": sample_id,
        "conversation": {
            "speaker_a": "Ada",
            "speaker_b": "Ben",
            "session_1_date_time": "1:00 pm on 8 May, 2023",
            "session_1": [
                {"speaker": "Ada", "dia_id": "D1:1", "text": "A memory."}
            ],
        },
        "qa": questions,
    }


def _load(tmp_path: Path, records: list[dict]):
    path = tmp_path / "locomo.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    return locomo.load_locomo_data(str(path))


@pytest.mark.parametrize(
    "unsafe_id",
    [
        "/tmp/outside",
        "../outside",
        "nested/child",
        r"nested\child",
        ".",
        "..",
        "",
        "   ",
        " leading",
        "trailing ",
        "control\x00id",
        "control\nline",
        "CON",
        "a" * 129,
        17,
    ],
)
def test_loader_rejects_unsafe_conversation_ids_without_echoing_them(
    tmp_path, unsafe_id
):
    with pytest.raises(BenchmarkIntegrityError) as raised:
        _load(tmp_path, [_raw_record(unsafe_id)])

    if isinstance(unsafe_id, str) and len(unsafe_id) > 4:
        assert unsafe_id not in str(raised.value)


def test_loader_rejects_duplicate_conversation_ids_before_grouping(tmp_path):
    with pytest.raises(BenchmarkIntegrityError, match="duplicate LoCoMo conversation"):
        _load(tmp_path, [_raw_record("same"), _raw_record("same")])


def test_loader_rejects_casefolded_filesystem_aliases(tmp_path):
    with pytest.raises(BenchmarkIntegrityError, match="filesystem alias"):
        _load(tmp_path, [_raw_record("Conv-26"), _raw_record("conv-26")])


def test_loader_preserves_safe_ids_and_explicit_question_ids(tmp_path):
    conversations = _load(
        tmp_path, [_raw_record("conv-26.v2", qids=["official-question-7"])]
    )

    assert conversations[0]["id"] == "conv-26.v2"
    assert conversations[0]["qa"][0]["question_id"] == "official-question-7"
    assert conversations[0]["qa"][0]["qa_id"] == "official-question-7"


@pytest.mark.parametrize("field", ["question_id", "qa_id"])
def test_loader_rejects_duplicate_explicit_question_ids(tmp_path, field):
    record = _raw_record("conv-26", qids=[])
    record["qa"] = [
        {
            "question": "First?", "answer": "one", "category": 1,
            field: "duplicate-question",
        },
        {
            "question": "Second?", "answer": "two", "category": 1,
            field: "duplicate-question",
        },
    ]

    with pytest.raises(BenchmarkIntegrityError, match="duplicate LoCoMo question"):
        _load(tmp_path, [record])


def test_loader_rejects_duplicate_question_ids_across_conversations(tmp_path):
    with pytest.raises(BenchmarkIntegrityError, match="duplicate LoCoMo question"):
        _load(
            tmp_path,
            [
                _raw_record("conv-a", qids=["shared-question"]),
                _raw_record("conv-b", qids=["shared-question"]),
            ],
        )


def test_loader_rejects_explicit_id_colliding_with_generated_id(tmp_path):
    record = _raw_record("conv-26", qids=[])
    record["qa"] = [
        {"question": "Generated?", "answer": "one", "category": 1},
        {
            "question": "Explicit?", "answer": "two", "category": 1,
            "question_id": "conv-26_q0",
        },
    ]

    with pytest.raises(BenchmarkIntegrityError, match="duplicate LoCoMo question"):
        _load(tmp_path, [record])


def _normalized_conversation(conversation_id: object, *question_ids: str) -> dict:
    return {
        "id": conversation_id,
        "n_sessions": 1,
        "qa": [
            {"qa_id": qid, "question_id": qid}
            for qid in (question_ids or ("question-1",))
        ],
    }


@pytest.mark.parametrize(
    "conversation_id,target_name",
    [
        ("../outside", "outside"),
        ("nested/child", "outside"),
        (".", "outside"),
        ("..", "outside"),
    ],
)
def test_fresh_rejects_unsafe_id_before_deleting_base_or_outside_sentinels(
    tmp_path, conversation_id, target_name
):
    base = tmp_path / "stores"
    base.mkdir()
    base_sentinel = base / "base-sentinel"
    base_sentinel.write_text("keep", encoding="utf-8")
    outside = tmp_path / target_name
    outside.mkdir(exist_ok=True)
    outside_sentinel = outside / "outside-sentinel"
    outside_sentinel.write_text("keep", encoding="utf-8")
    args = SimpleNamespace(db_dir=str(base), fresh=True)

    with pytest.raises(BenchmarkIntegrityError):
        locomo.evaluate_conversation(
            _normalized_conversation(conversation_id), args, None, None
        )

    assert base_sentinel.read_text(encoding="utf-8") == "keep"
    assert outside_sentinel.read_text(encoding="utf-8") == "keep"


def test_fresh_rejects_absolute_id_before_deleting_target(tmp_path):
    base = tmp_path / "stores"
    base.mkdir()
    target = tmp_path / "absolute-target"
    target.mkdir()
    sentinel = target / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(BenchmarkIntegrityError):
        locomo.evaluate_conversation(
            _normalized_conversation(str(target)),
            SimpleNamespace(db_dir=str(base), fresh=True),
            None,
            None,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_duplicate_questions_fail_before_fresh_store_mutation(tmp_path):
    base = tmp_path / "stores"
    root = base / "safe-conv"
    root.mkdir(parents=True)
    sentinel = root / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(BenchmarkIntegrityError, match="duplicate LoCoMo question"):
        locomo.evaluate_conversation(
            _normalized_conversation("safe-conv", "same", "same"),
            SimpleNamespace(db_dir=str(base), fresh=True),
            None,
            None,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_fresh_rejects_symlink_store_root_and_preserves_target(tmp_path):
    base = tmp_path / "stores"
    base.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    (base / "safe-conv").symlink_to(outside, target_is_directory=True)

    with pytest.raises(BenchmarkIntegrityError, match="symbolic link"):
        locomo.evaluate_conversation(
            _normalized_conversation("safe-conv"),
            SimpleNamespace(db_dir=str(base), fresh=True),
            None,
            None,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_fresh_rejects_symlink_database_before_store_mutation(tmp_path):
    base = tmp_path / "stores"
    root = base / "safe-conv"
    root.mkdir(parents=True)
    outside_database = tmp_path / "outside.sqlite"
    outside_database.write_text("keep", encoding="utf-8")
    (root / "hymem.sqlite").symlink_to(outside_database)

    with pytest.raises(BenchmarkIntegrityError, match="symbolic link"):
        locomo.evaluate_conversation(
            _normalized_conversation("safe-conv"),
            SimpleNamespace(db_dir=str(base), fresh=True),
            None,
            None,
        )

    assert outside_database.read_text(encoding="utf-8") == "keep"
    assert root.is_dir()


def test_safe_store_root_is_resolved_direct_child_and_existing_reuse_is_stable(
    tmp_path,
):
    base = tmp_path / "stores"
    root = base / "conv-26"
    root.mkdir(parents=True)
    (root / "hymem.sqlite").touch()

    resolved = locomo.resolve_locomo_store_root(base, "conv-26")

    assert resolved == root.resolve()
    assert resolved.parent == base.resolve()
    assert (resolved / "hymem.sqlite").exists()
