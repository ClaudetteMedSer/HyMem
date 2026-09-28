"""Offline LME treatment plumbing and identity checks; no provider calls."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys

import pytest

from benchmarks import lme_protocol as protocol
from benchmarks import lme_registry as registry
from benchmarks import longmemeval_adapter as lme
from benchmarks.strictness import (
    AtomicCheckpoint, BenchmarkIntegrityError, build_manifest, content_hash,
    effective_hymem_config_identity, freeze_calibration, load_calibration,
)
from hymem.dreaming.summary_policy import (
    LEGACY_COMPLETE_V1 as LEGACY, BOUNDED_HIGHLIGHTS_V1 as BOUNDED,
)
from tests.test_lme_protocol_hardening import (
    _refresh_manifest, _write_cli_fixture_dataset, make_artifact,
)


def cli_args(monkeypatch, extra=()):
    class Parsed(BaseException):
        pass

    original = argparse.ArgumentParser.parse_args
    captured = []

    def capture(parser, *args, **kwargs):
        value = original(parser, *args, **kwargs)
        captured.append(value)
        raise Parsed()

    with monkeypatch.context() as patch:
        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
        patch.setattr(sys, "argv", ["longmemeval_adapter.py", *extra])
        with pytest.raises(Parsed):
            lme._main()
    return captured[0]


def policy_config(name, *, tmp_path=Path("/benchmark-fixture")):
    adapter = lme.HyMemAdapter(tmp_path / "store.sqlite", digest_summary_policy=name)
    return {
        "label_free_answer_path": True,
        "digest_summary_policy": name,
        "effective_hymem_config": effective_hymem_config_identity(adapter.build_config()),
    }


def manifest(config):
    return build_manifest(benchmark="LongMemEval", code_sha256=content_hash("code"),
        data_sha256=content_hash("data"), config=config, models={}, seed=0,
        expected_ids=["q1", "q2"], protocol_split="full")


def artifact_with_policy(name):
    artifact = make_artifact()
    artifact["config"]["digest_summary_policy"] = name
    artifact["config"]["effective_hymem_config"]["digest_summary_policy"] = name
    _refresh_manifest(artifact)
    return artifact


@pytest.mark.parametrize("name", [LEGACY, BOUNDED])
def test_cli_exact_treatments_and_config_pinning(monkeypatch, tmp_path, name):
    args = cli_args(monkeypatch, ["--digest-summary-policy", name])
    adapter = lme._adapter_for_args(tmp_path / "store.sqlite", args, "")
    assert args.digest_summary_policy == adapter.digest_summary_policy == name
    assert adapter.build_config().digest_summary_policy == name
    assert policy_config(name)["effective_hymem_config"]["digest_summary_policy"] == name


def test_cli_and_old_manual_namespace_default_to_legacy(monkeypatch, tmp_path):
    args = cli_args(monkeypatch)
    assert args.digest_summary_policy == LEGACY
    del args.digest_summary_policy
    adapter = lme._adapter_for_args(tmp_path / "store.sqlite", args, "")
    assert adapter.digest_summary_policy == adapter.build_config().digest_summary_policy == LEGACY


def test_build_config_revalidates_policy_before_provider_construction(tmp_path):
    adapter = lme.HyMemAdapter(tmp_path / "store.sqlite")
    adapter.digest_summary_policy = None
    with pytest.raises(ValueError):
        adapter.build_config()


@pytest.mark.parametrize("bad", [None, True, False, 1, {}, [], "", "bounded", "BOUNDED_HIGHLIGHTS_V1",
                                "bounded_highlights_v2", " bounded_highlights_v1"])
def test_direct_constructor_and_explicit_manual_args_reject_unknown_policy(monkeypatch, tmp_path, bad):
    with pytest.raises(ValueError):
        lme.HyMemAdapter(tmp_path / "store.sqlite", digest_summary_policy=bad)
    args = cli_args(monkeypatch)
    args.digest_summary_policy = bad
    with pytest.raises(ValueError):
        lme._adapter_for_args(tmp_path / "store.sqlite", args, "")


@pytest.mark.parametrize("bad", ["none", "bounded", "bounded_highlights_v2", "legacy_complete_v1 "])
def test_cli_rejects_unknown_treatment_before_clients(monkeypatch, bad):
    monkeypatch.setattr(sys, "argv", ["longmemeval_adapter.py", "--digest-summary-policy", bad])
    with pytest.raises(SystemExit) as captured:
        lme._main()
    assert captured.value.code == 2


@pytest.mark.parametrize("name", [LEGACY, BOUNDED])
def test_open_uses_identical_pinned_config_without_live_clients(monkeypatch, tmp_path, name):
    import hymem
    from hymem.contrib import openai_client

    captured = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def close(self):
            pass

    class FakeStore:
        def __init__(self, cfg, **kwargs):
            captured.append(cfg)

        def close(self):
            pass

    monkeypatch.setattr(openai_client, "OpenAICompatibleClient", FakeClient)
    monkeypatch.setattr(hymem, "HyMem", FakeStore)
    adapter = lme.HyMemAdapter(tmp_path / "store.sqlite", digest_summary_policy=name)
    expected = adapter.build_config()
    adapter.open()
    try:
        assert captured == [expected]
        assert captured[0].digest_summary_policy == name
    finally:
        adapter.close()


@pytest.mark.parametrize("name", [LEGACY, BOUNDED])
def test_distillation_constructor_receives_same_treatment(monkeypatch, tmp_path, name):
    args = cli_args(monkeypatch, ["--digest-summary-policy", name])
    question = _write_cli_fixture_dataset(tmp_path)[0]
    seen = []

    class StopBeforeOpen(BaseException):
        pass

    def capture(*args, **kwargs):
        seen.append(kwargs)
        raise StopBeforeOpen()

    monkeypatch.setattr(lme, "HyMemAdapter", capture)
    with pytest.raises(StopBeforeOpen):
        lme._distill_run_one(question, args, None, None, "", check_gold_in_context=False)
    assert len(seen) == 1 and seen[0]["digest_summary_policy"] == name


@pytest.mark.parametrize("name", [LEGACY, BOUNDED])
def test_cli_calibration_records_explicit_effective_policy_without_clients(monkeypatch, tmp_path, name):
    _write_cli_fixture_dataset(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("provider construction forbidden")

    monkeypatch.setattr(lme, "LLMClient", forbidden)
    output = tmp_path / "calibration.json"
    monkeypatch.setattr(sys, "argv", ["longmemeval_adapter.py", "--data-dir", str(tmp_path),
        "--results-dir", str(tmp_path), "--sample", "0", "--no-prereg",
        "--digest-summary-policy", name, "--freeze-calibration", str(output)])
    lme._main()
    config = json.loads(output.read_text())["config"]
    assert config["digest_summary_policy"] == name
    assert config["effective_hymem_config"]["digest_summary_policy"] == name
    assert protocol.validate_lme_summary_policy_binding(config) == name


@pytest.mark.parametrize("name", [LEGACY, BOUNDED])
def test_protocol_accepts_matching_recorded_treatment_without_mutating_config(name):
    config = policy_config(name)
    before = deepcopy(config)
    assert protocol.validate_lme_summary_policy_binding(config) == name
    assert config == before


@pytest.mark.parametrize("where", ["declared", "effective"])
@pytest.mark.parametrize("bad", [None, True, 1, [], {}, "bounded", "legacy_complete_v2"])
def test_protocol_rejects_explicit_unknown_or_none_at_either_crosslink(where, bad):
    config = policy_config(LEGACY)
    target = config if where == "declared" else config["effective_hymem_config"]
    target["digest_summary_policy"] = bad
    with pytest.raises(BenchmarkIntegrityError, match="summary policy"):
        protocol.validate_lme_summary_policy_binding(config)


@pytest.mark.parametrize("where", ["declared", "effective"])
def test_protocol_rejects_one_sided_disclosure_and_opposing_treatments(where):
    config = policy_config(BOUNDED)
    target = config if where == "declared" else config["effective_hymem_config"]
    target["digest_summary_policy"] = LEGACY
    with pytest.raises(BenchmarkIntegrityError, match="summary policy differs"):
        protocol.validate_lme_summary_policy_binding(config)
    del target["digest_summary_policy"]
    with pytest.raises(BenchmarkIntegrityError, match="disclosure is incomplete"):
        protocol.validate_lme_summary_policy_binding(config)


def test_historical_missing_policy_means_only_legacy_without_inventing_disclosure():
    artifact = make_artifact()
    config = artifact["config"]
    before = deepcopy(config)
    assert "digest_summary_policy" not in config
    assert "digest_summary_policy" not in config["effective_hymem_config"]
    assert protocol.validate_lme_summary_policy_binding(config) == LEGACY
    protocol.validate_archived_artifact(artifact)
    assert config == before


def test_historical_interpretation_does_not_follow_future_runtime_default(monkeypatch):
    from hymem.dreaming import summary_policy

    monkeypatch.setattr(summary_policy, "DEFAULT_SUMMARY_POLICY", BOUNDED)
    artifact = make_artifact()
    assert protocol.validate_lme_summary_policy_binding(artifact["config"]) == LEGACY
    assert "digest_summary_policy" not in artifact["config"]


@pytest.mark.parametrize("reader", [protocol.validate_strict_artifact, protocol.validate_archived_artifact])
@pytest.mark.parametrize("name", [LEGACY, BOUNDED])
def test_strict_and_historical_artifact_readers_validate_recorded_treatment(reader, name):
    artifact = artifact_with_policy(name)
    assert reader(artifact)["counts"]["expected"] == 1
    artifact["config"]["effective_hymem_config"]["digest_summary_policy"] = (
        BOUNDED if name == LEGACY else LEGACY)
    _refresh_manifest(artifact)
    with pytest.raises(BenchmarkIntegrityError, match="summary policy differs"):
        reader(artifact)


def test_different_policy_changes_manifest_config_hash_and_run_id_without_forcing_noncomparability():
    left = manifest(policy_config(LEGACY))
    right = manifest(policy_config(BOUNDED))
    assert left["config_hash"] != right["config_hash"]
    assert left["run_id"] != right["run_id"]
    assert left["exploratory_non_comparable"] is right["exploratory_non_comparable"] is False


@pytest.mark.parametrize("original,other", [(LEGACY, BOUNDED), (BOUNDED, LEGACY)])
def test_checkpoint_and_calibration_cannot_be_reused_between_policies(tmp_path, original, other):
    config, changed = policy_config(original), policy_config(other)
    checkpoint = tmp_path / "checkpoint.json"
    owned = AtomicCheckpoint(checkpoint, manifest=manifest(config), expected_ids=["q1", "q2"])
    owned.close()
    with pytest.raises(BenchmarkIntegrityError, match="identity|manifest"):
        AtomicCheckpoint(checkpoint, manifest=manifest(changed), expected_ids=["q1", "q2"], resume=True)
    owned = AtomicCheckpoint(checkpoint, manifest=manifest(config), expected_ids=["q1", "q2"], resume=True)
    owned.close()
    receipt = tmp_path / "calibration.json"
    freeze_calibration(receipt, benchmark="LongMemEval", dataset_hash=content_hash("data"),
        ids=["q1", "q2"], config=config, models={}, seed=0)
    with pytest.raises(BenchmarkIntegrityError, match="config_hash mismatch"):
        load_calibration(receipt, benchmark="LongMemEval", dataset_hash=content_hash("data"),
            ids=["q1", "q2"], config=changed, models={})
    assert load_calibration(receipt, benchmark="LongMemEval", dataset_hash=content_hash("data"),
        ids=["q1", "q2"], config=config, models={})["config"]["digest_summary_policy"] == original


def test_registry_migration_preserves_unrecorded_old_rows_as_null(monkeypatch, tmp_path):
    path = tmp_path / "registry.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(registry.SCHEMA.replace("    digest_summary_policy      TEXT,\n", ""))
    connection.execute("INSERT INTO runs (archive, run_date) VALUES (?, ?)", ("old.json", "2020-01-01"))
    connection.commit()
    connection.close()
    monkeypatch.setattr(registry, "DB", path)
    connection = registry.connect()
    try:
        assert connection.execute("SELECT digest_summary_policy FROM runs").fetchall() == [(None,)]
    finally:
        connection.close()


@pytest.mark.parametrize("name", [None, LEGACY, BOUNDED])
def test_registry_records_explicit_treatment_and_shows_it_without_guessing(monkeypatch, tmp_path, capsys, name):
    artifact = make_artifact() if name is None else artifact_with_policy(name)
    source = tmp_path / "longmemeval-v2-hymem-20260918T120000Z.json"
    source.write_text(json.dumps(artifact))
    monkeypatch.setattr(registry, "DB", tmp_path / "registry.sqlite")
    connection = registry.connect()
    try:
        assert registry.ingest_file(connection, source, validation_scope="historical") == "inserted"
        connection.commit()
        row = connection.execute("SELECT digest_summary_policy, extras FROM runs").fetchone()
        assert row[0] == name
        cfg = json.loads(row[1])["config"]
        assert ("digest_summary_policy" in cfg) is (name is not None)
        if name is not None:
            assert cfg["digest_summary_policy"] == name
    finally:
        connection.close()
    registry.cmd_list()
    assert "digest_summary_policy" in capsys.readouterr().out
