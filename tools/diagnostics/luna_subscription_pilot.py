"""Private, experimental Luna v2 canary and one source-ordered LME-S question.

The frozen typed canary is unchanged. This does not produce a canonical R9
artifact or an official-model score.
Run only after the separately reviewed subscription isolation probe succeeds.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import traceback

TRANSPORT_SHA256 = "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491"
SOURCE_MAP_SHA256 = "35e796d51aa4dd0a947105721936b797d7b695aac1bb8c13f7227081b9d70e51"
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
SOURCE_FILE_COUNT = 508
MAX_FIRST_ITEM_BYTES = 16 * 1024 * 1024
HEX = re.compile(r"[0-9a-f]{64}\Z")


class PilotStop(RuntimeError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise PilotStop(code)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular_absolute(path: Path) -> bool:
    try:
        return path.is_absolute() and stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def verify_inventory(candidate: Path, stamp_path: Path, stamp_sha256: str, *,
                     expected_map_sha256: str = SOURCE_MAP_SHA256,
                     expected_file_count: int = SOURCE_FILE_COUNT) -> int:
    require(regular_absolute(stamp_path) and digest(stamp_path) == stamp_sha256,
            "inventory_stamp_invalid")
    stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
    require(type(stamp) is dict and stamp, "inventory_stamp_invalid")
    expected = stamp.get("source_sha256", stamp)
    require(type(expected) is dict and expected, "inventory_stamp_invalid")
    encoded = json.dumps(expected, sort_keys=True, separators=(",", ":")).encode()
    require(hashlib.sha256(encoded).hexdigest() == expected_map_sha256
            and len(expected) == expected_file_count, "inventory_pin_mismatch")
    actual = {}
    for path in candidate.rglob("*"):
        rel = path.relative_to(candidate)
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in rel.parts):
            continue
        if path.suffix in {".pyc", ".pyo"}:
            continue
        require(not path.is_symlink(), "inventory_symlink")
        if path.is_file():
            actual[rel.as_posix()] = digest(path)
        else:
            require(path.is_dir(), "inventory_special_file")
    require(actual == expected, "inventory_drift")
    require("hymem/extraction/chunk.py" in actual
            and "benchmarks/extraction_canary.py" in actual
            and "benchmarks/longmemeval_adapter.py" in actual,
            "inventory_incomplete")
    return len(actual)


def first_source_item(path: Path) -> dict:
    """Decode exactly the first top-level array item with bounded RAM."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as source:
        head = source.read(1)
        if head != "[":
            raise PilotStop("dataset_shape_invalid")
        buffer = ""
        while len(buffer.encode("utf-8")) <= MAX_FIRST_ITEM_BYTES:
            block = source.read(64 * 1024)
            if not block:
                raise PilotStop("first_item_unavailable")
            buffer += block
            stripped = buffer.lstrip()
            if not stripped:
                continue
            try:
                item, offset = decoder.raw_decode(stripped)
            except json.JSONDecodeError:
                continue
            tail = stripped[offset:].lstrip()
            while not tail:
                tail = source.read(1)
                if not tail:
                    break
                tail = tail.lstrip()
            require(tail.startswith((",", "]")), "dataset_shape_invalid")
            require(type(item) is dict, "dataset_shape_invalid")
            return item
    raise PilotStop("first_item_unavailable")


def _core_path_and_types(canary, responses, path):
    """Recount only contextual emissions with optional types left untouched."""
    from hymem.extraction.jsonio import loads_exact_or_fenced
    from hymem.extraction.triples import normalize_combined_triple_item

    repaired = dict(path)
    for claim in ("table", "prose"):
        for context in ("exact_context", "wrong_context"):
            repaired[f"{claim}_claim_{context}_emissions"] = 0
    type_slots = {
        f"{index}_{side}": {"present": False, "invalid": False,
                            "wrong": False}
        for index in range(len(canary._CANARY_EXPECTED_CLAIMS))
        for side in ("subject", "object")
    }
    core_fields = ("subject", "predicate", "object", "polarity",
                   "source_message_id")
    for request, response in responses:
        payloads, failures = canary._request_source_payloads(request)
        contexts = set()
        if failures == 0:
            for payload in payloads:
                content = payload.get("content")
                if not isinstance(content, str):
                    continue
                if (payload.get("source_message_id") ==
                        canary.EXTRACTION_CANARY_TABLE_SOURCE_MESSAGE_ID
                        and canary._TABLE_CLAIM_ROW in content
                        and canary._strict_equal(
                            payload.get("source_fragment_context"),
                            canary._EXPECTED_TABLE_CONTEXT)):
                    contexts.add(0)
                if (payload.get("source_message_id") ==
                        canary.EXTRACTION_CANARY_PROSE_SOURCE_MESSAGE_ID
                        and canary._PROSE_BOUNDARY_RIGHT in content
                        and canary._strict_equal(
                            payload.get("source_boundary_context"),
                            canary._EXPECTED_PROSE_BOUNDARY_CONTEXT)):
                    contexts.add(1)
        data = loads_exact_or_fenced(response)
        if not isinstance(data, dict) or not isinstance(data.get("triples"), list):
            continue
        seen = set()
        for raw in data["triples"]:
            if not isinstance(raw, dict):
                continue
            identity_item = {key: raw[key] for key in core_fields if key in raw}
            normalized_id, id_errors, _ = normalize_combined_triple_item(
                identity_item, require_source_message_id=True)
            if normalized_id is None or id_errors:
                continue
            typed_item = dict(identity_item)
            typed_item.update({key: raw[key] for key in ("subject_type", "object_type")
                               if key in raw})
            normalized_typed, typed_errors, _ = normalize_combined_triple_item(
                typed_item, require_source_message_id=True)
            core_item = {key: value for key, value in raw.items()
                         if key not in {"subject_type", "object_type"}}
            normalized_core, errors, _ = normalize_combined_triple_item(
                core_item, require_source_message_id=True)
            for index, expected in enumerate(canary._CANARY_EXPECTED_CLAIMS):
                expected_core = dict(zip(core_fields,
                    (expected[0], expected[2], expected[3], expected[5], expected[6])))
                if not all(key in normalized_id and canary._strict_equal(
                        normalized_id[key], value)
                           for key, value in expected_core.items()):
                    continue
                for side, expected_type in (("subject", expected[1]),
                                            ("object", expected[4])):
                    key = side + "_type"
                    slot = type_slots[f"{index}_{side}"]
                    if key in raw:
                        slot["present"] = True
                        if (normalized_typed is None or typed_errors
                                or key not in normalized_typed):
                            slot["invalid"] = True
                        elif normalized_typed[key] != expected_type:
                            slot["wrong"] = True
                if normalized_core is not None and not errors:
                    seen.add(index)
        for index in seen:
            claim = "table" if index == 0 else "prose"
            context = "exact_context" if index in contexts else "wrong_context"
            repaired[f"{claim}_claim_{context}_emissions"] += 1
    return repaired, type_slots


def experimental_canary(canary, chunk, client, *, evidence=None) -> dict:
    """Versioned core-claim gate over the frozen fixture and extraction path."""
    recording = canary._RecordingClient(client)
    starting_turns = client.observed_turns
    try:
        result = chunk.extract_chunk(
            recording, canary._CANARY_CONTENT,
            source_records=canary._source_records(),
            completion_call_limit=canary.EXTRACTION_CANARY_MAX_COMPLETION_CALLS,
        )
    finally:
        if evidence:
            evidence({"requests": [vars(item) for item in recording.requests],
                      "responses": [reply for _request, reply in recording.responses],
                      "provider_output_truncations":
                          recording.provider_output_truncations})
    expected = canary._CANARY_EXPECTED_CLAIMS
    policy = canary.extraction_canary_policy()
    execution_path = canary._request_execution_path(
        recording.requests, recording.responses)
    execution_path["provider_output_truncations"] = (
        recording.provider_output_truncations)
    core_path, type_slots = _core_path_and_types(
        canary, recording.responses, execution_path)
    path_valid = True
    try:
        canary._validate_execution_path(
            core_path, completion_calls=result.completion_calls,
            initial_leaves=result.initial_prepartition_leaves,
            passed=not result.failed, policy=policy)
    except Exception:
        path_valid = False
    types = result.entity_type_hints
    expected_types = {entity: entity_type
                      for subject, subject_type, _predicate, object_, object_type,
                          _polarity, _source_id in expected
                      for entity, entity_type in ((subject, subject_type),
                                                  (object_, object_type))}
    final_types_valid = all(
        entity in expected_types and observed == expected_types[entity]
        for entity, observed in types.items())
    matched = 0
    for subject, subject_type, predicate, object_, object_type, polarity, source_id in expected:
        matches = [triple for triple in result.triples
                   if (triple.subject, triple.predicate, triple.object,
                       triple.polarity, triple.source_message_id)
                   == (subject, predicate, object_, polarity, source_id)]
        if len(matches) != 1:
            continue
        triple = matches[0]
        if all(getattr(triple, field) is None
               for field in canary._CANARY_OPTIONAL_TRIPLE_FIELDS):
            matched += 1
    passed = (not result.failed and matched == len(expected)
              and len(result.triples) == len(expected) and not result.markers
              and result.duplicate_triples_collapsed == 0
              and not result.entity_property_hints
              and final_types_valid
              and not any(slot["invalid"] or slot["wrong"]
                          for slot in type_slots.values())
              and path_valid
              and result.completion_calls == len(recording.requests)
              and result.completion_calls == len(recording.responses)
              and policy["minimum_pass_completion_calls"] <= result.completion_calls
              <= policy["max_completion_calls"]
              and result.initial_prepartition_leaves == policy["expected_prepartition_leaves"]
              and client.observed_turns - starting_turns == result.completion_calls
              and client.usage_complete and client.observed_tokens is not None
              and sum(core_path[f"{claim}_claim_{context}_emissions"]
                      for claim in ("table", "prose")
                      for context in ("exact_context", "wrong_context"))
                  <= len(recording.responses)
              and set(types) <= set(expected_types))
    return {"schema": "luna-experimental-canary-v2", "passed": passed,
            "fixture_sha256": canary.EXTRACTION_CANARY_FIXTURE_SHA256,
            "matched_core_claims": matched,
            "expected_core_claims": len(expected),
            "type_fields_expected": len(type_slots),
            "type_fields_present": sum(slot["present"] for slot in type_slots.values()),
            "type_fields_absent": sum(not slot["present"] for slot in type_slots.values()),
            "type_fields_invalid": sum(slot["invalid"]
                                       for slot in type_slots.values()),
            "type_fields_wrong": sum(slot["wrong"]
                                     for slot in type_slots.values()),
            "final_type_hint_mismatch": not final_types_valid,
            "type_presence": {key: value["present"] for key, value in type_slots.items()},
            "completion_calls": result.completion_calls,
            "internal_http_attempts": None,
            "execution_path_valid": path_valid,
            "core_execution_path_exact": core_path == policy["normal_execution_path"],
            "initial_prepartition_leaves": result.initial_prepartition_leaves,
            "failure_code": None if passed else "canary_contract_failed"}


class ChatBridge:
    """Preserve frozen reader/judge text; one-user judge has empty system text."""
    def __init__(self, client, request_type):
        self.client = client
        self.request_type = request_type

    def chat(self, messages, *, temperature=0.0, max_tokens=1024):
        roles = [m.get("role") for m in messages]
        if roles == ["system", "user"]:
            system, user = messages[0]["content"], messages[1]["content"]
        elif roles == ["user"]:
            system, user = "", messages[0]["content"]
        else:
            raise PilotStop("chat_shape_invalid")
        return self.client.complete(self.request_type(
            system=system, user=user, response_format="text",
            temperature=temperature, max_tokens=max_tokens))


class SubscriptionMemoryClient:
    """Versioned experimental producer identity around one pinned transport."""
    def __init__(self, delegate, *, progress=None, deadline=None):
        self.delegate = delegate
        self.progress = progress
        self.deadline = deadline
        self.invocation_in_flight = False

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def complete(self, request):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise PilotStop("wall_limit")
        self.invocation_in_flight = True
        try:
            if self.progress:
                self.progress(self)
            return self.delegate.complete(request)
        finally:
            self.invocation_in_flight = False
            if self.progress:
                self.progress(self)

    def phase1_producer_declaration(self):
        from hymem.extraction.producer import Phase1ProducerDeclaration
        return Phase1ProducerDeclaration(
            client_id="luna-subscription-pilot-memory-v1",
            implementation="sha256:" + digest(Path(__file__)),
            model="gpt-6-luna", endpoint=None,
            effective_request={
                "transport_sha256": TRANSPORT_SHA256,
                "codex_cli_version": "0.158.0",
                "account_route": "chatgpt", "model": "gpt-6-luna",
                "isolation": "read-only-no-network-empty-environments",
                "thread_lifecycle": "fresh-ephemeral-per-request",
                "messages": ["system", "user"],
                "temperature_effective": None,
                "output_cap_effective": None,
                "json_mode_effective": None,
                "internal_http_attempts": None,
            },
            retry_policy={"pilot_rerolls": 0, "transport_retry": "none",
                          "provider_internal_retries": "unknown"},
        )

    def memory_producer_declaration(self):
        return self.phase1_producer_declaration()


def make_adapter_class(adapter_module, client):
    class ExperimentalAdapter(adapter_module.HyMemAdapter):
        def open(self):
            from hymem import HyMem
            cfg = self.build_config()
            self.pipeline_llm = client
            self.embedding_client = None
            try:
                self.hy = HyMem(cfg, llm=client, embedding_client=None)
                self._owned_resources.own(self.hy, label="experimental memory store")
                return self
            except BaseException as exc:
                self._owned_resources.close(primary_exception=exc)
                raise
    return ExperimentalAdapter


def indexing_health_snapshot(indexing: object) -> dict | None:
    if not isinstance(indexing, dict):
        return None
    final = indexing.get("final_status")
    final = final if isinstance(final, dict) else {}
    def total(field):
        value = final.get(field)
        if not isinstance(value, dict) or any(type(item) is not int or item < 0
                                                for item in value.values()):
            return None
        return sum(value.values())
    terminal = final.get("terminal_loss")
    coverage = final.get("coverage_integrity")
    return {
        "cycles": indexing.get("cycles") if type(indexing.get("cycles")) is int else None,
        "outcome": indexing.get("outcome") if indexing.get("outcome") in {
            "success", "success_with_summary_degradation", "failure"} else None,
        "healthy": indexing.get("healthy") if type(indexing.get("healthy")) is bool else None,
        "summary_healthy": indexing.get("summary_healthy")
            if type(indexing.get("summary_healthy")) is bool else None,
        "pending": total("pending"),
        "malformed": total("malformed"),
        "quarantined": total("quarantined"),
        "terminal_loss_chunks": terminal.get("chunks") if isinstance(terminal, dict) else None,
        "coverage_integrity_failures": coverage.get("failures") if isinstance(coverage, dict) else None,
    }


def run_pilot(*, transport, llm_request, canary, chunk, lme, protocol,
              binary: str, question: dict, store_root: Path,
              client_factory=None, progress=None, row_progress=None,
              canary_evidence=None,
              wall_deadline=None) -> tuple[dict, dict | None]:
    """One shared subscription budget across canary, indexing, reader, judge."""
    delegate = (client_factory() if client_factory else
              transport.CodexSubscriptionClient(binary))
    summary = {"schema": "luna-subscription-pilot-v2", "canary": None,
               "question_started": False, "question_completed": False,
               "benchmark_failure": None, "correct": None,
               "observed_turns": 0, "known_tokens": None,
               "usage_complete": False, "invocation_in_flight": False,
               "internal_http_attempts": None,
               "requested_vs_unsupported_controls": [],
               "judge_protocol": "legacy-custom",
               "official_model_score": False, "canonical_r9_artifact": False,
               "indexing_outcome": None, "cleanup_ok": None}
    row = None
    adapter = None
    def update_usage(active):
        summary["observed_turns"] = active.observed_turns
        summary["known_tokens"] = active.observed_tokens
        summary["usage_complete"] = active.usage_complete
        summary["invocation_in_flight"] = active.invocation_in_flight
        summary["requested_vs_unsupported_controls"] = active.requested_controls
        if progress:
            progress(summary)
    client = SubscriptionMemoryClient(
        delegate, progress=update_usage, deadline=wall_deadline)
    try:
        admission = client.preflight()
        require(admission.get("config_isolation_admitted") is True
                and admission.get("auth") == "chatgpt"
                and admission.get("model") == "gpt-6-luna"
                and admission.get("inference_enabled") is False,
                "preflight_rejected")
        delegate.inference_accepted = True
        summary["canary"] = experimental_canary(
            canary, chunk, client, evidence=canary_evidence)
        if progress:
            progress(summary)
        require(summary["canary"]["passed"], "canary_failed")
        require(client.usage_complete and client.observed_tokens is not None,
                "usage_incomplete")
        bridge = ChatBridge(client, llm_request)
        cls = make_adapter_class(lme, client)
        adapter = cls(store_root / "question.sqlite", embeddings=False,
                      aggregation_nodes=False, episode_granularity=False,
                      pipeline_model="gpt-6-luna")
        adapter.open()
        summary["question_started"] = True
        row = lme.evaluate_question(
            bridge, bridge, adapter, question, top_k=15,
            auto_ability=True, no_dream=False, permissive_default=True,
            distill=False, retrieval_only=False,
            max_input_tokens=lme.DEFAULT_MAX_INPUT_TOKENS,
            max_input_bytes=lme.DEFAULT_MAX_INPUT_BYTES,
            judge_protocol="legacy-custom", indexing_max_cycles=100,
            indexing_timeout_s=3600.0, indexing_require_healthy=True,
        )
        if row_progress:
            row_progress(row)
        indexing = row.get("indexing")
        require(type(indexing) is dict and indexing.get("outcome") == "success"
                and indexing.get("healthy") is True
                and indexing.get("summary_healthy") is True
                and protocol._validate_versioned_indexing(
                    indexing, require_healthy=True, allow_failure=False) is True,
                "indexing_unhealthy")
        summary["indexing_outcome"] = indexing.get("outcome")
        summary["indexing_health"] = indexing_health_snapshot(indexing)
        summary["benchmark_failure"] = row.get("benchmark_failure")
        summary["correct"] = row.get("correct")
        require(row.get("benchmark_failure") is None
                and row.get("judge_error") is False
                and row.get("judge_parse_valid") is True
                and type(row.get("correct")) is bool,
                "question_unscored")
        require(client.usage_complete and client.observed_tokens is not None,
                "usage_incomplete")
        require(wall_deadline is None or time.monotonic() < wall_deadline,
                "wall_limit")
        summary["question_completed"] = True
        if progress:
            progress(summary)
    finally:
        cleanup_error = False
        if adapter is not None:
            last_indexing = getattr(adapter, "last_indexing_summary", None)
            if isinstance(last_indexing, dict):
                summary["indexing_private"] = last_indexing
                summary["indexing_outcome"] = last_indexing.get("outcome")
                summary["indexing_health"] = indexing_health_snapshot(
                    last_indexing)
            try:
                adapter.close()
            except Exception:
                cleanup_error = True
        summary["cleanup_ok"] = not cleanup_error
        if cleanup_error:
            summary["question_completed"] = False
        summary["observed_turns"] = client.observed_turns
        summary["known_tokens"] = client.observed_tokens
        summary["usage_complete"] = client.usage_complete
        summary["invocation_in_flight"] = client.invocation_in_flight
        summary["requested_vs_unsupported_controls"] = client.requested_controls
        if progress:
            progress(summary)
        if cleanup_error and sys.exc_info()[1] is None:
            raise PilotStop("cleanup_failed")
    return summary, row


def _load_pinned(candidate: Path, transport_path: Path):
    sys.path.insert(0, str(candidate))
    from hymem.extraction.llm import LLMRequest
    from benchmarks import extraction_canary, longmemeval_adapter, lme_protocol
    from hymem.extraction import chunk
    spec = importlib.util.spec_from_file_location("pinned_luna_transport", transport_path)
    require(spec is not None and spec.loader is not None, "transport_import_failed")
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    return transport, LLMRequest, extraction_canary, chunk, longmemeval_adapter, lme_protocol


def control_counts(controls: object) -> list[dict]:
    """Public control telemetry without a per-turn log or prompt material."""
    if not isinstance(controls, list):
        return []
    allowed = ("temperature_requested", "temperature_effective",
               "max_tokens_requested", "max_tokens_effective",
               "response_format_requested", "response_format_effective")
    grouped: dict[str, dict] = {}
    for item in controls:
        if not isinstance(item, dict):
            continue
        value = {key: item.get(key) for key in allowed}
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
        grouped.setdefault(encoded, {"controls": value, "count": 0})["count"] += 1
    return [grouped[key] for key in sorted(grouped)]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("binary", "transport", "transport-sha256", "candidate", "inventory-stamp",
                 "inventory-sha256", "dataset", "dataset-sha256", "output-dir"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    started = time.monotonic()
    report = {"ok": False, "stop_code": None, "canary_passed": False,
              "question_completed": False, "internal_http_attempts": None}
    output = Path(args.output_dir)
    created_output = False
    try:
        binary, transport_path, candidate, inventory_stamp, dataset = map(
            Path, (args.binary, args.transport, args.candidate,
                   args.inventory_stamp, args.dataset))
        require(all(p.is_absolute() for p in
                    (binary, transport_path, candidate, inventory_stamp, dataset, output)),
                "input_invalid")
        require(all(regular_absolute(p) for p in
                    (binary, transport_path, inventory_stamp, dataset)), "input_invalid")
        require(candidate.is_dir() and not candidate.is_symlink(),
                "input_invalid")
        require(all(HEX.fullmatch(value) for value in
                    (args.transport_sha256, args.inventory_sha256,
                     args.dataset_sha256)), "input_invalid")
        require(args.transport_sha256 == TRANSPORT_SHA256
                and digest(transport_path) == TRANSPORT_SHA256, "transport_drift")
        require(args.dataset_sha256 == DATASET_SHA256, "dataset_pin_mismatch")
        count = verify_inventory(candidate, inventory_stamp, args.inventory_sha256)
        require(digest(dataset) == args.dataset_sha256, "dataset_drift")
        question = first_source_item(dataset)
        transport, request_type, canary, chunk, lme, protocol = _load_pinned(
            candidate, transport_path)
        question = protocol.validate_lme_dataset([question], scale="S")[0]
        require(not output.exists() and output.parent.is_dir(), "output_not_fresh")
        output.mkdir(mode=0o700)
        created_output = True
        private_progress = output / "private-progress.json"
        private_row = output / "private-row.json"
        private_canary = output / "private-canary-evidence.json"
        def save_progress(value):
            descriptor = os.open(private_progress, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True)
        def save_row(value):
            descriptor = os.open(private_row, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True)
        def save_canary(value):
            descriptor = os.open(private_canary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(value, stream, sort_keys=True)
        with (output / "private-run.log").open("w", encoding="utf-8") as log:
            os.chmod(output / "private-run.log", 0o600)
            with redirect_stdout(log), redirect_stderr(log):
                summary, row = run_pilot(
                    transport=transport, llm_request=request_type,
                    canary=canary, chunk=chunk, lme=lme, protocol=protocol,
                    binary=str(binary), question=question, store_root=output,
                    progress=save_progress, row_progress=save_row,
                    canary_evidence=save_canary,
                    wall_deadline=started + 90 * 60)
        (output / "private-result.json").write_text(
            json.dumps({"summary": summary, "row": row}, sort_keys=True),
            encoding="utf-8")
        os.chmod(output / "private-result.json", 0o600)
        report.update({"ok": True, "canary_passed": True,
                       "question_completed": True,
                       "correct": summary["correct"],
                       "indexing_outcome": summary["indexing_outcome"],
                       "observed_turns": summary["observed_turns"],
                       "known_tokens": summary["known_tokens"],
                       "usage_complete": summary["usage_complete"],
                       "invocation_in_flight": summary["invocation_in_flight"],
                       "indexing_health": summary.get("indexing_health"),
                       "source_files_verified": count,
                       "dataset_sha256": args.dataset_sha256,
                       "transport_sha256": TRANSPORT_SHA256,
                       "requested_vs_unsupported_controls": control_counts(
                           summary["requested_vs_unsupported_controls"])})
    except PilotStop as exc:
        report["stop_code"] = str(exc)
        if created_output:
            private_failure = output / "private-failure.txt"
            descriptor = os.open(private_failure, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                traceback.print_exception(exc, file=stream)
    except Exception as exc:
        report["stop_code"] = "pilot_failure"
        report["exception_type"] = (
            type(exc).__name__ if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,80}",
                                                type(exc).__name__) else "Exception")
        if created_output:
            private_failure = output / "private-failure.txt"
            descriptor = os.open(private_failure, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                traceback.print_exception(exc, file=stream)
    if created_output:
        progress_path = output / "private-progress.json"
        if progress_path.is_file():
            try:
                progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
                report["canary_passed"] = (progress_data.get("canary") or {}).get("passed") is True
                report["question_completed"] = progress_data.get("question_completed") is True
                report["observed_turns"] = progress_data.get("observed_turns")
                report["known_tokens"] = progress_data.get("known_tokens")
                report["usage_complete"] = progress_data.get("usage_complete")
                report["invocation_in_flight"] = progress_data.get(
                    "invocation_in_flight")
                if report["invocation_in_flight"] is True:
                    report["usage_complete"] = False
                report["indexing_outcome"] = progress_data.get("indexing_outcome")
                report["indexing_health"] = progress_data.get("indexing_health")
                report["requested_vs_unsupported_controls"] = control_counts(
                    progress_data.get("requested_vs_unsupported_controls"))
            except (OSError, ValueError, TypeError):
                report["stop_code"] = "private_progress_invalid"
                report["ok"] = False
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
