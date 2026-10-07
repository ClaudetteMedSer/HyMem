"""Opt-in, noncanonical four-question LongMemEval diagnostic.

The CLI supports offline preflight and a one-shot run only after a pinned
receipt and live systemd/cgroup containment proof. It does not launch a
service. No canonical LME artifact is created here, and a diagnostic
admission never repairs canonical health.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import threading
import time
from typing import Any, Callable


SCHEMA = "luna-lme-semantic-diagnostic-v1"
ACCEPTED_MAP_SHA256 = "9e3fe4e495585f63bb560b123fbc0edb6441396ec28ae3f8fa6bfb6ba5b8cfae"
ACCEPTED_FILES = 514
DATASET_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
DIAGNOSTIC_HELPER_SHA256 = "b07cdb4d26ad2f95ab236fe8af4a1665d19c376ffaf077e90a561ce500ac1181"
MAX_LIMITS = {"campaign": (8012, 48_160_000, 14_400),
              "question": (2000, 12_000_000, 12_600),
              "canary": (12, 160_000, 600)}
PINS = {
    "benchmarks/codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
    "benchmarks/codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
    "benchmarks/codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    "benchmarks/codex_subscription_warm_v3.py": "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d",
    "benchmarks/codex_subscription_staged_v1.py": "c483975d53ca3523708cb589a68a8b0523664312ea259057e48ee01e7dad6ca8",
    "tools/diagnostics/luna_subscription_pilot.py": "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0",
    "tools/diagnostics/luna_subscription_lme_warm_v2.py": "3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567",
}
CANDIDATE_PINS = {
    "hymem/extraction/chunk.py": "c644513152e2ddefbe0d0d18dce7d597d18b4ba78ce275dd04ad4b2133ee7b92",
    "benchmarks/extraction_canary.py": "861dd9562db848fa61d97f5108ba2407e06c879a5563a0a4ce388fd47d6032ee",
    "benchmarks/longmemeval_adapter.py": "19ef4b5fa030d14c0c4e7e2c843ac9058bb09d19e4a704dbb84114b084f70f31",
    "hymem/dreaming/digest.py": "f0635472740bd8f1e2ba3a37fb08745917db22f110bd6f2b0eff61158f7c2cb0",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular(path: Path) -> bool:
    try:
        return path.is_absolute() and stat.S_ISREG(path.lstat().st_mode)
    except OSError:
        return False


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("module_load_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _verify_import_origins(origins: tuple[tuple[Any, Path], ...]) -> None:
    if any(Path(getattr(module, "__file__", "")).resolve() != path.resolve()
           for module, path in origins):
        raise ValueError("import_origin_invalid")


def load_verified(root: Path, inventory: Path, inventory_sha256: str,
                  dataset: Path, binary: Path, binary_sha256: str,
                  helper_sha256: str) -> dict[str, Any]:
    """Verify the accepted candidate and exact helper/transport bytes before import."""
    if not root.is_absolute() or root.is_symlink():
        raise ValueError("root_invalid")
    root = root.resolve(strict=True)
    candidate, code = root / "candidate", root / "code"
    if (candidate.is_symlink() or code.is_symlink()
            or not candidate.is_dir() or not code.is_dir() or not _regular(inventory)
            or not _regular(dataset) or not _regular(binary)
            or re.fullmatch(r"[0-9a-f]{64}", binary_sha256) is None
            or _sha(binary) != binary_sha256
            or len(inventory_sha256) != 64 or _sha(inventory) != inventory_sha256
            or helper_sha256 != DIAGNOSTIC_HELPER_SHA256
            or _sha(dataset) != DATASET_SHA256):
        raise ValueError("input_identity_invalid")
    for relative, expected in PINS.items():
        path = code / relative
        if not _regular(path) or (expected and _sha(path) != expected):
            raise ValueError("transport_or_helper_drift")
    helper_path = code / "benchmarks/lme_diagnostic.py"
    if not _regular(helper_path) or _sha(helper_path) != helper_sha256:
        raise ValueError("diagnostic_helper_drift")
    for relative, expected in CANDIDATE_PINS.items():
        if not _regular(candidate / relative) or _sha(candidate / relative) != expected:
            raise ValueError("candidate_source_drift")
    for name, module in tuple(sys.modules.items()):
        if (name == "hymem" or name.startswith("hymem.")
                or name == "benchmarks" or name.startswith("benchmarks.")
                or name == "tools" or name.startswith("tools.")):
            path = getattr(module, "__file__", None)
            if path and not (Path(path).resolve().is_relative_to(candidate)
                             or Path(path).resolve().is_relative_to(code)):
                raise ValueError("cached_source_mismatch")
    for path in (code, candidate):
        if str(path) in sys.path:
            sys.path.remove(str(path))
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(candidate))
    from tools.diagnostics import luna_subscription_lme_warm_v2 as prior
    prior.old.verify_inventory(candidate, inventory, inventory_sha256,
        expected_map_sha256=ACCEPTED_MAP_SHA256, expected_file_count=ACCEPTED_FILES)
    from benchmarks import codex_subscription_staged_v1 as staged
    from benchmarks import extraction_canary as canary
    from benchmarks import longmemeval_adapter as lme
    from benchmarks import lme_protocol as protocol
    from benchmarks import strictness
    from hymem.extraction import chunk
    from hymem.extraction.llm import LLMRequest
    from hymem.dreaming.summary_state import classify_summary_state
    from hymem.extraction.contract import extraction_contract_identity
    diagnostic = _load_file("pinned_lme_diagnostic_for_run", helper_path)
    origins = (
        (prior, code / "tools/diagnostics/luna_subscription_lme_warm_v2.py"),
        (prior.old, code / "tools/diagnostics/luna_subscription_pilot.py"),
        (staged, code / "benchmarks/codex_subscription_staged_v1.py"),
        (staged.warm, code / "benchmarks/codex_subscription_warm_v3.py"),
        (staged.warm.concurrent, code / "benchmarks/codex_subscription_concurrent_v2.py"),
        (canary, candidate / "benchmarks/extraction_canary.py"),
        (lme, candidate / "benchmarks/longmemeval_adapter.py"),
        (protocol, candidate / "benchmarks/lme_protocol.py"),
        (strictness, candidate / "benchmarks/strictness.py"),
        (chunk, candidate / "hymem/extraction/chunk.py"),
        (sys.modules[LLMRequest.__module__], candidate / "hymem/extraction/llm.py"),
        (sys.modules[classify_summary_state.__module__],
            candidate / "hymem/dreaming/summary_state.py"),
        (sys.modules[extraction_contract_identity.__module__],
            candidate / "hymem/extraction/contract.py"),
        (diagnostic, helper_path),
    )
    _verify_import_origins(origins)
    expected_identity = ("hymem-extraction-contract-sha256-v1:"
        "94a1adfc028694e9b1868ec8712baff38d4d7a99a9d2e3547c66819711890f95")
    if (staged._ROOT != candidate or staged.warm.concurrent.SharedBudget is not staged.warm.SharedBudget
            or extraction_contract_identity() != expected_identity
            or Path(canary.__file__).resolve() != candidate / "benchmarks/extraction_canary.py"
            or Path(lme.__file__).resolve() != candidate / "benchmarks/longmemeval_adapter.py"):
        raise ValueError("candidate_binding_invalid")
    selection = prior.SelectedQuestions(dataset, 4, protocol)
    return dict(root=root, candidate=candidate, code=code, dataset=dataset,
        binary=binary, prior=prior, staged=staged, warm=staged.warm,
        canary=canary, lme=lme, protocol=protocol, strictness=strictness,
        chunk=chunk, request_type=LLMRequest, diagnostic=diagnostic,
        summary_classifier=classify_summary_state, questions=list(selection))


class RegistrationAlias:
    """Second transport joins exactly one previously registered question slot."""
    def __init__(self, budget: Any, key: str, limits: Any):
        self._budget, self._key, self._limits = budget, key, limits

    def register(self, key: str, limits: Any) -> None:
        snapshot = self._budget.snapshot()["questions"]
        if key != self._key or limits != self._limits or key not in snapshot:
            raise ValueError("shared_question_registration_invalid")

    def __getattr__(self, name: str) -> Any:
        return getattr(self._budget, name)


class DualClient:
    def __init__(self, ordinary: Any, staged: Any, budget: Any, key: str):
        self.ordinary, self.staged, self.budget, self.key = ordinary, staged, budget, key

    def complete(self, request: Any) -> str:
        return self.ordinary.complete(request)

    def complete_stage(self, request: Any, batch: Any, stage: str, recheck: bool) -> str:
        return self.staged.complete_stage(request, batch, stage, recheck)

    @property
    def observed_turns(self) -> int:
        return self.budget.snapshot()["questions"][self.key]["turns"]

    @property
    def observed_tokens(self) -> int:
        return self.budget.snapshot()["questions"][self.key]["known_tokens"]

    @property
    def usage_complete(self) -> bool:
        return self.budget.snapshot()["questions"][self.key]["usage_complete"]

    def close(self) -> None:
        errors = []
        for client in (self.staged, self.ordinary):
            try:
                client.close()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            self.budget.halt("cleanup_failure")
            raise RuntimeError("dual_client_cleanup_failure") from errors[0]


def make_dual(loaded: dict, budget: Any, key: str, limits: Any) -> DualClient:
    warm, staged = loaded["warm"], loaded["staged"]
    ordinary = warm.WarmSubscriptionClient(str(loaded["binary"]), budget, key, limits,
        max_requests=16, max_age_seconds=300)
    try:
        second = staged.StagedSubscriptionClient(str(loaded["binary"]),
            RegistrationAlias(budget, key, limits), key, limits,
            max_requests=16, max_age_seconds=300)
    except BaseException:
        ordinary.close()
        raise
    return DualClient(ordinary, second, budget, key)


class CanaryRecorder:
    """Capture the real dispatch path; raw payloads never enter public output."""
    def __init__(self, delegate: DualClient, staged_module: Any):
        self.delegate, self.staged_module = delegate, staged_module
        self.ordinary_requests: list[Any] = []
        self.ordinary_responses: list[tuple[Any, Any]] = []
        self.stage_records: list[dict[str, Any]] = []
        self.truncations = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def complete(self, request: Any) -> str:
        self.ordinary_requests.append(request)
        try:
            response = self.delegate.complete(request)
        except Exception as exc:
            if type(exc).__name__ == "LLMOutputTruncatedError":
                self.truncations += 1
            raise
        self.ordinary_responses.append((request, response))
        return response

    def complete_stage(self, request: Any, batch: Any, stage: str, recheck: bool) -> str:
        parser = self.staged_module.staged
        if stage == "original":
            parser.validate_original_request(request, batch)
            schema = parser.build_original_output_schema(batch)
            batch_hash = batch.batch_sha256
            prior_hash = None
            sources = batch.sources
        elif stage == "alternatives" and recheck is False:
            parser.validate_alternatives_request(request, batch)
            schema = parser.build_alternatives_output_schema(batch)
            batch_hash = batch.classification_batch.batch_sha256
            prior_hash = batch.original_response_sha256
            sources = batch.classification_batch.sources
        else:
            raise ValueError("canary_stage_invalid")
        record = {"stage": stage, "recheck": recheck, "batch_sha256": batch_hash,
            "prior_response_sha256": prior_hash,
            "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest(),
            "source_ids": [source.source_message_id for source in sources],
            "source_contents": [source.content for source in sources],
            "source_contexts": [[context.content for context in source.contexts]
                for source in sources],
            "request": vars(request), "response": None, "returned": False}
        self.stage_records.append(record)
        response = self.delegate.complete_stage(request, batch, stage, recheck)
        record["response"], record["returned"] = response, True
        controls = self.delegate.staged.requested_controls
        record["schema_sent"] = controls[-1].get("output_schema_sent") if controls else None
        record["schema_acknowledged"] = controls[-1].get("output_schema_acknowledged") if controls else None
        return response


def _canary_gold(canary: Any, result: Any) -> bool:
    expected = canary._CANARY_EXPECTED_CLAIMS
    if result.failed or len(result.triples) != len(expected) or result.markers or result.duplicate_triples_collapsed:
        return False
    if result.entity_property_hints:
        return False
    expected_hints = {entity: entity_type
        for subject, subject_type, _, obj, object_type, _, _ in expected
        for entity, entity_type in ((subject, subject_type), (obj, object_type))}
    if result.entity_type_hints != expected_hints:
        return False
    for subject, subject_type, predicate, obj, object_type, polarity, source_id in expected:
        matches = [triple for triple in result.triples if
            (triple.subject, triple.predicate, triple.object, triple.polarity, triple.source_message_id)
            == (subject, predicate, obj, polarity, source_id)]
        if len(matches) != 1 or any(getattr(matches[0], name) is not None
                for name in canary._CANARY_OPTIONAL_TRIPLE_FIELDS):
            return False
        if result.entity_type_hints.get(subject) != subject_type or result.entity_type_hints.get(obj) != object_type:
            return False
    return True


def _ordinary_source_contexts(canary: Any, requests: list[Any]) -> dict[tuple[int, str], tuple[str, ...]] | None:
    """Derive exact scoped context from the ordinary requests actually sent."""
    expected: dict[tuple[int, str], tuple[str, ...]] = {}
    for request in requests:
        payloads, failures = canary._request_source_payloads(request)
        if failures:
            return None
        for payload in payloads:
            sid, content = payload.get("source_message_id"), payload.get("content")
            if type(sid) is not int or type(content) is not str or not content:
                return None
            contexts = []
            for key in ("source_fragment_context", "source_boundary_context"):
                region = payload.get(key)
                if region is None:
                    continue
                if type(region) is not dict or type(region.get("content")) is not str:
                    return None
                contexts.append(region["content"])
                if key == "source_fragment_context" and "prelude_content" in region:
                    if type(region["prelude_content"]) is not str:
                        return None
                    contexts.append(region["prelude_content"])
            pair = (sid, content)
            values = tuple(contexts)
            if pair in expected and expected[pair] != values:
                return None
            expected[pair] = values
    return expected


def _stage_source_context_bound(records: list[dict], expected_content: dict[int, str],
                                fixture_ids: set[int],
                                ordinary_contexts: dict[tuple[int, str], tuple[str, ...]] | None) -> bool:
    """Stage sources must carry the exact context of their ordinary fragments."""
    if not records or ordinary_contexts is None:
        return False
    for record in records:
        ids = record.get("source_ids")
        contents = record.get("source_contents")
        contexts = record.get("source_contexts")
        if (not isinstance(ids, list) or not ids or not isinstance(contents, list)
                or not isinstance(contexts, list)
                or not (len(ids) == len(contents) == len(contexts))):
            return False
        for sid, content, regions in zip(ids, contents, contexts, strict=True):
            if (type(sid) is not int or sid not in fixture_ids
                    or type(content) is not str or not content
                    or content not in expected_content[sid]
                    or not isinstance(regions, list)
                    or (sid, content) not in ordinary_contexts
                    or tuple(regions) != ordinary_contexts[(sid, content)]
                    or any(type(text) is not str or not text
                           or text not in expected_content[sid]
                           for text in regions)):
                return False
    return True


def run_live_canary(loaded: dict, budget: Any, limits: Any, private_dir: Path) -> dict:
    """One actual frozen extraction, with independent structure and gold fields."""
    canary, chunk = loaded["canary"], loaded["chunk"]
    client = make_dual(loaded, budget, "canary", limits)
    recording = CanaryRecorder(client, loaded["staged"])
    result = None
    cleanup_ok = False
    try:
        result = chunk.extract_chunk(recording, canary._CANARY_CONTENT,
            source_records=canary._source_records(),
            completion_call_limit=canary.EXTRACTION_CANARY_MAX_COMPLETION_CALLS)
    finally:
        try:
            private_dir.mkdir(mode=0o700, exist_ok=True)
            loaded["prior"].atomic_private(private_dir / "private-canary-dispatch.json", {
                "ordinary_requests": [vars(item) for item in recording.ordinary_requests],
                "ordinary_responses": [reply for _, reply in recording.ordinary_responses],
                "staged": recording.stage_records})
        finally:
            client.close()
            cleanup_ok = True
    path = canary._request_execution_path(recording.ordinary_requests,
        recording.ordinary_responses)
    path["provider_output_truncations"] = recording.truncations
    canary._validate_execution_path(path,
        completion_calls=len(recording.ordinary_requests),
        initial_leaves=result.initial_prepartition_leaves, passed=False,
        policy=canary.extraction_canary_policy())
    fixture_ids = set(canary.EXTRACTION_CANARY_SOURCE_MESSAGE_IDS)
    exact_path = (path["source_message_ids_seen"] == sorted(fixture_ids)
        and path["source_record_parse_failures"] == 0
        and path["table_claim_requests"] >= 1
        and path["table_claim_requests"] == path["table_claim_exact_context_requests"]
        and path["prose_claim_requests"] >= 1
        and path["prose_claim_requests"] == path["prose_claim_exact_context_requests"]
        and path["table_claim_self_contained_requests"] == 0
        and path["prose_claim_self_contained_requests"] == 0
        and path["protected_control_split_boundaries"] == 0
        and path["list_control_probe_atomic"] and path["fenced_code_control_probe_atomic"])
    expected_content = dict(zip(canary.EXTRACTION_CANARY_SOURCE_MESSAGE_IDS,
        canary._CANARY_SOURCE_CONTENTS, strict=True))
    ordinary_contexts = _ordinary_source_contexts(canary, recording.ordinary_requests)
    stages_bound = _stage_source_context_bound(recording.stage_records,
        expected_content, fixture_ids, ordinary_contexts) and all(record["returned"]
        and record.get("schema_sent") is True
        and record.get("schema_acknowledged") is True
        for record in recording.stage_records)
    state = budget.snapshot()
    counts_ok = (result.completion_calls == len(recording.ordinary_requests) + len(recording.stage_records)
        and result.grounding_calls == len(recording.stage_records)
        and result.grounding_recheck_calls == sum(record["recheck"] for record in recording.stage_records)
        and state["questions"]["canary"]["turns"] == result.completion_calls
        and state["reserved"] == state["in_flight"] == 0
        and state["usage_complete"] and client.usage_complete)
    structural = bool(exact_path and stages_bound and counts_ok and cleanup_ok
        and result.initial_prepartition_leaves == canary.EXTRACTION_CANARY_EXPECTED_PREPARTITION_LEAVES)
    full_fixture_exercised = False
    if structural and not result.failed:
        try:
            canary._validate_execution_path(path,
                completion_calls=len(recording.ordinary_requests),
                initial_leaves=result.initial_prepartition_leaves, passed=True,
                policy=canary.extraction_canary_policy())
            full_fixture_exercised = True
        except Exception:
            pass
    semantic_failure = (result.failed and loaded["diagnostic"]._semantic_reason(
        result.failure_reason, json.dumps(list(result.failure_details))))
    return {"schema": "luna-lme-live-extraction-canary-v1",
        "fixture_sha256": canary.EXTRACTION_CANARY_FIXTURE_SHA256,
        "structural_valid": structural, "observed_contexts_bound": bool(exact_path and stages_bound),
        "full_fixture_exercised": full_fixture_exercised,
        "model_gold_match": _canary_gold(canary, result),
        "quality_failure_reason": result.failure_reason if result.failed else None,
        "semantic_failure_proved": bool(semantic_failure),
        "ordinary_calls": len(recording.ordinary_requests),
        "staged_calls": len(recording.stage_records),
        "completion_calls": result.completion_calls,
        "grounding_rechecks": result.grounding_recheck_calls,
        "known_tokens": state["questions"]["canary"]["known_tokens"],
        "usage_complete": state["usage_complete"], "client_cleanup_ok": cleanup_ok}


class AccountedClient:
    """Attribute every paid call to a pinned frozen call site, before dispatch."""
    def __init__(self, delegate: DualClient, candidate: Path):
        self.delegate, self.candidate = delegate, candidate
        self.counts: dict[str, dict[str, int]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def _stage(self, staged: bool) -> str:
        frame = sys._getframe(2)
        while frame is not None:
            path = Path(frame.f_code.co_filename)
            try:
                relative = path.relative_to(self.candidate).as_posix()
            except ValueError:
                relative = ""
            name = frame.f_code.co_name
            if relative == "hymem/extraction/chunk.py" and name == "grounding_call":
                return "grounding" if staged else "unclassified"
            if relative == "hymem/extraction/chunk.py" and name == "single_attempt":
                return "extraction" if not staged else "unclassified"
            if relative == "hymem/dreaming/digest.py" and name == "extract_session_digest":
                return "digest" if not staged else "unclassified"
            if relative == "benchmarks/longmemeval_adapter.py" and name == "answer_question_raw":
                return "reader" if not staged else "unclassified"
            if relative == "benchmarks/longmemeval_adapter.py" and name == "judge_answer_raw":
                return "judge" if not staged else "unclassified"
            frame = frame.f_back
        return "unclassified"

    def _call(self, label: str, invoke: Callable[[], str]) -> str:
        if label == "unclassified":
            self.delegate.budget.halt("stage_accounting_failure")
            raise RuntimeError("stage_accounting_failure")
        before = self.delegate.budget.snapshot()["questions"][self.delegate.key]
        returned = False
        try:
            value = invoke()
            returned = True
            return value
        finally:
            after = self.delegate.budget.snapshot()["questions"][self.delegate.key]
            slot = self.counts.setdefault(label, {"attempts": 0, "returned": 0,
                "turns": 0, "known_tokens": 0})
            slot["attempts"] += 1
            slot["returned"] += returned
            slot["turns"] += after["turns"] - before["turns"]
            slot["known_tokens"] += after["known_tokens"] - before["known_tokens"]

    def complete(self, request: Any) -> str:
        return self._call(self._stage(False), lambda: self.delegate.complete(request))

    def complete_stage(self, request: Any, batch: Any, stage: str, recheck: bool) -> str:
        label = self._stage(True)
        if label != "grounding" or stage not in {"original", "alternatives"}:
            self.delegate.budget.halt("stage_accounting_failure")
            raise RuntimeError("stage_accounting_failure")
        return self._call(f"grounding_{stage}_{'recheck' if recheck else 'initial'}",
            lambda: self.delegate.complete_stage(request, batch, stage, recheck))

    def reconcile(self) -> bool:
        state = self.delegate.budget.snapshot()["questions"][self.delegate.key]
        return (sum(item["turns"] for item in self.counts.values()) == state["turns"]
            and sum(item["known_tokens"] for item in self.counts.values()) == state["known_tokens"]
            and all(item["attempts"] >= item["turns"] >= item["returned"]
                    for item in self.counts.values()))


def _memory_client(loaded: dict, client: AccountedClient):
    """Stable candidate producer identity independent of question index."""
    transport_sha = PINS["benchmarks/codex_subscription_staged_v1.py"]

    class MemoryClient:
        def __getattr__(self, name):
            return getattr(client, name)

        def complete(self, request):
            return client.complete(request)

        def complete_stage(self, request, batch, stage, recheck):
            return client.complete_stage(request, batch, stage, recheck)

        def phase1_producer_declaration(self):
            from hymem.extraction.producer import Phase1ProducerDeclaration
            return Phase1ProducerDeclaration(
                client_id="luna-lme-staged-diagnostic-v1",
                implementation="sha256:" + _sha(Path(__file__)), model="gpt-6-luna",
                endpoint=None,
                effective_request={"transport_sha256": transport_sha,
                    "ordinary_transport_sha256": PINS["benchmarks/codex_subscription_warm_v3.py"],
                    "account_route": "chatgpt", "model": "gpt-6-luna",
                    "isolation": "read-only-no-network-empty-environments",
                    "thread_lifecycle": "warm-rotated-ephemeral",
                    "messages": ["system", "user"],
                    "temperature_effective": None, "output_cap_effective": None,
                    "json_mode_effective": None, "internal_http_attempts": None},
                retry_policy={"pilot_rerolls": 0, "transport_retry": "none",
                              "provider_internal_retries": "unknown"})

        def memory_producer_declaration(self):
            return self.phase1_producer_declaration()

    return MemoryClient()


def validate_diagnostic_row(loaded: dict, row: dict, decision: dict,
                            question_id: str) -> dict:
    """Own envelope: never call the canonical scored-artifact validator."""
    protocol = loaded["protocol"]
    indexing = row.get("indexing")
    if (row.get("question_id") != question_id or type(row.get("correct")) is not bool
            or row.get("benchmark_failure") is not None
            or row.get("retrieval_only") is not False
            or row.get("judge_error") is not False
            or row.get("judge_parse_valid") is not True
            or type(row.get("context_sha")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", row["context_sha"]) is None
            or not isinstance(indexing, dict) or not isinstance(decision, dict)
            or decision.get("mode") != loaded["diagnostic"].MODE
            or decision.get("admitted") is not True
            or decision.get("kind") not in {"strict_healthy", "summary_degradation", "semantic_quarantine"}):
        raise ValueError("diagnostic_row_invalid")
    strict_healthy = protocol._validate_versioned_indexing(
        indexing, require_healthy=True, allow_failure=True)
    if (decision["kind"] == "semantic_quarantine") != (strict_healthy is False):
        raise ValueError("diagnostic_strict_health_mismatch")
    if decision["kind"] == "semantic_quarantine" and (
            indexing.get("outcome") != "failure" or indexing.get("healthy") is not False):
        raise ValueError("diagnostic_unhealthy_rewritten")
    return {"question_id": question_id, "correct": row["correct"],
        "benchmark_failure": None, "diagnostic_kind": decision["kind"],
        "strict_indexing_healthy": strict_healthy,
        "quarantined_chunks": decision["quarantined_chunks"],
        "summary_degraded_sessions": decision["summary_degraded_sessions"],
        "context_sha": row.get("context_sha")}


def _identity_manifest(loaded: dict, questions: list[dict], limits: dict,
                       helper_sha256: str) -> dict:
    strictness = loaded["strictness"]
    ids = [item["question_id"] for item in questions]
    rows = [hashlib.sha256(json.dumps(item, sort_keys=True,
        separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        for item in questions]
    manifest = {"schema": SCHEMA, "mode": loaded["diagnostic"].MODE,
        "canonical_r9_artifact": False, "official_model_score": False,
        "candidate_map_sha256": ACCEPTED_MAP_SHA256,
        "dataset_sha256": DATASET_SHA256,
        "selected_row_sha256": rows,
        "selected_source_order": "first_n",
        "expected_count": len(ids),
        "expected_ids_hash": strictness.content_hash(ids),
        "scored_run": True,
        "diagnostic_helper_sha256": helper_sha256,
        "runner_sha256": _sha(Path(__file__)),
        "transport_sha256": PINS["benchmarks/codex_subscription_staged_v1.py"],
        "ordinary_transport_sha256": PINS["benchmarks/codex_subscription_warm_v3.py"],
        "limits": limits, "rerolls": 0}
    manifest["run_id"] = strictness.content_hash(manifest)
    return manifest


def _question_worker(loaded: dict, budget: Any, limits: Any, question: dict,
                     index: int, output: Path, indexing_seconds: float) -> dict:
    key = f"q-{index:04d}"
    directory = output / key
    directory.mkdir(mode=0o700)
    client = None
    adapter = None
    close_error = False
    try:
        client = make_dual(loaded, budget, key, limits)
        accounted = AccountedClient(client, loaded["candidate"])
        memory = _memory_client(loaded, accounted)
        bridge = loaded["prior"].old.ChatBridge(memory, loaded["request_type"])
        base_adapter = loaded["prior"].old.make_adapter_class(loaded["lme"], memory)
        diagnostic_adapter = loaded["diagnostic"].make_diagnostic_adapter_class(
            loaded["lme"], loaded["protocol"], loaded["strictness"],
            loaded["summary_classifier"], base_adapter)
        adapter = diagnostic_adapter(directory / "hymem.sqlite", embeddings=False,
            aggregation_nodes=False, episode_granularity=False,
            pipeline_model="gpt-6-luna")
        adapter.open()
        row = loaded["lme"].evaluate_question(bridge, bridge, adapter, question,
            top_k=15, auto_ability=True, no_dream=False,
            permissive_default=True, distill=False, retrieval_only=False,
            max_input_tokens=loaded["lme"].DEFAULT_MAX_INPUT_TOKENS,
            max_input_bytes=loaded["lme"].DEFAULT_MAX_INPUT_BYTES,
            judge_protocol="legacy-custom", indexing_max_cycles=100,
            indexing_timeout_s=indexing_seconds, indexing_require_healthy=True)
        loaded["prior"].atomic_private(directory / "private-row.json", row)
        decision = adapter.diagnostic_indexing
        loaded["prior"].atomic_private(directory / "private-diagnostic-indexing.json", decision)
        projected = validate_diagnostic_row(loaded, row, decision, question["question_id"])
        if not accounted.reconcile() or not client.usage_complete:
            raise RuntimeError("question_accounting_invalid")
        return {"projection": projected, "accounting": accounted.counts,
            "stop_code": None}
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            budget.halt("interrupted")
            raise
        budget.halt("question_failure")
        reason = ("indexing_rejected" if isinstance(exc, loaded["lme"].IndexingConvergenceError)
            else "question_failure")
        return {"projection": None, "accounting": None, "stop_code": reason}
    finally:
        if adapter is not None:
            try:
                if isinstance(getattr(adapter, "last_indexing_summary", None), dict):
                    loaded["prior"].atomic_private(directory / "private-indexing.json",
                        adapter.last_indexing_summary)
                adapter.close()
            except BaseException:
                close_error = True
                budget.halt("adapter_cleanup_failure")
        if client is not None:
            try:
                client.close()
            except BaseException:
                close_error = True
                budget.halt("client_cleanup_failure")
        if close_error:
            raise RuntimeError("question_cleanup_failure")


def run_campaign(loaded: dict, *, output: Path, campaign_limits: Any,
                 canary_limits: Any, question_limits: Any,
                 indexing_seconds: float, workers: int,
                 helper_sha256: str, containment: Callable[[dict], None]) -> dict:
    """Execute only after an external launch policy proves containment.

    No resume/retry path exists: fresh output and one first-source selection are
    mandatory. A failed worker still occupies its original checkpoint ID.
    """
    bounded = all((type(getattr(cap, "turns", None)) is int
        and 0 < cap.turns <= MAX_LIMITS[name][0]
        and type(getattr(cap, "known_tokens", None)) is int
        and 0 < cap.known_tokens <= MAX_LIMITS[name][1]
        and type(getattr(cap, "seconds", None)) in (int, float)
        and 0 < cap.seconds <= MAX_LIMITS[name][2])
        for name, cap in (("campaign", campaign_limits), ("canary", canary_limits),
                          ("question", question_limits)))
    if (not callable(containment) or not bounded or not 1 <= workers <= 4
            or not 1 <= len(loaded["questions"]) <= 4
            or not 0 < indexing_seconds < question_limits.seconds < campaign_limits.seconds
            or not output.is_absolute() or output.exists() or not output.parent.is_dir()):
        raise ValueError("campaign_preflight_invalid")
    if containment(loaded) is not True:
        raise ValueError("containment_unverified")
    if _sha(loaded["dataset"]) != DATASET_SHA256:
        raise ValueError("dataset_drift_before_run")
    output.mkdir(mode=0o700)
    questions = loaded["questions"]
    ids = [item["question_id"] for item in questions]
    limits = {"campaign": vars(campaign_limits), "canary": vars(canary_limits),
        "question": vars(question_limits), "indexing_seconds": indexing_seconds,
        "workers": workers}
    manifest = _identity_manifest(loaded, questions, limits, helper_sha256)
    strictness = loaded["strictness"]
    budget = loaded["warm"].SharedBudget(campaign_limits, max_in_flight=workers)
    entries: dict[str, dict] = {}
    canary = None
    campaign_stop = None
    with strictness.AtomicCheckpoint(output / "diagnostic-checkpoint.json",
            manifest=manifest, expected_ids=ids, scored=True,
            resume=False, retry_failures=False) as checkpoint:
        try:
            canary = run_live_canary(loaded, budget, canary_limits, output / "canary")
            loaded["prior"].atomic_private(output / "private-canary-result.json", canary)
            if not canary["structural_valid"]:
                raise RuntimeError("canary_structural_failure")
            if not canary["model_gold_match"] and canary["quality_failure_reason"] is not None:
                if not canary["semantic_failure_proved"]:
                    raise RuntimeError("canary_nonsemantic_failure")
            if not budget.snapshot()["usage_complete"] or budget.snapshot()["stopped"]:
                raise RuntimeError("canary_budget_failure")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_question_worker, loaded, budget, question_limits,
                    question, index, output, indexing_seconds): question["question_id"]
                    for index, question in enumerate(questions)}
                for future in as_completed(futures):
                    qid = futures[future]
                    try:
                        item = future.result()
                    except BaseException:
                        budget.halt("worker_runtime_failure")
                        item = {"projection": None, "accounting": None,
                            "stop_code": "worker_runtime_failure"}
                    entries[qid] = item
                    if item["projection"] is None:
                        checkpoint.record(qid, row=None, failure=item["stop_code"])
                    else:
                        checkpoint.record(qid, row=item["projection"])
                    loaded["prior"].atomic_private(output / "private-progress.json", {
                        "completed": len(entries), "selected": len(ids),
                        "budget": budget.snapshot()})
        except BaseException as exc:
            budget.halt("campaign_failure")
            campaign_stop = type(exc).__name__ if type(exc).__name__ in {
                "RuntimeError", "ValueError", "KeyboardInterrupt"} else "finite_other"
        finally:
            for qid in ids:
                if qid not in entries:
                    checkpoint.record(qid, row=None, failure="not_started_after_campaign_stop")
            state = checkpoint.finalize()
    snapshot = budget.snapshot()
    if (snapshot["reserved"] or snapshot["in_flight"] or not snapshot["usage_complete"]
            or _sha(loaded["dataset"]) != DATASET_SHA256):
        campaign_stop = campaign_stop or "final_accounting_or_dataset_failure"
    completed = [item["projection"] for item in entries.values()
        if item["projection"] is not None]
    if snapshot["stopped"] or len(completed) != len(ids):
        campaign_stop = campaign_stop or snapshot["stop_code"] or "incomplete_questions"
    result = {"schema": SCHEMA, "run_id": manifest["run_id"],
        "canonical_r9_artifact": False, "official_model_score": False,
        "selected_denominator": len(ids), "scored_count": len(completed),
        "correct_count": sum(item["correct"] for item in completed),
        "incorrect_count": sum(not item["correct"] for item in completed),
        "quality_accuracy_full_selected": (sum(item["correct"] for item in completed) / len(ids)
            if len(completed) == len(ids) else None),
        "failed_or_unscored_count": len(ids) - len(completed),
        "strict_unhealthy_count": sum(item["strict_indexing_healthy"] is False
            for item in completed),
        "canary": canary, "campaign_stop": campaign_stop,
        "checkpoint_counts": state["counts"], "budget": snapshot}
    loaded["prior"].atomic_private(output / "diagnostic-result.json", result)
    return result


def verify_launch_receipt(root: Path, receipt_sha256: str, loaded: dict) -> dict:
    """Bind a one-shot run to a fresh operator-supplied source receipt."""
    path = root / "launch-receipt.json"
    marker = root / "launch-attempt.json"
    if (re.fullmatch(r"[0-9a-f]{64}", receipt_sha256) is None
            or not _regular(path) or _sha(path) != receipt_sha256
            or not _regular(marker)):
        raise ValueError("launch_receipt_invalid")
    receipt = json.loads(path.read_text())
    expected = {"schema", "root", "unit", "expected_cgroup", "source_sha256",
        "candidate_map_sha256", "inventory_sha256", "dataset_sha256",
        "binary_sha256", "selected_count", "workers", "indexing_seconds",
        "output_dir", "limits"}
    if (type(receipt) is not dict or set(receipt) != expected
            or receipt["schema"] != "luna-lme-diagnostic-launch-v1"
            or receipt["root"] != str(root)
            or receipt["candidate_map_sha256"] != ACCEPTED_MAP_SHA256
            or receipt["dataset_sha256"] != DATASET_SHA256
            or receipt["binary_sha256"] != _sha(loaded["binary"])
            or receipt["selected_count"] != len(loaded["questions"])
            or type(receipt["workers"]) is not int
            or not 1 <= receipt["workers"] <= receipt["selected_count"]
            or receipt["indexing_seconds"] != 10_800
            or receipt["output_dir"] != str(root / "run")
            or type(receipt["unit"]) is not str
            or re.fullmatch(r"hymem-luna-lme-diagnostic-[a-z0-9_-]{8,}\.service", receipt["unit"]) is None
            or type(receipt["expected_cgroup"]) is not str
            or not receipt["expected_cgroup"].endswith("/" + receipt["unit"])):
        raise ValueError("launch_receipt_identity_invalid")
    sources = receipt["source_sha256"]
    expected_sources = {**PINS,
        "benchmarks/lme_diagnostic.py": DIAGNOSTIC_HELPER_SHA256,
        "tools/diagnostics/luna_lme_diagnostic_v1.py": _sha(Path(__file__))}
    if (type(sources) is not dict or sources != expected_sources
            or receipt["inventory_sha256"] != _sha(loaded["root"] / "source-map.json")):
        raise ValueError("launch_receipt_source_invalid")
    for relative, digest in expected_sources.items():
        if _sha(loaded["code"] / relative) != digest:
            raise ValueError("launch_receipt_source_drift")
    observed_marker = json.loads(marker.read_text())
    if (observed_marker != {"receipt_sha256": receipt_sha256, "one_shot": True}):
        raise ValueError("launch_attempt_invalid")
    return receipt


def verify_live_containment(root: Path, receipt: dict) -> None:
    """Read actual systemd/cgroup state; the receipt alone never authorizes spend."""
    properties = ("ActiveState,SubState,MainPID,ControlGroup,NRestarts,MemoryMax,"
        "TasksMax,CPUQuotaPerSecUSec,KillMode,Restart,RemainAfterExit,OOMPolicy,"
        "RuntimeMaxUSec,TimeoutStopUSec")
    completed = subprocess.run(["/usr/bin/systemctl", "--user", "show",
        receipt["unit"], "--property=" + properties, "--no-pager"],
        capture_output=True, text=True, check=True, timeout=10)
    values = dict(line.split("=", 1) for line in completed.stdout.splitlines() if "=" in line)
    if (set(values) != set(properties.split(","))
            or values["ActiveState"] != "active" or values["SubState"] != "running"
            or values["MainPID"] != str(os.getpid())
            or values["ControlGroup"] != receipt["expected_cgroup"]
            or values["NRestarts"] != "0" or values["MemoryMax"] != "4294967296"
            or values["TasksMax"] != "128"
            or values["CPUQuotaPerSecUSec"] not in {"2s", "2.000s"}
            or values["KillMode"] != "control-group"
            or values["Restart"] != "no" or values["RemainAfterExit"] != "yes"
            or values["OOMPolicy"] != "kill"
            or values["RuntimeMaxUSec"] not in {"4h 2min 10s", "4h 2min 10.000s", "14530s", "14530.000s"}
            or values["TimeoutStopUSec"] not in {"10s", "10.000s"}):
        raise ValueError("service_policy_invalid")
    cgroup = receipt["expected_cgroup"]
    if f"0::{cgroup}" not in Path("/proc/self/cgroup").read_text().splitlines():
        raise ValueError("cgroup_identity_invalid")
    cgroup_path = Path("/sys/fs/cgroup" + cgroup)
    if (not cgroup_path.resolve().is_relative_to(Path("/sys/fs/cgroup"))
            or (cgroup_path / "memory.max").read_text().strip() != "4294967296"
            or (cgroup_path / "pids.max").read_text().strip() != "128"):
        raise ValueError("cgroup_resource_policy_invalid")
    cpu = (cgroup_path / "cpu.max").read_text().split()
    if len(cpu) != 2 or not all(value.isdecimal() for value in cpu) or int(cpu[0]) != 2 * int(cpu[1]):
        raise ValueError("cgroup_cpu_policy_invalid")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "inventory", "inventory-sha256", "dataset",
                 "binary", "binary-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--questions", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--output-dir")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    loaded = load_verified(Path(args.root), Path(args.inventory),
        args.inventory_sha256, Path(args.dataset), Path(args.binary),
        args.binary_sha256,
        DIAGNOSTIC_HELPER_SHA256)
    selected = loaded["questions"][:args.questions]
    loaded["questions"] = selected
    if args.run:
        if (not args.receipt_sha256 or not args.output_dir or args.workers > args.questions
                or Path(args.output_dir) != loaded["root"] / "run"
                or Path(args.inventory) != loaded["root"] / "source-map.json"):
            raise ValueError("live_arguments_invalid")
        receipt = verify_launch_receipt(loaded["root"], args.receipt_sha256, loaded)
        if (receipt["limits"] != {key: list(value) for key, value in MAX_LIMITS.items()}
                or receipt["workers"] != args.workers):
            raise ValueError("launch_limits_invalid")
        warm = loaded["warm"]
        verify_live_containment(loaded["root"], receipt)
        log_fd = os.open(loaded["root"] / "private-diagnostic-run.log",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
        with os.fdopen(log_fd, "w", encoding="utf-8") as log:
            with redirect_stdout(log), redirect_stderr(log):
                result = run_campaign(loaded, output=Path(args.output_dir),
                    campaign_limits=warm.BudgetLimits(*MAX_LIMITS["campaign"]),
                    canary_limits=warm.BudgetLimits(*MAX_LIMITS["canary"]),
                    question_limits=warm.BudgetLimits(*MAX_LIMITS["question"]),
                    indexing_seconds=10_800, workers=args.workers,
                    helper_sha256=DIAGNOSTIC_HELPER_SHA256,
                    containment=lambda _loaded: verify_live_containment(loaded["root"], receipt))
        print(json.dumps({"schema": SCHEMA, "run_id": result["run_id"],
            "selected_denominator": result["selected_denominator"],
            "scored_count": result["scored_count"],
            "campaign_stop": result["campaign_stop"]}, sort_keys=True))
        return 0 if result["campaign_stop"] is None else 1
    print(json.dumps({"schema": SCHEMA, "preflight_verified": True,
        "selected_count": len(selected), "selected_ids": [q["question_id"] for q in selected],
        "candidate_map_sha256": ACCEPTED_MAP_SHA256, "model_calls": 0},
        sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
