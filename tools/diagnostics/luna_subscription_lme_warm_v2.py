"""Versioned attributed warm-process Luna subscription experiment (not R9)."""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import sys
import threading
import time
import traceback
import types

CONCURRENT_SHA256 = "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0"
PILOT_SHA256 = "0cab3a92812ab7ed0527edff9dc520682d4e535f99b6f8206c4f93fe78cf05b0"
SCHEMA = "luna-subscription-lme-warm-v2"
WARM_SHA256 = "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593"
MAX_SELECTED_QUESTIONS = 500
RUNNER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

_old_path = Path(__file__).resolve().with_name("luna_subscription_pilot.py")
_old_bytes = _old_path.read_bytes()
if hashlib.sha256(_old_bytes).hexdigest() != PILOT_SHA256:
    raise RuntimeError("pilot_source_drift")
old = types.ModuleType("pinned_luna_pilot_helper")
old.__file__ = str(_old_path)
exec(compile(_old_bytes, str(_old_path), "exec"), old.__dict__)


class CampaignStop(BaseException):
    pass


def check(condition: bool, code: str) -> None:
    if not condition:
        raise CampaignStop(code)


def warm_metrics(client) -> dict:
    return {key: getattr(client, key, 0) for key in (
        "processes_started", "rotations", "cold_calls", "warm_calls",
        "startup_seconds", "unsubscribe_seconds", "rotation_cleanup_seconds",
        "final_cleanup_seconds")}


def safe_first_failure(value: object, warm) -> dict | None:
    """Permit only transport-defined atoms and bounded numeric lifecycle data."""
    if type(value) is not dict:
        return None
    code = value.get("code")
    if type(code) is not str or len(code) > 100:
        return None
    if code not in warm._FIXED_CODES | warm._OWN_BUDGET_CODES | {"fixed_other"}:
        parts = code.split(":", 1)
        if len(parts) != 2 or parts[0] not in {"process_exit", "protocol_failure", "rpc_failure", "unexpected_notification"}:
            return None
        allowed = warm._EVENT_METHODS if parts[0] == "unexpected_notification" else warm._RPC_METHODS
        if parts[1] not in allowed:
            return None
    phase = value.get("phase")
    if type(phase) is not str or phase not in {"startup", "rotation_cleanup", "preflight", "run", "unsubscribe", "cleanup"}:
        return None
    rpc = value.get("rpc")
    if rpc is not None and (type(rpc) is not str or rpc not in warm._RPC_METHODS):
        return None
    result = {"code": code, "phase": phase, "rpc": rpc}
    for key in ("process_index", "request_index", "retired_count", "queue_count", "known_tokens"):
        item = value.get(key)
        if type(item) is not int or not 0 <= item <= 10**12:
            return None
        result[key] = item
    age = value.get("process_age_seconds")
    if age is not None and (type(age) not in (float, int) or not math.isfinite(age) or not 0 <= age <= 1_000_000):
        return None
    result["process_age_seconds"] = age
    for key in ("turn_admitted", "known_usage", "usage_complete"):
        item = value.get(key)
        if type(item) is not bool:
            return None
        result[key] = item
    return result


def atomic_private(path: Path, value: object) -> None:
    """Replace only an owned private artifact, durably and without partial JSON."""
    encoded = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    temporary = path.with_name(path.name + ".pending")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def iter_source_items(path: Path, count: int):
    """Yield the first N array objects; never retain another item."""
    check(type(count) is int and 1 <= count <= MAX_SELECTED_QUESTIONS,
          "question_count_invalid")
    decoder = json.JSONDecoder()
    yielded = 0
    with path.open("r", encoding="utf-8") as source:
        check(source.read(1) == "[", "dataset_shape_invalid")
        buffer = ""
        eof = False
        while yielded < count:
            while True:
                buffer = buffer.lstrip()
                if not buffer and not eof:
                    block = source.read(64 * 1024)
                    eof = not block
                    buffer += block
                check(bool(buffer), "dataset_short")
                if yielded == 0:
                    check(buffer[0] not in ",]", "dataset_shape_invalid")
                elif buffer[0] == ",":
                    buffer = buffer[1:].lstrip()
                    if not buffer and not eof:
                        block = source.read(64 * 1024)
                        eof = not block
                        buffer += block
                    buffer = buffer.lstrip()
                    check(bool(buffer) and buffer[0] not in ",]",
                          "dataset_shape_invalid")
                    continue
                elif buffer[0] == "]":
                    raise CampaignStop("dataset_short")
                try:
                    item, end = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    check(not eof and len(buffer.encode("utf-8")) < old.MAX_FIRST_ITEM_BYTES,
                          "dataset_item_invalid")
                    block = source.read(64 * 1024)
                    eof = not block
                    buffer += block
                    continue
                check(type(item) is dict, "dataset_shape_invalid")
                check(len(buffer[:end].encode("utf-8")) <= old.MAX_FIRST_ITEM_BYTES,
                      "dataset_item_limit")
                yielded += 1
                buffer = buffer[end:].lstrip()
                if not buffer and not eof:
                    block = source.read(64 * 1024)
                    eof = not block
                    buffer += block
                check(bool(buffer) and buffer[0] in ",]", "dataset_shape_invalid")
                break
            yield item


def first_source_items(path: Path, count: int) -> list[dict]:
    """Small diagnostic helper; campaign uses a validated streaming sequence."""
    return list(iter_source_items(path, count))


class SelectedQuestions:
    """Two-pass source-ordered selection with bounded selected-row memory."""
    def __init__(self, path: Path, count: int, protocol):
        self.path, self.count, self.protocol = path, count, protocol
        seen = set()
        self._row_digests = []
        for raw in iter_source_items(path, count):
            self._row_digests.append(self._digest_row(raw))
            item = protocol.validate_lme_dataset([raw], scale="S")[0]
            qid = item["question_id"]
            check(qid not in seen, "duplicate_selected_question")
            seen.add(qid)
        check(len(seen) == count, "dataset_short")
    def __len__(self):
        return self.count
    def __iter__(self):
        for index, raw in enumerate(iter_source_items(self.path, self.count)):
            check(self._digest_row(raw) == self._row_digests[index],
                  "selected_row_drift")
            yield self.protocol.validate_lme_dataset([raw], scale="S")[0]
    @staticmethod
    def _digest_row(row):
        encoded = json.dumps(row, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(encoded).digest()


def load_verified(*, candidate: Path, inventory_stamp: Path, inventory_sha256: str,
                  dataset: Path, dataset_sha256: str, binary: Path,
                  base_path: Path, concurrent_path: Path, warm_path: Path):
    check(all(path.is_absolute() for path in (candidate, inventory_stamp, dataset,
                                               binary, base_path, concurrent_path, warm_path)), "input_invalid")
    check(candidate.is_dir() and not candidate.is_symlink(), "input_invalid")
    check(all(old.regular_absolute(path) for path in
              (inventory_stamp, dataset, binary, base_path, concurrent_path, warm_path)), "input_invalid")
    check(old.HEX.fullmatch(inventory_sha256) is not None and
          old.HEX.fullmatch(dataset_sha256) is not None, "input_invalid")
    check(old.digest(Path(old.__file__)) == PILOT_SHA256, "pilot_source_drift")
    check(old.digest(base_path) == old.TRANSPORT_SHA256, "base_transport_drift")
    check(old.digest(concurrent_path) == CONCURRENT_SHA256, "concurrent_transport_drift")
    check(old.digest(warm_path) == WARM_SHA256, "warm_transport_drift")
    check(concurrent_path.parent == base_path.parent and
          concurrent_path.name == "codex_subscription_concurrent_v2.py" and
          base_path.name == "codex_subscription.py" and
          warm_path.parent == base_path.parent and
          warm_path.name == "codex_subscription_warm_v2.py", "transport_layout_invalid")
    files = old.verify_inventory(candidate, inventory_stamp, inventory_sha256)
    check(dataset_sha256 == old.DATASET_SHA256 and old.digest(dataset) == dataset_sha256,
          "dataset_drift")
    for name, module in tuple(sys.modules.items()):
        if name == "hymem" or name.startswith("hymem.") or name == "benchmarks" or name.startswith("benchmarks."):
            file = getattr(module, "__file__", None)
            check(file is None or Path(file).resolve().is_relative_to(candidate.resolve()),
                  "cached_source_mismatch")
    sys.path.insert(0, str(candidate))
    from hymem.extraction.llm import LLMRequest as request_type
    from benchmarks import extraction_canary as canary
    from benchmarks import longmemeval_adapter as lme
    from benchmarks import lme_protocol as protocol
    from hymem.extraction import chunk
    for module in (sys.modules["hymem.extraction.llm"], canary, chunk, lme, protocol):
        check(Path(module.__file__).resolve().is_relative_to(candidate.resolve()),
              "loaded_source_mismatch")
    concurrent = types.ModuleType("pinned_concurrent_transport")
    concurrent.__file__ = str(concurrent_path)
    sys.modules[concurrent.__name__] = concurrent
    content = concurrent_path.read_bytes()
    check(hashlib.sha256(content).hexdigest() == CONCURRENT_SHA256,
          "concurrent_transport_drift")
    exec(compile(content, str(concurrent_path), "exec"), concurrent.__dict__)
    warm = types.ModuleType("pinned_warm_subscription_transport")
    warm.__file__ = str(warm_path)
    sys.modules[warm.__name__] = warm
    warm_source = warm_path.read_bytes()
    check(hashlib.sha256(warm_source).hexdigest() == WARM_SHA256, "warm_transport_drift")
    exec(compile(warm_source, str(warm_path), "exec"), warm.__dict__)
    return files, warm.concurrent, request_type, canary, chunk, lme, protocol, warm


class _ProgressClient:
    def __init__(self, delegate, callback, active, key, lock):
        self.delegate, self.callback = delegate, callback
        self.active, self.key, self.lock = active, key, lock
    def __getattr__(self, name):
        return getattr(self.delegate, name)
    def _publish(self):
        try:
            self.callback()
        except Exception:
            self.delegate.budget.halt("artifact_write_failure")
            raise CampaignStop("artifact_write_failure")
    def complete(self, request):
        with self.lock:
            self.active.add(self.key)
        try:
            self._publish()
            return self.delegate.complete(request)
        finally:
            with self.lock:
                self.active.discard(self.key)
            self._publish()


def make_memory_client(delegate, *, concurrent_sha256: str = CONCURRENT_SHA256):
    """Same source identity for every question, independent of question index."""
    class MemoryClient:
        def __getattr__(self, name):
            return getattr(delegate, name)
        def complete(self, request):
            return delegate.complete(request)
        def phase1_producer_declaration(self):
            from hymem.extraction.producer import Phase1ProducerDeclaration
            return Phase1ProducerDeclaration(
                client_id="luna-subscription-concurrent-memory-v2",
                implementation="sha256:" + RUNNER_SHA256,
                model="gpt-6-luna", endpoint=None,
                effective_request={
                    "transport_sha256": concurrent_sha256,
                    "base_transport_sha256": old.TRANSPORT_SHA256,
                    "account_route": "chatgpt", "model": "gpt-6-luna",
                    "isolation": "read-only-no-network-empty-environments",
                    "thread_lifecycle": "fresh-ephemeral-per-request",
                    "messages": ["system", "user"],
                    "temperature_effective": None, "output_cap_effective": None,
                    "json_mode_effective": None, "internal_http_attempts": None,
                },
                retry_policy={"pilot_rerolls": 0, "transport_retry": "none",
                              "provider_internal_retries": "unknown"})
        def memory_producer_declaration(self):
            return self.phase1_producer_declaration()
    return MemoryClient()


def _safe_census(adapter) -> dict | None:
    try:
        state = adapter.hy.dream_status()
        prefixes = ("pending_", "malformed_", "quarantined_", "terminal_loss_")
        extras = {"coverage_integrity_failures", "summary_missing_sessions",
                  "summary_degraded_sessions"}
        result = {key: value for key, value in state.items()
                  if (key.startswith(prefixes) or key in extras)
                  and type(value) is int and value >= 0}
        result["summary_healthy"] = state.get("summary_healthy") if type(
            state.get("summary_healthy")) is bool else None
        return result
    except Exception:
        return None


def terminalize_owned_dream_run(adapter, question_dir: Path, *, abnormal: bool) -> dict:
    """Close only lifecycle telemetry in this fresh, owned question store.

    This runs after the frozen dream call has unwound. It never edits chunks,
    messages, cursors, or completion/health claims. An active lease or unknown
    DB identity is a cleanup failure, not permission to repair another writer.
    """
    expected = question_dir / "hymem.sqlite"
    if (not question_dir.is_absolute() or question_dir.is_symlink()
            or expected.is_symlink() or not expected.is_file()
            or not stat.S_ISREG(expected.stat().st_mode)
            or Path(adapter.db_path).resolve() != expected.resolve()
            or expected.parent.resolve() != question_dir.resolve()):
        raise CampaignStop("owned_store_identity_invalid")
    connection = sqlite3.connect(expected.as_uri() + "?mode=rw", uri=True,
                                 timeout=0.0, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout=0")
        connection.execute("BEGIN IMMEDIATE")
        leases = connection.execute("SELECT COUNT(*) FROM run_lock").fetchone()[0]
        if leases != 0:
            connection.execute("ROLLBACK")
            raise CampaignStop("active_dream_lease")
        before = connection.execute(
            "SELECT COUNT(*) FROM dream_runs WHERE ended_at IS NULL").fetchone()[0]
        terminalized = 0
        if before and abnormal:
            terminalized = connection.execute(
                "UPDATE dream_runs SET ended_at=CURRENT_TIMESTAMP, "
                "error='execution_interrupted:subscription_control' "
                "WHERE ended_at IS NULL").rowcount
        after = connection.execute(
            "SELECT COUNT(*) FROM dream_runs WHERE ended_at IS NULL").fetchone()[0]
        if after != 0 or terminalized != (before if abnormal else 0):
            connection.execute("ROLLBACK")
            raise CampaignStop("open_dream_run_unresolved")
        connection.execute("COMMIT")
        return {"open_runs_before": before, "terminalized": terminalized,
                "open_runs_after": after, "active_leases_after": 0}
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def run_campaign(*, concurrent, request_type, canary, chunk, lme, protocol,
                 binary: str, questions: list[dict], output: Path,
                 campaign_limits, question_limits, canary_limits,
                 indexing_timeout_s: float, workers: int = 2,
                 client_factory=None, progress=None, warm=None,
                 warm_max_requests: int = 16, warm_max_age_seconds: float = 300) -> dict:
    check(type(workers) is int and 1 <= workers <= 4 and
          1 <= len(questions) <= MAX_SELECTED_QUESTIONS,
          "worker_count_invalid")
    check(0 < indexing_timeout_s < question_limits.seconds < campaign_limits.seconds,
          "deadline_order_invalid")
    budget = concurrent.SharedBudget(campaign_limits, max_in_flight=workers)
    lock = threading.RLock()
    active = set()
    result = {"schema": SCHEMA, "canary": None, "questions": [None] * len(questions),
              "campaign_stop": None, "budget": None,
              "canonical_r9_artifact": False, "official_model_score": False,
              "internal_http_attempts": None,
              "transport_policy": {"kind": "warm_process_fresh_ephemeral_thread",
                  "max_requests_per_process": warm_max_requests,
                  "rotate_before_call_after_seconds": warm_max_age_seconds}}
    def publish():
        with lock:
            result["budget"] = budget.snapshot()
            result["active_invocations"] = len(active)
            result["usage_complete_now"] = (
                result["budget"]["usage_complete"] and not active)
            if progress:
                progress(result)
    def make_client(key, limits):
        return (client_factory(key, limits, budget) if client_factory else
                warm.WarmSubscriptionClient(binary, budget, key, limits,
                    max_requests=warm_max_requests,
                    max_age_seconds=warm_max_age_seconds))
    canary_started_at = time.monotonic()
    canary_client = make_client("canary", canary_limits)
    try:
        # The frozen request/fixture machinery is unchanged; v2 only treats
        # omitted optional type hints as diagnostic, never invents them.
        try:
            result["canary"] = old.experimental_canary(
                canary, chunk, _ProgressClient(canary_client, publish, active, "canary", lock),
                evidence=lambda value: atomic_private(output / "private-canary-evidence.json", value))
        finally:
            try:
                canary_client.close()
            except BaseException:
                budget.halt("canary_cleanup_failure")
                raise CampaignStop("canary_cleanup_failure")
            finally:
                result["canary_transport"] = warm_metrics(canary_client)
        publish()
        check(result["canary"]["passed"] is True and
              budget.snapshot()["usage_complete"] is True, "canary_failed")
        check(time.monotonic() - canary_started_at < canary_limits.seconds and
              time.monotonic() - budget.started_at < campaign_limits.seconds,
              "canary_wall_limit")
    except concurrent.ConcurrentStop as exc:
        budget.halt(str(exc))
        result["campaign_stop"] = str(exc)
        publish()
        return result
    except CampaignStop as exc:
        budget.halt(str(exc))
        result["campaign_stop"] = str(exc)
        publish()
        return result
    except Exception:
        budget.halt("canary_runtime_failure")
        result["campaign_stop"] = "canary_runtime_failure"
        publish()
        return result
    except BaseException:
        budget.halt("canary_interrupted")
        result["campaign_stop"] = budget.snapshot()["stop_code"]
        publish()
        return result

    def worker(index: int, question: dict) -> dict:
        key = f"q-{index:04d}"
        question_started_at = time.monotonic()
        directory = output / key
        directory.mkdir(mode=0o700)
        adapter = None
        delegate = None
        entry = {"index": index, "question_started": False,
                 "question_completed": False, "correct": None,
                 "stop_code": None, "indexing_last_cycle_stale": False,
                 "indexing_health": None, "fresh_census": None,
                 "cleanup_ok": None, "row_private": None}
        try:
            if budget.snapshot()["stopped"]:
                entry["stop_code"] = "campaign_stopped_before_question"
                return entry
            delegate = make_client(key, question_limits)
            wrapped = _ProgressClient(delegate, publish, active, key, lock)
            memory = make_memory_client(wrapped, concurrent_sha256=WARM_SHA256)
            bridge = old.ChatBridge(memory, request_type)
            adapter = old.make_adapter_class(lme, memory)(
                directory / "hymem.sqlite", embeddings=False,
                aggregation_nodes=False, episode_granularity=False,
                pipeline_model="gpt-6-luna")
            adapter.open()
            entry["question_started"] = True
            with lock:
                result["questions"][index] = dict(entry)
            publish()
            row = lme.evaluate_question(
                bridge, bridge, adapter, question, top_k=15,
                auto_ability=True, no_dream=False, permissive_default=True,
                distill=False, retrieval_only=False,
                max_input_tokens=lme.DEFAULT_MAX_INPUT_TOKENS,
                max_input_bytes=lme.DEFAULT_MAX_INPUT_BYTES,
                judge_protocol="legacy-custom", indexing_max_cycles=100,
                indexing_timeout_s=indexing_timeout_s,
                indexing_require_healthy=True)
            atomic_private(directory / "private-row.json", row)
            indexing = row.get("indexing")
            check(type(indexing) is dict and indexing.get("outcome") == "success"
                  and indexing.get("healthy") is True
                  and indexing.get("summary_healthy") is True
                  and protocol._validate_versioned_indexing(
                      indexing, require_healthy=True, allow_failure=False) is True,
                  "indexing_unhealthy")
            entry["indexing_health"] = old.indexing_health_snapshot(indexing)
            check(row.get("benchmark_failure") is None and
                  row.get("judge_error") is False and
                  row.get("judge_parse_valid") is True and
                  type(row.get("correct")) is bool, "question_unscored")
            check(delegate.usage_complete, "usage_incomplete")
            check(time.monotonic() - question_started_at < question_limits.seconds
                  and time.monotonic() - budget.started_at < campaign_limits.seconds,
                  "wall_limit_after_score")
            entry["correct"] = row["correct"]
            entry["question_completed"] = True
        except concurrent.ConcurrentStop as exc:
            entry["stop_code"] = str(exc)
            if budget.snapshot()["stopped"]:
                result["campaign_stop"] = budget.snapshot()["stop_code"]
        except CampaignStop as exc:
            entry["stop_code"] = str(exc)
        except Exception as exc:
            if isinstance(exc, lme.IndexingConvergenceError):
                entry["stop_code"] = "indexing_convergence_failure"
            else:
                entry["stop_code"] = "question_runtime_failure"
                budget.halt("question_runtime_failure")
                result["campaign_stop"] = "question_runtime_failure"
            with (directory / "private-failure.txt").open("w", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            os.chmod(directory / "private-failure.txt", 0o600)
        except BaseException:
            budget.halt("worker_interrupted")
            result["campaign_stop"] = "worker_interrupted"
            entry["stop_code"] = "worker_interrupted"
            raise
        finally:
            if adapter is not None:
                last = getattr(adapter, "last_indexing_summary", None)
                if isinstance(last, dict):
                    entry["indexing_health"] = old.indexing_health_snapshot(last)
                    failure = last.get("failure")
                    if isinstance(failure, dict) and isinstance(failure.get("code"), str):
                        entry["indexing_failure_code"] = failure["code"]
                    entry["indexing_last_cycle_stale"] = (
                        entry.get("indexing_failure_code") == "timeout_during_cycle" or
                        entry["stop_code"] is not None)
                    try:
                        atomic_private(directory / "private-indexing.json", last)
                    except Exception:
                        budget.halt("artifact_write_failure")
                        result["campaign_stop"] = "artifact_write_failure"
                        entry["stop_code"] = "artifact_write_failure"
                        entry["question_completed"] = False
                try:
                    entry["dream_run_housekeeping"] = terminalize_owned_dream_run(
                        adapter, directory,
                        abnormal=entry["stop_code"] is not None or
                                 not entry["question_completed"])
                except BaseException:
                    entry["dream_run_housekeeping"] = None
                    entry["question_completed"] = False
                    entry["stop_code"] = "dream_run_housekeeping_failure"
                    budget.halt("dream_run_housekeeping_failure")
                    result["campaign_stop"] = budget.snapshot()["stop_code"]
                entry["fresh_census"] = _safe_census(adapter)
                try:
                    adapter.close()
                    entry["cleanup_ok"] = entry["dream_run_housekeeping"] is not None
                except BaseException:
                    entry["cleanup_ok"] = False
                    entry["question_completed"] = False
                    entry["stop_code"] = "cleanup_failure"
                    budget.halt("cleanup_failure")
                    result["campaign_stop"] = "cleanup_failure"
            else:
                entry["cleanup_ok"] = True
            if delegate is not None:
                try:
                    delegate.close()
                except BaseException:
                    entry["cleanup_ok"] = False
                    entry["question_completed"] = False
                    entry["stop_code"] = "warm_client_cleanup_failure"
                    budget.halt("warm_client_cleanup_failure")
                    result["campaign_stop"] = budget.snapshot()["stop_code"]
                finally:
                    entry["transport"] = warm_metrics(delegate)
            if not entry["question_completed"]:
                entry["correct"] = None
            entry.pop("row_private")
            try:
                atomic_private(directory / "private-result.json", entry)
            except Exception:
                budget.halt("artifact_write_failure")
                result["campaign_stop"] = "artifact_write_failure"
                entry["stop_code"] = "artifact_write_failure"
                entry["question_completed"] = False
                entry["correct"] = None
            with lock:
                result["questions"][index] = entry
            publish()
        return entry

    with ThreadPoolExecutor(max_workers=workers) as pool:
        source = enumerate(questions)
        futures = {}
        def replenish():
            while len(futures) < workers and not budget.snapshot()["stopped"]:
                try:
                    index, item = next(source)
                except StopIteration:
                    break
                except (CampaignStop, Exception):
                    budget.halt("dataset_stream_failure")
                    result["campaign_stop"] = "dataset_stream_failure"
                    break
                futures[pool.submit(worker, index, item)] = index
        try:
            replenish()
            while futures:
                completed, _pending = wait(tuple(futures), return_when=FIRST_COMPLETED)
                for future in completed:
                    futures.pop(future)
                    try:
                        future.result()
                    except concurrent.ConcurrentStop:
                        budget.halt("worker_unhandled_stop")
                        result["campaign_stop"] = budget.snapshot()["stop_code"]
                        publish()
                    except Exception:
                        budget.halt("worker_unhandled_failure")
                        result["campaign_stop"] = budget.snapshot()["stop_code"]
                        publish()
                    except BaseException:
                        budget.halt("worker_interrupted")
                        result["campaign_stop"] = budget.snapshot()["stop_code"]
                        publish()
                replenish()
        except (KeyboardInterrupt, SystemExit):
            # Halt before the pool waits for its bounded in-flight turns. No
            # new question or model admission is possible after this point.
            budget.halt("controller_interrupted")
            result["campaign_stop"] = budget.snapshot()["stop_code"]
            for future in tuple(futures):
                try:
                    future.result()
                except BaseException:
                    pass
    if budget.snapshot()["stopped"]:
        result["campaign_stop"] = budget.snapshot()["stop_code"]
    publish()
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("binary", "base-transport", "concurrent-transport", "warm-transport", "candidate",
                 "inventory-stamp", "inventory-sha256", "dataset", "dataset-sha256",
                 "output-dir"):
        parser.add_argument("--" + name, required=True)
    for name in ("campaign-turns", "campaign-known-tokens", "campaign-seconds",
                 "question-turns", "question-known-tokens", "question-seconds",
                 "canary-turns", "canary-known-tokens", "canary-seconds",
                 "indexing-seconds"):
        parser.add_argument("--" + name, required=True, type=float if "seconds" in name else int)
    parser.add_argument("--questions", type=int, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--warm-max-requests", type=int, default=16)
    parser.add_argument("--warm-max-age-seconds", type=float, default=300)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    report = {"schema": SCHEMA, "ok": False, "stop_code": None,
              "first_failure": None,
              "internal_http_attempts": None, "canonical_r9_artifact": False}
    created_output = False
    try:
        check(output.is_absolute() and not output.exists() and output.parent.is_dir(),
              "output_not_fresh")
        check(type(args.warm_max_requests) is int and 1 <= args.warm_max_requests <= 16
              and 1 <= args.warm_max_age_seconds <= 300,
              "warm_lifetime_invalid")
        files, concurrent, request_type, canary, chunk, lme, protocol, warm = load_verified(
            candidate=Path(args.candidate), inventory_stamp=Path(args.inventory_stamp),
            inventory_sha256=args.inventory_sha256, dataset=Path(args.dataset),
            dataset_sha256=args.dataset_sha256, binary=Path(args.binary),
            base_path=Path(args.base_transport), concurrent_path=Path(args.concurrent_transport),
            warm_path=Path(args.warm_transport))
        campaign_limits = concurrent.BudgetLimits(args.campaign_turns,
            args.campaign_known_tokens, args.campaign_seconds)
        question_limits = concurrent.BudgetLimits(args.question_turns,
            args.question_known_tokens, args.question_seconds)
        canary_limits = concurrent.BudgetLimits(args.canary_turns,
            args.canary_known_tokens, args.canary_seconds)
        questions = SelectedQuestions(Path(args.dataset), args.questions, protocol)
        check(old.digest(Path(args.dataset)) == args.dataset_sha256,
              "dataset_drift_after_selection")
        output.mkdir(mode=0o700)
        created_output = True
        def progress(value):
            atomic_private(output / "private-progress.json", value)
        with (output / "private-run.log").open("w", encoding="utf-8") as log:
            os.chmod(output / "private-run.log", 0o600)
            with redirect_stdout(log), redirect_stderr(log):
                result = run_campaign(concurrent=concurrent, request_type=request_type,
                    canary=canary, chunk=chunk, lme=lme, protocol=protocol,
                    binary=str(Path(args.binary)), questions=questions, output=output,
                    campaign_limits=campaign_limits, question_limits=question_limits,
                    canary_limits=canary_limits, indexing_timeout_s=args.indexing_seconds,
                    workers=args.workers, progress=progress, warm=warm,
                    warm_max_requests=args.warm_max_requests,
                    warm_max_age_seconds=args.warm_max_age_seconds)
        atomic_private(output / "private-result.json", result)
        check(old.digest(Path(args.dataset)) == args.dataset_sha256,
              "dataset_drift_after_run")
        budget = result["budget"] or {}
        report.update({"ok": result["campaign_stop"] is None
            and (result["canary"] or {}).get("passed") is True
            and result.get("usage_complete_now") is True
            and budget.get("in_flight") == 0 and all(
            isinstance(item, dict) and item.get("question_completed") is True
            and item.get("cleanup_ok") is True
            for item in result["questions"]),
            "stop_code": result["campaign_stop"], "canary_passed":
                (result["canary"] or {}).get("passed") is True,
            "questions_completed": sum(isinstance(item, dict) and item.get(
                "question_completed") is True for item in result["questions"]),
            "questions_selected": len(questions), "turns": budget.get("turns"),
            "known_tokens": budget.get("known_tokens"),
            "usage_complete": result.get("usage_complete_now"),
            "in_flight": budget.get("in_flight"),
            "active_invocations": result.get("active_invocations"),
            "known_tokens_scope": budget.get("known_tokens_scope"),
            "correct_count": sum(isinstance(item, dict) and item.get("question_completed")
                                 and item.get("correct") is True for item in result["questions"]),
            "incorrect_count": sum(isinstance(item, dict) and item.get("question_completed")
                                   and item.get("correct") is False for item in result["questions"]),
            "failed_count": sum(isinstance(item, dict) and item.get("stop_code") is not None
                                for item in result["questions"]),
            "not_started_count": sum(item is None or not item.get("question_started")
                                     for item in result["questions"]),
            "source_files_verified": files,
            "dataset_sha256": args.dataset_sha256,
            "candidate_source_map_sha256": old.SOURCE_MAP_SHA256,
            "pilot_helper_sha256": PILOT_SHA256,
            "runner_sha256": RUNNER_SHA256,
            "base_transport_sha256": old.TRANSPORT_SHA256,
            "model": "gpt-6-luna",
            "concurrent_transport_sha256": CONCURRENT_SHA256,
            "warm_transport_sha256": WARM_SHA256,
            "transport_kind": "warm_process_fresh_ephemeral_thread",
            "warm_max_requests": args.warm_max_requests,
            "warm_max_age_seconds": args.warm_max_age_seconds})
        report["first_failure"] = safe_first_failure(budget.get("first_failure"), warm)
    except (CampaignStop, Exception) as exc:
        report["stop_code"] = str(exc) if isinstance(exc, CampaignStop) else "campaign_failure"
        if created_output:
            with (output / "private-failure.txt").open("w", encoding="utf-8") as stream:
                traceback.print_exception(exc, file=stream)
            os.chmod(output / "private-failure.txt", 0o600)
    if created_output:
        try:
            saved = json.loads((output / "private-progress.json").read_text(encoding="utf-8"))
            ledger = saved.get("budget") or {}
            report.setdefault("canary_passed", (saved.get("canary") or {}).get("passed") is True)
            report.setdefault("questions_selected", len(saved.get("questions") or []))
            report.setdefault("questions_completed", sum(
                isinstance(item, dict) and item.get("question_completed") is True
                for item in saved.get("questions") or []))
            report.setdefault("turns", ledger.get("turns"))
            report.setdefault("known_tokens", ledger.get("known_tokens"))
            report.setdefault("known_tokens_scope", ledger.get("known_tokens_scope"))
            report.setdefault("in_flight", ledger.get("in_flight"))
            report.setdefault("active_invocations", saved.get("active_invocations"))
            report.setdefault("usage_complete", saved.get("usage_complete_now"))
            if report["first_failure"] is None and "warm" in locals():
                report["first_failure"] = safe_first_failure(
                    ledger.get("first_failure"), warm)
        except (OSError, ValueError, TypeError):
            report["ok"] = False
            report["stop_code"] = report["stop_code"] or "private_progress_unavailable"
    if not report["ok"] and report["stop_code"] is None:
        report["stop_code"] = "questions_incomplete"
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
