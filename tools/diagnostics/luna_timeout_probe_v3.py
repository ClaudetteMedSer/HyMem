"""Source-only preparation and bounded invented-text Luna timeout probe.

The CLI can only prepare or verify a local source bundle. A future, separately
reviewed host wrapper may call ``load_prepared`` and ``run_probe`` after proving
live containment. This module has no launch, paid-run, or resume CLI action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import types
from typing import Any, Callable


SCHEMA = "luna-timeout-probe-v3"
SOURCE_PINS = {
    "benchmarks/codex_subscription.py": "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491",
    "benchmarks/codex_subscription_concurrent_v2.py": "cc2a8a4d8c528221747d51be939fe6dacfd581a8e8923125f3b2fe03c97878f0",
    "benchmarks/codex_subscription_warm_v2.py": "9dfab3fab7015e080a7116d07542bf839eaf40e4ec0b4721734d198b40926593",
    "benchmarks/codex_subscription_warm_v3.py": "0142435207e8d93ffc88e67b9444e58139b7385a940ce295cecae44ae18d5a2d",
    "benchmarks/codex_subscription_warm_v4.py": "43611c5b7c9b2242f8216daf1cb274c7b1f82c7c633abd11830bf5759fdb4138",
    "benchmarks/codex_subscription_warm_v5.py": "2df1ead8f6f1cee1f138075aa77d78c61ed28959195a7634ce59a0df00290702",
    "benchmarks/codex_subscription_warm_v6.py": "98422aa251ca9482a79be5851d48ae54decc17f17b6aa1149bd6b93b618b784b",
    "benchmarks/codex_subscription_warm_v7.py": "94234eca1daeb542a8d8f92a5b7178ba8b91060f33415f76343bd13e6d5953d9",
    "benchmarks/codex_subscription_warm_v8.py": "0a55d44053349eb90511a597dae20f295c19bb53c734a78b5db3c41197343c21",
    "benchmarks/codex_subscription_warm_v9.py": "f9081dda08a6e1975a3e951190ae6580161d89817303a1b75d0ba21982fcd470",
    "benchmarks/codex_subscription_timeout_v2.py": "476e00bcae40c0a061a1b75b10797d4563fb9822012917382e27d9b5a6c93287",
    "hymem/contrib/implementation_identity.py": "cbf18f140cb13f42f0cd3ce9975dd9ac70aab4f482891cff80f9aa68937d79d4",
    "hymem/extraction/llm.py": "c10e2187bc17be70800cf7d0a698d6b5bec790848c375e25faf38f3c89762fe2",
}
SELF_RELATIVE = "tools/diagnostics/luna_timeout_probe_v3.py"
TURN_LIMIT = 16
WORKERS = 1
PER_WORKER = 16
TOKEN_LIMIT = 160_000
WALL_SECONDS = 600
INVOCATION_SECONDS = 120
SYSTEM = "Answer each invented item in one short sentence."
USERS = tuple(f"Invented timing item {index:02d}: describe a {color} paper shape."
    for index, color in enumerate(("blue", "amber", "green", "violet") * 4))
FIXTURE_SHA256 = hashlib.sha256(json.dumps({"system": SYSTEM, "users": USERS},
    sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
LIMITS = {"turns": TURN_LIMIT, "known_tokens": TOKEN_LIMIT, "seconds": WALL_SECONDS,
          "workers": WORKERS, "calls_per_worker": PER_WORKER,
          "invocation_seconds": INVOCATION_SECONDS}
POLICY = {"model": "gpt-6-luna", "effort": "low", "auth": "chatgpt",
          "billing": "included_allowance_or_existing_finite_positive_credits_per_window_v2",
          "notifications": "agent_message_delta_optout_v1",
          "token_ceiling": "stop_before_next_observed_usage", "rerolls": 0}
PENDING_HOST = {"model_calls": 0, "host_admission": "pending",
    "one_shot_host_receipt": "pending", "service_runtime_seconds": 730,
    "service_stop_seconds": 10, "tasks_max": 256, "memory_max_bytes": 4_294_967_296,
    "cpu_percent": 200, "recursive_cleanup_verification": "pending"}
BINARY_PATH = "/home/atta/.codex/packages/standalone/releases/0.158.0-x86_64-unknown-linux-musl/bin/codex"
BINARY_SHA256 = "167c0148a849d2444f1b5a7fb5f8bb2de1de5ae13a2a504b833fc765980f5cd9"


def _require(ok: bool, code: str) -> None:
    if not ok:
        raise ValueError(code)


def _strict_match(value: Any, expected: Any) -> bool:
    if type(value) is not type(expected):
        return False
    if type(expected) is dict:
        return (set(value) == set(expected) and
                all(_strict_match(value[key], item) for key, item in expected.items()))
    if type(expected) in (list, tuple):
        return (len(value) == len(expected) and
                all(_strict_match(item, want) for item, want in zip(value, expected)))
    return value == expected


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _regular(path: Path) -> bool:
    try:
        return stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink()
    except OSError:
        return False


def _write_once(path: Path, value: dict[str, Any]) -> None:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def _sources(root: Path) -> dict[str, str]:
    source = {**SOURCE_PINS, SELF_RELATIVE: _sha(root / SELF_RELATIVE)}
    _require(source[SELF_RELATIVE] == _sha(Path(__file__)), "self_source_drift")
    for relative, digest in source.items():
        path = root / relative
        _require(_regular(path) and _sha(path) == digest, "source_drift")
    return source


def prepare(output: Path, *, source_root: Path | None = None) -> dict[str, Any]:
    """Create a fresh local source-only root; make no import or model call."""
    source_root = source_root or Path(__file__).resolve().parents[2]
    _require(output.is_absolute() and not output.exists() and not output.is_symlink()
             and output.parent.is_dir() and not output.parent.is_symlink(), "output_invalid")
    source = _sources(source_root)
    output.mkdir(mode=0o700)
    code = output / "code"
    code.mkdir(mode=0o700)
    for relative, digest in source.items():
        target = code / relative
        target.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        shutil.copyfile(source_root / relative, target, follow_symlinks=False)
        target.chmod(0o600)
        _require(_sha(target) == digest, "copy_drift")
    receipt = {"schema": SCHEMA, "kind": "local_source_only",
        "source_sha256": source, "fixture_sha256": FIXTURE_SHA256,
        "limits": LIMITS, "policy": POLICY, "binary_path": BINARY_PATH,
        "binary_sha256": BINARY_SHA256, "pending_host": PENDING_HOST}
    _write_once(output / "local-preparation-receipt.json", receipt)
    return {"schema": SCHEMA, "prepared": True, "model_calls": 0,
            "receipt_sha256": _sha(output / "local-preparation-receipt.json"),
            "source_files": len(source), "host_launch_authorized": False}


def verify_prepared(root: Path, receipt_sha256: str) -> dict[str, Any]:
    _require(root.is_absolute() and root.is_dir() and not root.is_symlink()
             and stat.S_IMODE(root.stat().st_mode) == 0o700, "root_invalid")
    receipt_path = root / "local-preparation-receipt.json"
    _require(_regular(receipt_path), "receipt_drift")
    _require(receipt_path.stat().st_size <= 16_384, "receipt_too_large")
    _require(_sha(receipt_path) == receipt_sha256, "receipt_drift")
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    source = receipt.get("source_sha256") if type(receipt) is dict else None
    _require(type(source) is dict and set(source) == set(SOURCE_PINS) | {SELF_RELATIVE}
             and all(type(x) is str and len(x) == 64 for x in source.values()),
             "receipt_sources_invalid")
    _require(_strict_match(receipt, {"schema": SCHEMA, "kind": "local_source_only",
        "source_sha256": source, "fixture_sha256": FIXTURE_SHA256,
        "limits": LIMITS, "policy": POLICY, "binary_path": BINARY_PATH,
        "binary_sha256": BINARY_SHA256, "pending_host": PENDING_HOST})
        and all(source[name] == digest for name, digest in SOURCE_PINS.items()),
        "receipt_invalid")
    _require(source[SELF_RELATIVE] == _sha(Path(__file__)), "self_source_drift")
    expected_files = {"local-preparation-receipt.json"} | {"code/" + name for name in source}
    expected_dirs = {"code"}
    for name in source:
        parent = (Path("code") / name).parent
        while parent != Path("."):
            expected_dirs.add(parent.as_posix())
            parent = parent.parent
    found_files = set()
    found_dirs = set()
    for path in root.rglob("*"):
        kind = path.lstat().st_mode
        _require(not stat.S_ISLNK(kind), "source_symlink")
        if stat.S_ISREG(kind):
            found_files.add(path.relative_to(root).as_posix())
        elif stat.S_ISDIR(kind):
            found_dirs.add(path.relative_to(root).as_posix())
        else:
            _require(False, "nonregular_entry")
    _require(found_files == expected_files and found_dirs == expected_dirs,
             "extra_or_missing_file")
    for relative, digest in source.items():
        path = root / "code" / relative
        _require(_regular(path) and _sha(path) == digest, "source_drift")
    return receipt


def _load_source(name: str, path: Path, expected_sha256: str) -> Any:
    _require(_regular(path), "module_source_invalid")
    source = path.read_bytes()
    _require(hashlib.sha256(source).hexdigest() == expected_sha256,
             "module_source_drift")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = name.rpartition(".")[0]
    sys.modules[name] = module
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    _require(Path(module.__file__).resolve() == path.resolve(), "module_origin_invalid")
    return module


def load_prepared(root: Path, receipt_sha256: str) -> tuple[Any, Any]:
    """Import only the verified closure in a fresh isolated Python process."""
    receipt = verify_prepared(root, receipt_sha256)
    code = root / "code"
    _require(not any(name == "hymem" or name.startswith("hymem.") or
                     name == "benchmarks" or name.startswith("benchmarks.")
                     for name in sys.modules), "ambient_module_present")
    for name, part in (("hymem", "hymem"), ("hymem.contrib", "hymem/contrib"),
                       ("hymem.extraction", "hymem/extraction")):
        package = types.ModuleType(name)
        package.__path__ = [str(code / part)]
        package.__file__ = None
        sys.modules[name] = package
    identity = _load_source("hymem.contrib.implementation_identity",
        code / "hymem/contrib/implementation_identity.py",
        receipt["source_sha256"]["hymem/contrib/implementation_identity.py"])
    llm = _load_source("hymem.extraction.llm", code / "hymem/extraction/llm.py",
        receipt["source_sha256"]["hymem/extraction/llm.py"])
    observer = _load_source("pinned_luna_timeout_probe_observer",
        code / "benchmarks/codex_subscription_timeout_v2.py",
        receipt["source_sha256"]["benchmarks/codex_subscription_timeout_v2.py"])
    _require(observer.PINNED_WARM_V9_SHA256 == SOURCE_PINS["benchmarks/codex_subscription_warm_v9.py"]
             and observer.warm.BILLING_POLICY == POLICY["billing"]
             and observer.warm.NOTIFICATION_POLICY == POLICY["notifications"]
             and observer.base.MODEL == POLICY["model"]
             and observer.base.OVERRIDES["model_reasoning_effort"] == POLICY["effort"]
             and observer.base.OVERRIDES["forced_login_method"] == POLICY["auth"],
             "policy_binding_invalid")
    origins = {"hymem/contrib/implementation_identity.py": identity,
        "hymem/extraction/llm.py": llm,
        "benchmarks/codex_subscription_timeout_v2.py": observer,
        "benchmarks/codex_subscription_warm_v9.py": observer.warm,
        "benchmarks/codex_subscription_warm_v8.py": observer.warm.v8,
        "benchmarks/codex_subscription_warm_v7.py": observer.warm.v8.v7,
        "benchmarks/codex_subscription_warm_v6.py": observer.warm.v8.v7.v6,
        "benchmarks/codex_subscription_warm_v5.py": observer.warm.v8.v7.v6.v5,
        "benchmarks/codex_subscription_warm_v4.py": observer.warm.v8.v7.v6.v5.v4,
        "benchmarks/codex_subscription_warm_v3.py": observer.warm.v8.v7.v6.v5.v4.v3,
        "benchmarks/codex_subscription_warm_v2.py": observer.warm.v8.v7.v6.v5.v4.v3.v2,
        "benchmarks/codex_subscription_concurrent_v2.py": observer.warm.concurrent,
        "benchmarks/codex_subscription.py": observer.base}
    for relative, module in origins.items():
        _require(Path(module.__file__).resolve() == (code / relative).resolve()
                 and _sha(Path(module.__file__)) == receipt["source_sha256"][relative],
                 "module_origin_invalid")
    return observer, llm.LLMRequest


def _attestation(value: Any) -> bool:
    return (type(value) is dict and set(value) == {"containment", "denials", "oom"}
        and value["containment"] is True and type(value["denials"]) is int
        and value["denials"] == 0 and type(value["oom"]) is int and value["oom"] == 0)


def _project_failure(observer: Any, state: dict[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    raw = state.get("first_failure")
    if raw is None:
        return None, False
    try:
        projected = observer.warm.serialize_failure(raw)
        if projected is not None and observer.warm.serialize_failure(projected) == projected:
            return projected, False
    except Exception:
        pass
    return None, True


def _client_counts(client: Any) -> dict[str, int | None]:
    """Read only finite accepted-client counters, with unknown kept distinct from zero."""
    names = ("processes_started", "rotations", "cold_calls", "warm_calls",
             "requests_on_process")
    if client is None:
        return {name: None for name in names}
    result: dict[str, int | None] = {}
    for name in names:
        try:
            value = getattr(client, name)
            result[name] = value if type(value) is int and 0 <= value <= TURN_LIMIT else None
        except BaseException:
            result[name] = None
    return result


def run_probe(observer: Any, request_type: Any, binary: str, *,
              attest: Callable[[str, int | None], dict[str, Any]] | None = None,
              client_factory: Any = None) -> dict[str, Any]:
    """Run the fixed sixteen calls on one worker under a caller's live gate."""
    _require(callable(attest) and type(binary) is str and bool(binary),
             "containment_attestor_required")
    if client_factory is None:
        path = Path(binary)
        _require(binary == BINARY_PATH and _regular(path) and _sha(path) == BINARY_SHA256,
                 "binary_binding_invalid")
    _require(len(USERS) == TURN_LIMIT and TURN_LIMIT == WORKERS * PER_WORKER,
             "fixture_invalid")
    budget = observer.SharedBudget(observer.BudgetLimits(TURN_LIMIT, TOKEN_LIMIT, WALL_SECONDS),
        max_in_flight=WORKERS)
    factory = client_factory or observer.TimeoutSubscriptionClient
    client: Any = None
    slots: list[dict[str, Any]] = [
        {"record_id": i, "worker": 0, "status": "not_attempted",
         "observation": None, "sequence_position": None, "identity_checked": False}
        for i in range(TURN_LIMIT)]
    runner_fault: str | None = None
    coverage_fault: str | None = None
    cleanup_ok = True
    observation_fault = False
    identity_checks = 0
    first_session: Any = None
    first_process: Any = None
    first_pid: int | None = None

    def stop(code: str) -> None:
        nonlocal runner_fault
        if runner_fault is None:
            runner_fault = code
        budget.halt(code)

    def observe_success(index: int) -> None:
        """Compare private live identities and counters before any final close."""
        nonlocal first_session, first_process, first_pid, identity_checks, coverage_fault
        counts = _client_counts(client)
        position = counts["requests_on_process"]
        slots[index]["sequence_position"] = position if type(position) is int and 1 <= position <= TURN_LIMIT else None
        try:
            session = client.session
            process = session.process
            pid = process.pid
            alive = process.poll() is None
            identity_available = (session is not None and process is not None
                and type(pid) is int and pid > 0 and alive)
        except BaseException:
            identity_available = False
            session = process = None
            pid = None
        if identity_available and first_session is None:
            first_session, first_process, first_pid = session, process, pid
        same_identity = bool(identity_available and session is first_session
            and process is first_process and pid == first_pid)
        slots[index]["identity_checked"] = same_identity
        identity_checks += int(same_identity)
        if any(value is None for value in counts.values()):
            coverage_fault = "counter_invalid"
            stop("lifecycle_metadata_invalid")
        elif (counts["processes_started"] != 1 or counts["rotations"] != 0
                or counts["cold_calls"] != 1):
            coverage_fault = "rotation" if (
                counts["processes_started"] == 2
                and counts["rotations"] == 1
                and counts["cold_calls"] == 2
                and counts["warm_calls"] == index - 1
                and identity_available and not same_identity
                and position == 1
            ) else "counter_mismatch"
        elif counts["warm_calls"] != index or position != index + 1:
            coverage_fault = "counter_mismatch"
        elif not identity_available:
            coverage_fault = "identity_unverified"
        elif not same_identity:
            coverage_fault = "identity_mismatch"
        if coverage_fault == "rotation":
            stop("sequence_coverage_not_reached")
        elif coverage_fault is not None and coverage_fault != "counter_invalid":
            stop("sequence_integrity_failure")

    try:
        _require(_attestation(attest("initial", None)), "containment_unverified")
        client = factory(binary, budget, "worker-0",
            observer.BudgetLimits(PER_WORKER, TOKEN_LIMIT, WALL_SECONDS),
            max_requests=16, max_age_seconds=300)
        for index in range(TURN_LIMIT):
            if budget.snapshot()["stopped"]:
                break
            try:
                if not _attestation(attest("before_admission", index)):
                    stop("containment_unverified")
                    break
            except BaseException:
                stop("containment_unverified")
                break
            if budget.snapshot()["stopped"]:
                break
            slots[index]["status"] = "attempted"
            try:
                response = client.complete(request_type(SYSTEM, USERS[index],
                    response_format="text", max_tokens=1024, temperature=0.0))
                _require(type(response) is str and bool(response), "empty_response")
                slots[index]["status"] = "returned"
                del response
                observe_success(index)
            except BaseException:
                slots[index]["status"] = "failed"
                stop("invocation_failure")
                break
    except BaseException:
        stop("setup_or_containment_failure")
    finally:
        # A failed invocation may already have cleared the session. Counters survive
        # close in the accepted client; retain their pre-close values privately.
        counts = _client_counts(client)
        if client is not None:
            try:
                client.close()
                cleanup_ok = client.session is None and client.directory is None
            except BaseException:
                cleanup_ok = False
                stop("client_cleanup_failure")
        try:
            terminal_attested = _attestation(attest("terminal", None))
        except BaseException:
            terminal_attested = False
        if not terminal_attested:
            stop("terminal_resource_unverified")

    if client is not None:
        try:
            records = client.diagnostic_records()
            indices = [i for i in range(TURN_LIMIT) if slots[i]["status"] != "not_attempted"]
            _require(type(records) is tuple and len(records) == len(indices),
                     "observation_count_invalid")
            for index, record in zip(indices, records):
                projected = observer._project_record(record)
                _require(projected == record, "observation_invalid")
                slots[index]["observation"] = projected
        except BaseException:
            observation_fault = True
            stop("observation_invalid")

    state = budget.snapshot()
    _require(type(state["turns"]) is int and type(state["known_tokens"]) is int,
             "accounting_invalid")
    first_failure, metadata_fault = _project_failure(observer, state)
    if metadata_fault:
        stop("metadata_invalid")
        state = budget.snapshot()
    overshoot = state["known_tokens"] > TOKEN_LIMIT
    if overshoot:
        stop("known_token_ceiling_exceeded")
        state = budget.snapshot()
    returned = sum(slot["status"] == "returned" for slot in slots)
    failed = sum(slot["status"] == "failed" for slot in slots)
    attempted = returned + failed
    valid_slots = all(slot["observation"] is not None for slot in slots if
                      slot["status"] != "not_attempted")
    accounted = (state["reserved"] == state["in_flight"] == 0
        and state["turns"] <= TURN_LIMIT and state["known_tokens"] >= 0
        and valid_slots)
    positions = [slot["sequence_position"] for slot in slots if slot["status"] == "returned"]
    coverage_complete = (returned == TURN_LIMIT and failed == 0 and
        counts == {"processes_started": 1, "rotations": 0, "cold_calls": 1,
                   "warm_calls": 15, "requests_on_process": 16}
        and positions == list(range(1, TURN_LIMIT + 1))
        and identity_checks == TURN_LIMIT and coverage_fault is None)
    coverage_reason = (None if coverage_complete else
        coverage_fault or "incomplete_sequence")
    clean = (returned == TURN_LIMIT and coverage_complete and accounted
        and state["usage_complete"] is True and not state["stopped"]
        and not overshoot and not metadata_fault and cleanup_ok and terminal_attested
        and runner_fault is None and first_failure is None)
    coverage_only = (coverage_fault == "rotation" and failed == 0
        and runner_fault == "sequence_coverage_not_reached"
        and not observation_fault and not metadata_fault and not overshoot
        and cleanup_ok and terminal_attested and first_failure is None)
    status = ("observed_success" if clean else
              "coverage_not_reached" if coverage_only else "incomplete_or_failed")
    lifecycle = {**counts, "identity_checks": identity_checks,
        "successful_positions": positions, "coverage_complete": coverage_complete,
        "coverage_reason": coverage_reason}
    result = {"schema": SCHEMA, "status": status, "fixture_sha256": FIXTURE_SHA256,
        "limits": dict(LIMITS), "policy": dict(POLICY),
        "attempted": attempted, "returned": returned, "failed": failed,
        "not_attempted": TURN_LIMIT - attempted, "turns": state["turns"],
        "known_tokens": state["known_tokens"], "usage_complete": state["usage_complete"],
        "reserved": state["reserved"], "in_flight": state["in_flight"],
        "budget_stopped": state["stopped"], "budget_stop_code": state["stop_code"],
        "runner_fault": runner_fault, "observation_fault": observation_fault,
        "metadata_fault": metadata_fault, "known_token_overshoot": overshoot,
        "client_cleanup_verified": cleanup_ok, "terminal_attested_by_caller": terminal_attested,
        "independent_recursive_cleanup_verified": None,
        "historical_timeout_cause_proved": False, "lme_readiness_proved": False,
        "first_failure": first_failure, "records": slots, "lifecycle": lifecycle}
    _require(validate_result(result, observer), "result_invalid")
    return result


def validate_result(value: Any, observer: Any) -> bool:
    """Validate finite public metadata and reject invented sequence coverage."""
    try:
        if type(value) is not dict or set(value) != {"schema", "status", "fixture_sha256",
                "limits", "policy", "attempted", "returned", "failed", "not_attempted", "turns",
                "known_tokens", "usage_complete", "reserved", "in_flight", "budget_stopped",
                "budget_stop_code", "runner_fault", "observation_fault", "metadata_fault",
                "known_token_overshoot", "client_cleanup_verified", "terminal_attested_by_caller",
                "independent_recursive_cleanup_verified", "historical_timeout_cause_proved",
                "lme_readiness_proved", "first_failure", "records", "lifecycle"}:
            return False
        if (value["schema"] != SCHEMA or value["status"] not in
                {"observed_success", "coverage_not_reached", "incomplete_or_failed"}
                or value["fixture_sha256"] != FIXTURE_SHA256
                or not _strict_match(value["limits"], LIMITS)
                or not _strict_match(value["policy"], POLICY)
                or type(value["records"]) is not list
                or len(value["records"]) != TURN_LIMIT):
            return False
        for key in ("attempted", "returned", "failed", "not_attempted", "turns", "known_tokens",
                    "reserved", "in_flight"):
            if type(value[key]) is not int or value[key] < 0:
                return False
        for key in ("usage_complete", "budget_stopped", "observation_fault", "metadata_fault",
                    "known_token_overshoot", "client_cleanup_verified",
                    "terminal_attested_by_caller"):
            if type(value[key]) is not bool:
                return False
        if (value["attempted"] + value["not_attempted"] != TURN_LIMIT
                or value["returned"] + value["failed"] != value["attempted"]
                or value["turns"] > value["attempted"] or value["reserved"] != 0
                or value["in_flight"] != 0
                or value["known_token_overshoot"] != (value["known_tokens"] > TOKEN_LIMIT)
                or value["independent_recursive_cleanup_verified"] is not None
                or value["historical_timeout_cause_proved"] is not False
                or value["lme_readiness_proved"] is not False):
            return False
        safe_codes = {None, "containment_unverified", "invocation_failure",
            "sequence_coverage_not_reached", "sequence_integrity_failure",
            "lifecycle_metadata_invalid",
            "setup_or_containment_failure",
            "client_cleanup_failure", "terminal_resource_unverified",
            "observation_invalid", "metadata_invalid", "known_token_ceiling_exceeded"}
        if value["runner_fault"] not in safe_codes:
            return False
        code = value["budget_stop_code"]
        if value["budget_stopped"] != (code is not None):
            return False
        if code is not None and (type(code) is not str or
                observer.warm.serialize_failure({"code": code, "phase": "run",
                    "rpc": "turn/events"})["code"] != code and code not in safe_codes
                and code not in observer.warm.v8.v7.v6.v5.v4.v3.v2._OWN_BUDGET_CODES):
            return False
        fault = value["first_failure"]
        if value["metadata_fault"] and fault is not None:
            return False
        if fault is not None and (type(fault) is not dict or
                observer.warm.serialize_failure(fault) != fault):
            return False
        life = value["lifecycle"]
        counter_keys = {"processes_started", "rotations", "cold_calls", "warm_calls",
                        "requests_on_process"}
        if type(life) is not dict or set(life) != counter_keys | {
                "identity_checks", "successful_positions", "coverage_complete",
                "coverage_reason"}:
            return False
        for key in counter_keys:
            item = life[key]
            if item is not None and (type(item) is not int or not 0 <= item <= TURN_LIMIT):
                return False
        if (type(life["identity_checks"]) is not int
                or not 0 <= life["identity_checks"] <= value["returned"]
                or type(life["successful_positions"]) is not list
                or any(item is not None and
                       (type(item) is not int or not 1 <= item <= TURN_LIMIT)
                       for item in life["successful_positions"])
                or type(life["coverage_complete"]) is not bool
                or life["coverage_reason"] not in {None, "rotation", "identity_mismatch",
                                                   "identity_unverified", "counter_mismatch",
                                                   "counter_invalid",
                                                   "incomplete_sequence"}):
            return False
        if (life["rotations"] is not None and life["processes_started"] is not None
                and life["rotations"] > max(0, life["processes_started"] - 1)):
            return False
        if (life["cold_calls"] is not None and life["processes_started"] is not None
                and life["cold_calls"] != life["processes_started"]):
            return False
        if (life["cold_calls"] is not None and life["warm_calls"] is not None
                and life["cold_calls"] + life["warm_calls"] > value["attempted"]):
            return False
        attempted = returned = failed = checked = 0
        positions: list[int | None] = []
        seen_unattempted = False
        for index, slot in enumerate(value["records"]):
            if type(slot) is not dict or set(slot) != {"record_id", "worker", "status",
                    "observation", "sequence_position", "identity_checked"}:
                return False
            if (type(slot["record_id"]) is not int or slot["record_id"] != index
                    or type(slot["worker"]) is not int or slot["worker"] != 0
                    or slot["status"] not in {"not_attempted", "returned", "failed"}
                    or type(slot["identity_checked"]) is not bool):
                return False
            if slot["status"] == "not_attempted":
                seen_unattempted = True
                if slot["observation"] is not None or slot["sequence_position"] is not None or slot["identity_checked"]:
                    return False
                continue
            if seen_unattempted:
                return False
            attempted += 1
            observation = slot["observation"]
            if observation is None:
                if value["status"] == "observed_success" or not value["observation_fault"]:
                    return False
            elif (type(observation) is not dict or
                  observer._project_record(observation) != observation):
                return False
            if slot["status"] == "returned":
                returned += 1
                position = slot["sequence_position"]
                if position is not None and (type(position) is not int or not 1 <= position <= TURN_LIMIT):
                    return False
                positions.append(position)
                checked += int(slot["identity_checked"])
                if observation is not None and observation["status"] != "success":
                    return False
            else:
                failed += 1
                if (slot["sequence_position"] is not None or slot["identity_checked"]
                        or failed > 1 or index != value["attempted"] - 1):
                    return False
                if observation is not None and observation["status"] != "failure":
                    return False
        if (attempted != value["attempted"] or returned != value["returned"]
                or failed != value["failed"] or life["successful_positions"] != positions
                or life["identity_checks"] != checked):
            return False
        covered = (returned == TURN_LIMIT and failed == 0 and
            all(slot["identity_checked"] for slot in value["records"]) and
            positions == list(range(1, TURN_LIMIT + 1)) and
            all(life[key] == expected for key, expected in {
                "processes_started": 1, "rotations": 0, "cold_calls": 1,
                "warm_calls": 15, "requests_on_process": 16}.items()))
        if life["coverage_complete"] != covered:
            return False
        if covered:
            if life["coverage_reason"] is not None:
                return False
        elif life["coverage_reason"] is None:
            return False
        if value["status"] == "observed_success":
            return (covered and value["turns"] == TURN_LIMIT and
                0 < value["known_tokens"] <= TOKEN_LIMIT and value["usage_complete"]
                and not value["budget_stopped"] and not value["known_token_overshoot"]
                and not value["observation_fault"] and not value["metadata_fault"]
                and value["client_cleanup_verified"] and value["terminal_attested_by_caller"]
                and fault is None and value["runner_fault"] is None)
        if value["status"] == "coverage_not_reached":
            return (not covered and value["failed"] == 0 and
                life["coverage_reason"] == "rotation"
                and value["runner_fault"] == "sequence_coverage_not_reached"
                and value["budget_stopped"] and not value["observation_fault"]
                and not value["metadata_fault"] and not value["known_token_overshoot"]
                and value["client_cleanup_verified"] and value["terminal_attested_by_caller"]
                and fault is None)
        return not (covered and value["turns"] == TURN_LIMIT and
            0 < value["known_tokens"] <= TOKEN_LIMIT and value["usage_complete"]
            and not value["budget_stopped"] and not value["known_token_overshoot"]
            and not value["observation_fault"] and not value["metadata_fault"]
            and value["client_cleanup_verified"] and value["terminal_attested_by_caller"]
            and fault is None and value["runner_fault"] is None)
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-output")
    action.add_argument("--verify-root")
    parser.add_argument("--receipt-sha256")
    args = parser.parse_args(argv)
    try:
        if args.prepare_output:
            _require(args.receipt_sha256 is None, "unexpected_receipt")
            result = prepare(Path(args.prepare_output))
        else:
            _require(type(args.receipt_sha256) is str, "receipt_required")
            receipt = verify_prepared(Path(args.verify_root), args.receipt_sha256)
            result = {"schema": SCHEMA, "verified": True, "model_calls": 0,
                      "source_files": len(receipt["source_sha256"]),
                      "host_launch_authorized": False}
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except BaseException:
        print(json.dumps({"schema": SCHEMA, "verified": False, "model_calls": 0,
            "host_launch_authorized": False}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
