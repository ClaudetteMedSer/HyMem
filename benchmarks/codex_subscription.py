"""Experimental ChatGPT Codex transport; deliberately not an API protocol adapter.

Inference remains explicitly gated until runtime isolation is independently
verified. No credentials are read and no provider HTTP counts are invented.
"""
from __future__ import annotations

import json
from collections import deque
import math
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
from typing import Any

from hymem.extraction.llm import LLMRequest

MODEL = "gpt-6-luna"
VERSION = "0.158.0"
MAX_BENIGN_WARNINGS = 16
_CODE_MODE_DISABLED_NOTICE = (
    "Code Mode is unavailable because code-mode host is disabled. "
    "Code mode will fail closed; enable `features.code_mode_host` and install "
    "`codex-code-mode-host`."
)
MAX_TURNS = 1200
MAX_TOKENS = 4_000_000
MAX_ELAPSED_SECONDS = 90 * 60
MAX_EVENTS = 4096
MAX_OUTPUT_CHARS = 1_000_000
SUBSCRIPTION_PLANS = frozenset({"plus", "pro", "prolite", "free", "go", "team"})
DISABLED_FEATURES = (
    "apps", "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "code_mode", "code_mode_host", "computer_use", "goals", "hooks",
    "image_generation", "memories", "multi_agent", "multi_agent_v2", "plugins",
    "remote_plugin", "shell_snapshot", "shell_tool", "skill_mcp_dependency_install",
    "skill_search", "sleep_tool", "tool_suggest", "unified_exec", "view_image",
    "workspace_dependencies", "unbounded_connection_retries",
)
OVERRIDES = {
    "forced_login_method": "chatgpt", "model_provider": "openai", "model": MODEL,
    "model_reasoning_effort": "low", "web_search": "disabled",
    "project_doc_max_bytes": 0,
    "memories.use_memories": False, "memories.generate_memories": False,
    "developer_instructions": "",
    "service_tier": "default", "features.skip_host_skill_discovery": True,
    **{f"features.{name}": False for name in DISABLED_FEATURES},
}


class SubscriptionTransportError(RuntimeError):
    """A fixed safe code, never a server error or prompt."""


def _fail(code: str) -> None:
    raise SubscriptionTransportError(code)


def _safe_method(value: Any) -> str:
    if isinstance(value, str) and len(value) <= 96 and re.fullmatch(r"[A-Za-z][A-Za-z0-9_/]*", value):
        return value
    return "invalid_method"


def _validate_warning(event: dict[str, Any], thread_id: str | None) -> str | None:
    """Accept only two observed 0.158.0 fail-closed advisory notices.

    Their presence does not prove the literal tool inventory is empty.
    """
    params = event.get("params")
    if not isinstance(params, dict) or set(params) != {"message", "threadId"}:
        _fail("warning_unapproved")
    if params.get("message") != _CODE_MODE_DISABLED_NOTICE:
        config_home = os.environ.get("CODEX_HOME") or os.path.join(os.environ.get("HOME", ""), ".codex")
        config_path = os.path.join(config_home, "config.toml")
        if (not os.path.isabs(config_path) or len(config_path) > 512
                or any(ord(char) < 32 or ord(char) >= 127 for char in config_path)):
            _fail("warning_unapproved")
        expected = ("Under-development features enabled: skip_host_skill_discovery. "
            "Under-development features are incomplete and may behave unpredictably. "
            "To suppress this warning, set `suppress_unstable_features_warning = true` "
            f"in {config_path}.")
        if params.get("message") != expected:
            _fail("warning_unapproved")
    target = params.get("threadId")
    if not isinstance(target, str) or not target:
        _fail("warning_thread_unverified")
    if thread_id is not None and target != thread_id:
        _fail("warning_thread_unverified")
    return target


def sanitized_environment(source: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if source is None else source
    return {k: source[k] for k in ("HOME", "PATH", "TMPDIR", "CODEX_HOME") if k in source}


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        _fail("invalid_quota")
    return float(value)


def quota_metadata(response: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = response.get("rateLimitsByLimitId")
    if isinstance(snapshots, dict) and snapshots:
        snapshots = list(snapshots.values())
    else:
        snapshots = [response.get("rateLimits")]
    windows = []
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            _fail("unknown_quota")
        if snapshot.get("planType") is not None and snapshot["planType"] not in SUBSCRIPTION_PLANS:
            _fail("subscription_plan_unverified")
        if snapshot.get("spendControlReached") is True or snapshot.get("rateLimitReachedType") is not None:
            _fail("quota_exhausted")
        credits = snapshot.get("credits")
        if isinstance(credits, dict) and credits.get("hasCredits") is True:
            _fail("credit_balance_present")
        for key in ("primary", "secondary"):
            window = snapshot.get(key)
            if window is None:
                continue
            used = _number(window.get("usedPercent"))
            if not 0 <= used <= 100:
                _fail("invalid_quota")
            windows.append({"remaining_percent": 100 - used,
                            "window_minutes": None if window.get("windowDurationMins") is None else _number(window["windowDurationMins"]),
                            "resets_at": None if window.get("resetsAt") is None else _number(window["resetsAt"])})
        individual = snapshot.get("individualLimit")
        if individual is not None:
            if not isinstance(individual, dict):
                _fail("invalid_quota")
            remaining = _number(individual.get("remainingPercent"))
            if not 0 <= remaining <= 100:
                _fail("invalid_quota")
            windows.append({"remaining_percent": remaining, "window_minutes": None,
                            "resets_at": _number(individual.get("resetsAt"))})
    if not windows:
        _fail("unknown_quota")
    if any(w["remaining_percent"] < 25 for w in windows):
        _fail("quota_floor")
    return windows


def _validate_account_notification(event: dict[str, Any]) -> None:
    params = event.get("params")
    if not isinstance(params, dict):
        _fail("account_notification_invalid")
    if event.get("method") == "account/updated":
        if params.get("authMode") != "chatgpt" or params.get("planType") not in SUBSCRIPTION_PLANS:
            _fail("account_changed")
    elif event.get("method") == "account/rateLimits/updated":
        quota_metadata({"rateLimits": params.get("rateLimits")})
    else:
        _fail("account_notification_invalid")


class StdioSession:
    """One bounded process; stderr is discarded to avoid leaking config/secrets."""
    def __init__(self, binary: str, cwd: str, timeout: float = 120):
        self.deadline = time.monotonic() + timeout
        try:
            version = subprocess.run([binary, "--version"], env=sanitized_environment(),
                cwd=cwd, capture_output=True, text=True, timeout=5, check=True)
        except (OSError, ValueError, subprocess.SubprocessError):
            _fail("binary_version_unverified")
        if version.stdout.strip() != f"codex-cli {VERSION}":
            _fail("binary_version_mismatch")
        args = [binary, "--strict-config"]
        for key, value in OVERRIDES.items():
            args += ["-c", f"{key}={json.dumps(value)}"]
        args += ["app-server", "--listen", "stdio://"]
        self.next_id = 0
        self.stage = "startup"
        self.events: queue.Queue[Any] = queue.Queue(maxsize=4096)
        self.pending: deque[dict[str, Any]] = deque()
        self.warning_targets: list[str] = []
        self.bound_thread_id: str | None = None
        try:
            self.process = subprocess.Popen(args, cwd=cwd, env=sanitized_environment(),
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, start_new_session=True)
        except (OSError, ValueError):
            _fail("startup_failed")
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        try:
            while line := self.process.stdout.readline(8_000_001):
                if len(line) > 8_000_000:
                    break
                self.events.put(json.loads(line), timeout=1)
        except (ValueError, queue.Full):
            pass
        try:
            self.events.put(None, timeout=1)
        except queue.Full:
            pass

    def receive(self) -> dict[str, Any]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            _fail("timeout")
        try:
            event = self.events.get(timeout=remaining)
        except queue.Empty:
            _fail("timeout")
        if not isinstance(event, dict):
            _fail("process_exit:" + self.stage if self.process.poll() is not None else "protocol_failure:" + self.stage)
        if "id" in event and "method" in event:
            _fail("unexpected_server_request")
        return event

    def send(self, method: str, params: dict[str, Any], *, notification: bool = False) -> int:
        self.next_id += 1
        message = {"method": method, "params": params}
        if not notification:
            message["id"] = self.next_id
        try:
            self.process.stdin.write(json.dumps(message) + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError):
            _fail("protocol_failure")
        return self.next_id

    def rpc(self, method: str, params: dict[str, Any], *, preserve_notifications: bool = False) -> dict[str, Any]:
        self.stage = method
        request_id = self.send(method, params)
        for _ in range(4096):
            event = self.receive()
            if event.get("id") == request_id:
                if "error" in event or not isinstance(event.get("result"), dict):
                    _fail("rpc_failure:" + method)
                return event["result"]
            if "id" in event:
                _fail("unexpected_response")
            if event.get("method") == "remoteControl/status/changed":
                if event.get("params", {}).get("status") != "disabled":
                    _fail("remote_control_enabled")
                continue
            if event.get("method") == "warning":
                if self.bound_thread_id is None and method != "thread/start":
                    _fail("warning_unapproved")
                self.accept_warning(event, self.bound_thread_id)
                continue
            if event.get("method") in {"account/updated", "account/rateLimits/updated"}:
                _validate_account_notification(event)
                continue
            if preserve_notifications:
                self.pending.append(event)
                continue
            if event.get("method") != "thread/started":
                _fail("unexpected_notification:" + _safe_method(event.get("method")))
        _fail("event_limit")

    def next_event(self) -> dict[str, Any]:
        return self.pending.popleft() if self.pending else self.receive()

    def bind_thread_id(self, thread_id: str) -> None:
        if not isinstance(thread_id, str) or not thread_id or any(
                target != thread_id for target in self.warning_targets):
            _fail("warning_thread_unverified")
        self.bound_thread_id = thread_id

    def accept_warning(self, event: dict[str, Any], thread_id: str | None) -> None:
        target = _validate_warning(event, thread_id)
        if len(self.warning_targets) >= MAX_BENIGN_WARNINGS:
            _fail("warning_limit")
        self.warning_targets.append(target)

    def close(self) -> None:
        # Always address the owned group: descendants can survive parent exit.
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
        try:
            os.killpg(self.process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        self.process.wait(timeout=2)
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()


def inspect_preflight(session: Any, *, base_instructions: str = "") -> dict[str, Any]:
    initialized = session.rpc("initialize", {"clientInfo": {"name": "hymem_luna_pilot", "version": "0.1"},
                          "capabilities": {"experimentalApi": True}})
    session.send("initialized", {}, notification=True)
    account = session.rpc("account/read", {"refreshToken": False}).get("account")
    if not isinstance(account, dict) or account.get("type") != "chatgpt":
        _fail("subscription_auth_required")
    if account.get("planType") not in SUBSCRIPTION_PLANS:
        _fail("subscription_plan_unverified")
    catalog = session.rpc("model/list", {"includeHidden": True})
    if catalog.get("nextCursor") is not None:
        _fail("incomplete_model_catalog")
    models = [m for m in catalog.get("data", []) if m.get("model") == MODEL]
    if len(models) != 1 or not any(e.get("reasoningEffort") == "low" for e in models[0].get("supportedReasoningEfforts", [])):
        _fail("exact_model_unavailable")
    windows = quota_metadata(session.rpc("account/rateLimits/read", {}))
    config = session.rpc("config/read", {"includeLayers": False}).get("config")
    if not isinstance(config, dict) or config.get("forced_login_method") != "chatgpt" or config.get("model_provider") != "openai":
        _fail("routing_unverified")
    if config.get("service_tier") not in {None, "default"}:
        _fail("service_tier_unverified")
    if config.get("web_search") != "disabled" or config.get("model") != MODEL:
        _fail("model_or_search_unverified")
    memories = config.get("memories")
    if (config.get("project_doc_max_bytes") != 0
            or not isinstance(memories, dict)
            or memories.get("use_memories") is not False
            or memories.get("generate_memories") is not False
            or config.get("developer_instructions") not in {None, ""}):
        _fail("instruction_isolation_unverified")
    if (config.get("chatgpt_base_url") not in {None, "", "https://chatgpt.com/backend-api", "https://chatgpt.com/backend-api/"}
            or config.get("openai_base_url") not in {None, ""}
            or config.get("model_providers", {}).get("openai")):
        _fail("provider_endpoint_override")
    # ConfigRead's projection may omit feature/MCP entries. Omission is not proof
    # of disablement; do not manufacture an isolation attestation from flags.
    features = config.get("features")
    if (not isinstance(features, dict) or any(features.get(n) is not False for n in DISABLED_FEATURES)
            or features.get("skip_host_skill_discovery") is not True):
        _fail("capability_isolation_unverified")
    servers = config.get("mcp_servers", {})
    if not isinstance(servers, dict) or any(s.get("enabled") is not False for s in servers.values()):
        _fail("mcp_isolation_unverified")
    thread = session.rpc("thread/start", {"model": MODEL, "modelProvider": "openai",
        "allowProviderModelFallback": False, "ephemeral": True,
        "environments": [], "runtimeWorkspaceRoots": [], "selectedCapabilityRoots": [],
        "dynamicTools": [], "baseInstructions": base_instructions, "developerInstructions": "",
        "approvalPolicy": "never", "sandbox": "read-only"})
    t = thread.get("thread", {})
    sandbox = thread.get("sandbox")
    if (thread.get("model") != MODEL or thread.get("modelProvider") != "openai"
            or thread.get("runtimeWorkspaceRoots") != [] or thread.get("instructionSources") != []
            or thread.get("approvalPolicy") != "never" or thread.get("reasoningEffort") != "low"
            or thread.get("serviceTier") not in {None, "default"}
            or not isinstance(sandbox, dict) or sandbox.get("type") != "readOnly"
            or sandbox.get("networkAccess") is not False
            or t.get("ephemeral") is not True or t.get("environments") != []
            or t.get("turns") != [] or t.get("path") is not None):
        _fail("thread_isolation_unverified")
    bind_thread = getattr(session, "bind_thread_id", None)
    if callable(bind_thread):
        bind_thread(t.get("id"))
    # The response echoes empty environments, but not the resolved tool
    # inventory. An independent runtime probe remains necessary.
    return {"auth": "chatgpt", "model": MODEL, "quota_windows": windows,
            "observed_turns": 0, "internal_http_attempts": None,
            "inference_enabled": False, "config_isolation_admitted": True,
            "runtime_probe_verified": False, "tool_inventory_observable": False,
            "_thread_id": t.get("id")}


def _observed_usage(event: dict[str, Any], thread_id: str, turn_id: str) -> int:
    params = event.get("params", {})
    if params.get("threadId") != thread_id or params.get("turnId") != turn_id:
        _fail("usage_identity_mismatch")
    total = params.get("tokenUsage", {}).get("total", {}).get("totalTokens")
    if isinstance(total, bool) or not isinstance(total, int) or total < 0:
        _fail("usage_invalid")
    return total


def _run_turn(session: Any, thread_id: str, user: str) -> tuple[str, int]:
    if not isinstance(thread_id, str) or not thread_id:
        _fail("thread_id_missing")
    params = {"threadId": thread_id, "input": [{"type": "text", "text": user}],
              "model": MODEL, "effort": "low", "environments": [],
              "runtimeWorkspaceRoots": [], "approvalPolicy": "never",
              "serviceTierForTurn": "default",
              "sandboxPolicy": {"type": "readOnly", "networkAccess": False}}
    response = session.rpc("turn/start", params, preserve_notifications=True)
    turn = response.get("turn", {})
    turn_id = turn.get("id")
    if not isinstance(turn_id, str) or not turn_id or turn.get("status") != "inProgress":
        _fail("turn_start_invalid")
    usage: int | None = None
    final: str | None = None
    completed = False
    queued_warning_count = 0
    started_items: dict[str, str] = {}
    for _ in range(MAX_EVENTS):
        event = session.next_event()
        if "id" in event:
            _fail("unexpected_response")
        method = event.get("method")
        data = event.get("params", {})
        if method == "thread/started":
            if data.get("thread", {}).get("id") != thread_id:
                _fail("thread_identity_mismatch")
            continue
        if method == "remoteControl/status/changed":
            if data.get("status") != "disabled":
                _fail("remote_control_enabled")
            continue
        if method == "warning":
            accept = getattr(session, "accept_warning", None)
            if callable(accept):
                accept(event, thread_id)
            else:
                _validate_warning(event, thread_id)
                queued_warning_count += 1
                if queued_warning_count > MAX_BENIGN_WARNINGS:
                    _fail("warning_limit")
            continue
        if method in {"account/updated", "account/rateLimits/updated"}:
            _validate_account_notification(event)
            continue
        if method in {"thread/status/changed", "turn/started"}:
            if method == "thread/status/changed" and data.get("threadId") != thread_id:
                _fail("thread_identity_mismatch")
            if method == "turn/started" and (data.get("threadId") != thread_id or data.get("turn", {}).get("id") != turn_id):
                _fail("turn_identity_mismatch")
            continue
        if method == "thread/tokenUsage/updated":
            updated = _observed_usage(event, thread_id, turn_id)
            if usage is not None and updated < usage:
                _fail("usage_regressed")
            usage = updated
            continue
        if method == "item/agentMessage/delta":
            if (data.get("threadId") != thread_id or data.get("turnId") != turn_id
                    or started_items.get(data.get("itemId")) != "agentMessage"
                    or not isinstance(data.get("delta"), str)):
                _fail("message_delta_invalid")
            # The completed final item is authoritative; streamed chunks are
            # validated but neither logged nor concatenated into output.
            continue
        if method in {"item/reasoning/summaryPartAdded", "item/reasoning/summaryTextDelta", "item/reasoning/textDelta"}:
            index_name = "contentIndex" if method == "item/reasoning/textDelta" else "summaryIndex"
            index = data.get(index_name)
            if (data.get("threadId") != thread_id or data.get("turnId") != turn_id
                    or started_items.get(data.get("itemId")) != "reasoning"
                    or isinstance(index, bool) or not isinstance(index, int) or index < 0
                    or method != "item/reasoning/summaryPartAdded" and not isinstance(data.get("delta"), str)):
                _fail("reasoning_delta_invalid")
            continue
        if method in {"item/started", "item/completed"}:
            if data.get("threadId") != thread_id or data.get("turnId") != turn_id:
                _fail("item_identity_mismatch")
            item = data.get("item", {})
            kind = item.get("type")
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id:
                _fail("item_id_invalid")
            if kind not in {"agentMessage", "reasoning", "userMessage"}:
                _fail("tool_or_extra_item")
            if method == "item/started":
                if item_id in started_items:
                    _fail("duplicate_item")
                started_items[item_id] = kind
            elif started_items.get(item_id) != kind:
                _fail("item_lifecycle_invalid")
            if kind == "agentMessage" and method == "item/completed":
                if item.get("phase") != "final_answer" or final is not None or not isinstance(item.get("text"), str):
                    _fail("final_message_invalid")
                final = item["text"]
                if len(final) > MAX_OUTPUT_CHARS:
                    _fail("output_limit")
            continue
        if method == "turn/completed":
            if data.get("threadId") != thread_id or data.get("turn", {}).get("id") != turn_id:
                _fail("turn_identity_mismatch")
            if data["turn"].get("status") != "completed" or completed:
                _fail("turn_failed")
            completed = True
            break
        _fail("unexpected_notification:" + _safe_method(method))
    if not completed or final is None or usage is None or usage <= 0:
        _fail("incomplete_turn_or_usage")
    return final, usage


class CodexSubscriptionClient:
    def __init__(self, binary: str, *, session_factory: Any = StdioSession,
                 inference_accepted: bool = False):
        self.binary = binary
        self.session_factory = session_factory
        self.observed_turns = 0
        self.internal_http_attempts = None
        self.observed_tokens = None
        self.stopped = False
        self._flight = threading.Lock()
        self.usage_complete = True
        self.inference_accepted = inference_accepted
        self.started_at = time.monotonic()
        self.requested_controls: list[dict[str, Any]] = []

    def preflight(self) -> dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="hymem-luna-empty-") as cwd:
            session = self.session_factory(self.binary, cwd)
            try:
                result = inspect_preflight(session)
                result.pop("_thread_id", None)
                return result
            finally:
                session.close()

    def complete(self, request: LLMRequest) -> str:
        if not self._flight.acquire(blocking=False):
            _fail("concurrent_completion_rejected")
        try:
            return self._complete_locked(request)
        finally:
            self._flight.release()

    def _complete_locked(self, request: LLMRequest) -> str:
        if self.stopped:
            _fail("pilot_stopped")
        if not self.inference_accepted:
            self.stopped = True
            _fail("runtime_isolation_not_accepted")
        if (self.observed_turns >= MAX_TURNS or self.observed_tokens is None and self.observed_turns > 0
                or not self.usage_complete
                or self.observed_tokens is not None and self.observed_tokens >= MAX_TOKENS
                or time.monotonic() - self.started_at >= MAX_ELAPSED_SECONDS):
            self.stopped = True
            _fail("pilot_budget_exhausted")
        if not isinstance(request.system, str) or not isinstance(request.user, str):
            self.stopped = True
            _fail("invalid_request")
        self.requested_controls.append({"temperature_requested": request.temperature,
            "max_tokens_requested": request.max_tokens, "temperature_effective": None,
            "max_tokens_effective": None, "response_format_requested": request.response_format,
            "response_format_effective": None})
        # One app-server and one ephemeral thread per request; no history reuse.
        turn_started = False
        usage_recorded = False
        try:
            with tempfile.TemporaryDirectory(prefix="hymem-luna-empty-") as cwd:
                remaining = MAX_ELAPSED_SECONDS - (time.monotonic() - self.started_at)
                if remaining <= 0:
                    _fail("pilot_budget_exhausted")
                session = self.session_factory(self.binary, cwd, timeout=min(120, remaining))
                try:
                    admission = inspect_preflight(session, base_instructions=request.system)
                    thread_id = admission.get("_thread_id")
                    if not thread_id:
                        _fail("thread_id_missing")
                    self.observed_turns += 1  # count the turn even if transport fails
                    turn_started = True
                    answer, used = _run_turn(session, thread_id, request.user)
                    self.observed_tokens = (self.observed_tokens or 0) + used
                    usage_recorded = True
                    return answer
                finally:
                    session.close()
        except SubscriptionTransportError:
            self.stopped = True
            if turn_started and not usage_recorded:
                self.usage_complete = False  # preserve known cumulative usage
            raise
        except Exception:
            self.stopped = True
            if turn_started and not usage_recorded:
                self.usage_complete = False
            _fail("transport_failure")

    def chat(self, messages: list[dict[str, str]], **controls: Any) -> str:
        if len(messages) != 2 or [m.get("role") for m in messages] != ["system", "user"]:
            _fail("unsupported_chat_history")
        if set(controls) - {"max_tokens", "temperature", "model", "response_format"}:
            _fail("unsupported_chat_control")
        if controls.get("model", MODEL) != MODEL or controls.get("response_format", "text") not in {"text", "json"}:
            _fail("unsupported_chat_control")
        return self.complete(LLMRequest(system=messages[0]["content"], user=messages[1]["content"],
            response_format=controls.get("response_format", "text"),
            max_tokens=controls.get("max_tokens", 1024), temperature=controls.get("temperature", 0.0)))
