import pytest
import queue

from benchmarks.codex_subscription import (
    CodexSubscriptionClient, DISABLED_FEATURES, MODEL, SubscriptionTransportError,
    inspect_preflight, quota_metadata, sanitized_environment,
    _safe_method,
    StdioSession, VERSION,
    _run_turn,
    _validate_warning,
    _validate_account_notification,
)
from hymem.extraction.llm import LLMRequest, measure_provider_attempts


class FakeSession:
    def __init__(self):
        self.calls = []
        self.responses = {
            "initialize": {"userAgent": "codex/0.158.0"},
            "account/read": {"account": {"type": "chatgpt", "planType": "pro", "email": "private@example.org"}},
            "model/list": {"data": [{"model": MODEL, "supportedReasoningEfforts": [{"reasoningEffort": "low"}]}]},
            "account/rateLimits/read": {"rateLimits": {"primary": {
                "usedPercent": 20, "windowDurationMins": 300, "resetsAt": 1000}}},
            "config/read": {"config": {"forced_login_method": "chatgpt", "model_provider": "openai",
                "model": MODEL, "web_search": "disabled", "project_doc_max_bytes": 0,
                "memories": {"use_memories": False, "generate_memories": False},
                "developer_instructions": "",
                "features": {**dict.fromkeys(DISABLED_FEATURES, False), "skip_host_skill_discovery": True},
                "mcp_servers": {}}},
            "thread/start": {"model": MODEL, "modelProvider": "openai", "runtimeWorkspaceRoots": [],
                "instructionSources": [], "sandbox": {"type": "readOnly", "networkAccess": False},
                "approvalPolicy": "never", "reasoningEffort": "low", "serviceTier": "default",
                "thread": {"ephemeral": True, "environments": [], "turns": [], "path": None}},
        }

    def rpc(self, method, params):
        self.calls.append((method, params))
        return self.responses[method]

    def send(self, method, params, **kwargs):
        self.calls.append((method, params))


def test_environment_has_only_auth_runtime_allowlist():
    assert sanitized_environment({"HOME": "/home/test", "PATH": "/bin", "CODEX_HOME": "/saved",
        "OPENAI_API_KEY": "secret", "OPENAI_BASE_URL": "http://bad", "HTTPS_PROXY": "bad"}) == {
            "HOME": "/home/test", "PATH": "/bin", "CODEX_HOME": "/saved"}


def test_preflight_never_infers_or_exposes_identity():
    fake = FakeSession()
    result = inspect_preflight(fake)
    assert result["observed_turns"] == 0
    assert result["internal_http_attempts"] is None
    assert result["config_isolation_admitted"] is True
    assert result["runtime_probe_verified"] is False
    assert "private" not in str(result)
    assert not any(m == "turn/start" for m, _ in fake.calls)
    params = dict(fake.calls)["thread/start"]
    assert params["environments"] == []
    assert params["allowProviderModelFallback"] is False


@pytest.mark.parametrize("change,code", [
    (lambda f: f.responses["account/read"].update(account={"type": "apiKey"}), "subscription_auth_required"),
    (lambda f: f.responses["model/list"].update(data=[]), "exact_model_unavailable"),
    (lambda f: f.responses["config/read"]["config"].pop("features"), "capability_isolation_unverified"),
    (lambda f: f.responses["config/read"]["config"]["features"].update(shell_tool=True), "capability_isolation_unverified"),
    (lambda f: f.responses["config/read"]["config"].update(mcp_servers={"x": {}}), "mcp_isolation_unverified"),
    (lambda f: f.responses["thread/start"].update(model="other"), "thread_isolation_unverified"),
    (lambda f: f.responses["thread/start"].update(instructionSources=["/secret/AGENTS.md"]), "thread_isolation_unverified"),
    (lambda f: f.responses["thread/start"]["thread"].update(path="/saved/thread"), "thread_isolation_unverified"),
])
def test_preflight_fails_closed(change, code):
    fake = FakeSession()
    change(fake)
    with pytest.raises(SubscriptionTransportError, match=code):
        inspect_preflight(fake)
    assert not any(m == "turn/start" for m, _ in fake.calls)


@pytest.mark.parametrize("value", [None, True, float("nan"), -1, 101, "20"])
def test_bad_quota_fails_closed(value):
    with pytest.raises(SubscriptionTransportError):
        quota_metadata({"rateLimits": {"primary": {"usedPercent": value,
            "windowDurationMins": 300, "resetsAt": 1000}}})


def test_all_account_windows_enforce_floor():
    with pytest.raises(SubscriptionTransportError, match="quota_floor"):
        quota_metadata({"rateLimitsByLimitId": {"one": {"primary": {
            "usedPercent": 20, "windowDurationMins": 300, "resetsAt": 1000}},
            "two": {"secondary": {"usedPercent": 76, "windowDurationMins": 10080, "resetsAt": 1000}}}})


def test_no_inference_before_runtime_acceptance_and_no_fake_http_counts():
    client = CodexSubscriptionClient("unused")
    with pytest.raises(SubscriptionTransportError, match="runtime_isolation_not_accepted"):
        with measure_provider_attempts(client) as measurement:
            client.complete(LLMRequest("system", "user"))
    assert client.stopped is True
    assert client.observed_turns == 0
    assert measurement.exact is False
    assert client.internal_http_attempts is None


def test_chat_rejects_history():
    with pytest.raises(SubscriptionTransportError, match="unsupported_chat_history"):
        CodexSubscriptionClient("unused").chat([{"role": "user", "content": "old"}])


def test_chat_rejects_model_override_or_unknown_controls():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    client = CodexSubscriptionClient("unused")
    with pytest.raises(SubscriptionTransportError, match="unsupported_chat_control"):
        client.chat(messages, model="gpt-5.6-luna")
    with pytest.raises(SubscriptionTransportError, match="unsupported_chat_control"):
        client.chat(messages, provider="api_key")


def test_usage_based_plan_is_rejected_before_thread_creation():
    fake = FakeSession()
    fake.responses["account/read"]["account"]["planType"] = "self_serve_business_usage_based"
    with pytest.raises(SubscriptionTransportError, match="subscription_plan_unverified"):
        inspect_preflight(fake)
    assert not any(m == "thread/start" for m, _ in fake.calls)


def test_individual_quota_floor_and_nullable_window_metadata():
    response = {"rateLimits": {"primary": {"usedPercent": 0}, "individualLimit": {
        "remainingPercent": 24, "resetsAt": 1000, "used": "private", "limit": "private"}}}
    with pytest.raises(SubscriptionTransportError, match="quota_floor"):
        quota_metadata(response)
    response["rateLimits"]["individualLimit"]["remainingPercent"] = 75
    windows = quota_metadata(response)
    assert windows[0]["window_minutes"] is None
    assert "private" not in str(windows)


def test_preflight_owns_cleanup_on_error():
    class OwnedSession(FakeSession):
        def __init__(self, binary, cwd):
            super().__init__()
            self.closed = False
            self.responses["account/read"] = {"account": {"type": "apiKey"}}
            created.append(self)

        def close(self):
            self.closed = True

    created = []
    client = CodexSubscriptionClient("unused", session_factory=OwnedSession)
    with pytest.raises(SubscriptionTransportError):
        client.preflight()
    assert created[0].closed is True


@pytest.mark.parametrize("value", ["contains private text", "warning: secret", "x" * 97, None, {"secret": "value"}])
def test_diagnostic_method_never_echoes_untrusted_strings(value):
    assert _safe_method(value) == "invalid_method"


def test_diagnostic_method_accepts_only_protocol_identifier():
    assert _safe_method("codex/event/mcp_startup_update") == "codex/event/mcp_startup_update"


@pytest.mark.parametrize("status", ["disabled", "connecting", "connected", "errored", None])
def test_remote_control_startup_notification_requires_disabled(status):
    session = object.__new__(StdioSession)
    session.send = lambda *args, **kwargs: 1
    events = iter([{"method": "remoteControl/status/changed", "params": {"status": status,
        "identity": "private"}}, {"id": 1, "result": {"ok": True}}])
    session.receive = lambda: next(events)
    if status == "disabled":
        assert session.rpc("initialize", {}) == {"ok": True}
    else:
        with pytest.raises(SubscriptionTransportError, match="remote_control_enabled"):
            session.rpc("initialize", {})


def test_previous_luna_catalog_never_substitutes_for_exact_requested_model():
    fake = FakeSession()
    fake.responses["model/list"]["data"][0]["model"] = "gpt-5.6-luna"
    with pytest.raises(SubscriptionTransportError, match="exact_model_unavailable"):
        inspect_preflight(fake)
    assert not any(m == "thread/start" for m, _ in fake.calls)


def test_installed_version_pin_rejects_old_binary_before_app_server(monkeypatch, tmp_path):
    class VersionResult:
        stdout = "codex-cli 0.152.1\n"

    called = []
    monkeypatch.setattr("benchmarks.codex_subscription.subprocess.run", lambda *args, **kwargs: VersionResult())
    monkeypatch.setattr("benchmarks.codex_subscription.subprocess.Popen",
                        lambda *args, **kwargs: called.append(True))
    assert VERSION == "0.158.0"
    with pytest.raises(SubscriptionTransportError, match="binary_version_mismatch"):
        StdioSession("/official/codex", str(tmp_path))
    assert called == []


@pytest.mark.parametrize("endpoint", ["https://chatgpt.com/backend-api", "https://chatgpt.com/backend-api/"])
def test_official_chatgpt_backend_default_is_accepted(endpoint):
    fake = FakeSession()
    fake.responses["config/read"]["config"]["chatgpt_base_url"] = endpoint
    result = inspect_preflight(fake)
    assert result["auth"] == "chatgpt"
    assert "chatgpt_base_url" not in result


@pytest.mark.parametrize("endpoint", [
    "http://chatgpt.com/backend-api", "https://chatgpt.com.evil.test/backend-api",
    "https://user@chatgpt.com/backend-api", "https://chatgpt.com/backend-api?x=1",
    "https://chatgpt.com/backend-api/extra", "https://CHATGPT.COM/backend-api",
])
def test_chatgpt_endpoint_lookalikes_rejected(endpoint):
    fake = FakeSession()
    fake.responses["config/read"]["config"]["chatgpt_base_url"] = endpoint
    with pytest.raises(SubscriptionTransportError, match="provider_endpoint_override"):
        inspect_preflight(fake)
    assert not any(m == "thread/start" for m, _ in fake.calls)


def _turn_events(*extra):
    base = [
        {"method": "turn/started", "params": {"threadId": "thr", "turn": {"id": "turn"}}},
        {"method": "item/started", "params": {"threadId": "thr", "turnId": "turn",
            "item": {"type": "agentMessage", "id": "msg", "phase": "final_answer", "text": ""}}},
        {"method": "item/completed", "params": {"threadId": "thr", "turnId": "turn",
            "item": {"type": "agentMessage", "id": "msg", "phase": "final_answer", "text": "answer"}}},
        {"method": "thread/tokenUsage/updated", "params": {"threadId": "thr", "turnId": "turn",
            "tokenUsage": {"total": {"totalTokens": 42}}}},
        {"method": "turn/completed", "params": {"threadId": "thr",
            "turn": {"id": "turn", "status": "completed"}}},
    ]
    return base[:2] + list(extra) + base[2:]


class FakeTurnSession:
    def __init__(self, events):
        self.events = iter(events)
        self.params = None

    def rpc(self, method, params, **kwargs):
        assert method == "turn/start"
        assert kwargs["preserve_notifications"] is True
        self.params = params
        return {"turn": {"id": "turn", "status": "inProgress"}}

    def next_event(self):
        return next(self.events)


def test_turn_success_uses_exact_ids_usage_and_readonly_environment():
    session = FakeTurnSession(_turn_events())
    assert _run_turn(session, "thr", "question") == ("answer", 42)
    assert session.params["input"] == [{"type": "text", "text": "question"}]
    assert session.params["environments"] == []
    assert session.params["sandboxPolicy"] == {"type": "readOnly", "networkAccess": False}


@pytest.mark.parametrize("event,code", [
    ({"method": "item/started", "params": {"threadId": "thr", "turnId": "turn",
      "item": {"type": "commandExecution", "id": "cmd"}}}, "tool_or_extra_item"),
    ({"method": "item/completed", "params": {"threadId": "other", "turnId": "turn",
      "item": {"type": "agentMessage", "phase": "final_answer", "text": "leak"}}}, "item_identity_mismatch"),
    ({"method": "model/rerouted", "params": {"threadId": "thr", "turnId": "turn"}}, "unexpected_notification:model/rerouted"),
    ({"method": "item/completed", "params": {"threadId": "thr", "turnId": "turn",
      "item": {"type": "agentMessage", "id": "another", "phase": "final_answer", "text": "second"}}}, "item_lifecycle_invalid"),
    ({"method": "hook/started", "params": {"threadId": "thr", "turnId": "turn"}}, "unexpected_notification:hook/started"),
])
def test_turn_rejects_unsafe_or_ambiguous_events(event, code):
    with pytest.raises(SubscriptionTransportError, match=code):
        _run_turn(FakeTurnSession(_turn_events(event)), "thr", "question")


def test_turn_missing_usage_fails_closed():
    events = [e for e in _turn_events() if e["method"] != "thread/tokenUsage/updated"]
    with pytest.raises(SubscriptionTransportError, match="incomplete_turn_or_usage"):
        _run_turn(FakeTurnSession(events), "thr", "question")


def test_turn_rejects_usage_from_other_turn():
    events = _turn_events()
    events[3]["params"]["turnId"] = "other"
    with pytest.raises(SubscriptionTransportError, match="usage_identity_mismatch"):
        _run_turn(FakeTurnSession(events), "thr", "question")


def test_turn_rejects_regressed_cumulative_usage():
    update = {"method": "thread/tokenUsage/updated", "params": {"threadId": "thr", "turnId": "turn",
              "tokenUsage": {"total": {"totalTokens": 50}}}}
    with pytest.raises(SubscriptionTransportError, match="usage_regressed"):
        _run_turn(FakeTurnSession(_turn_events(update)), "thr", "question")


def test_thread_start_requires_confirmed_readonly_sandbox():
    fake = FakeSession()
    fake.responses["thread/start"].pop("sandbox")
    with pytest.raises(SubscriptionTransportError, match="thread_isolation_unverified"):
        inspect_preflight(fake)


def test_streamed_message_delta_is_ignored_in_favor_of_final_item():
    delta = {"method": "item/agentMessage/delta", "params": {"threadId": "thr", "turnId": "turn",
        "itemId": "msg", "delta": "part"}}
    assert _run_turn(FakeTurnSession(_turn_events(delta)), "thr", "question") == ("answer", 42)


def test_complete_fresh_thread_counts_observed_usage_and_closes_session():
    created = []

    class CompleteSession(FakeSession):
        def __init__(self, binary, cwd, timeout):
            super().__init__()
            self.timeout = timeout
            self.closed = False
            self.events = iter(_turn_events())
            self.responses["thread/start"]["thread"]["id"] = "thr"
            created.append(self)

        def rpc(self, method, params, **kwargs):
            if method == "turn/start":
                self.calls.append((method, params))
                return {"turn": {"id": "turn", "status": "inProgress"}}
            return super().rpc(method, params)

        def next_event(self):
            return next(self.events)

        def close(self):
            self.closed = True

    client = CodexSubscriptionClient("unused", session_factory=CompleteSession, inference_accepted=True)
    assert client.complete(LLMRequest("SYSTEM", "USER", max_tokens=17, temperature=0.3)) == "answer"
    assert client.observed_turns == 1
    assert client.observed_tokens == 42
    assert client.internal_http_attempts is None
    assert client.requested_controls == [{"temperature_requested": 0.3,
        "max_tokens_requested": 17, "temperature_effective": None, "max_tokens_effective": None,
        "response_format_requested": "json", "response_format_effective": None}]
    assert created[0].timeout <= 120
    assert created[0].closed
    assert dict(created[0].calls)["thread/start"]["baseInstructions"] == "SYSTEM"
    assert dict(created[0].calls)["turn/start"]["input"] == [{"type": "text", "text": "USER"}]


def test_failed_completion_stops_further_calls_and_marks_usage_unknown():
    class FailedSession(FakeSession):
        def __init__(self, binary, cwd, timeout):
            super().__init__()
            self.responses["thread/start"]["thread"]["id"] = "thr"

        def rpc(self, method, params, **kwargs):
            if method == "turn/start":
                raise SubscriptionTransportError("rpc_failure:turn/start")
            return super().rpc(method, params)

        def close(self):
            pass

    client = CodexSubscriptionClient("unused", session_factory=FailedSession, inference_accepted=True)
    with pytest.raises(SubscriptionTransportError, match="rpc_failure:turn/start"):
        client.complete(LLMRequest("S", "U"))
    assert client.stopped
    assert client.observed_turns == 1
    assert client.observed_tokens is None
    assert client.usage_complete is False
    with pytest.raises(SubscriptionTransportError, match="pilot_stopped"):
        client.complete(LLMRequest("S", "U"))


def test_account_and_quota_notifications_fail_closed():
    with pytest.raises(SubscriptionTransportError, match="account_changed"):
        _validate_account_notification({"method": "account/updated",
            "params": {"authMode": "apikey", "planType": "pro"}})
    with pytest.raises(SubscriptionTransportError, match="quota_floor"):
        _validate_account_notification({"method": "account/rateLimits/updated",
            "params": {"rateLimits": {"primary": {"usedPercent": 76}}}})


def test_effective_instruction_and_skill_controls_are_required():
    fake = FakeSession()
    fake.responses["config/read"]["config"]["memories"]["use_memories"] = True
    with pytest.raises(SubscriptionTransportError, match="instruction_isolation_unverified"):
        inspect_preflight(fake)
    fake = FakeSession()
    fake.responses["config/read"]["config"]["features"]["skip_host_skill_discovery"] = False
    with pytest.raises(SubscriptionTransportError, match="capability_isolation_unverified"):
        inspect_preflight(fake)


def test_single_flight_rejects_concurrent_completion():
    client = CodexSubscriptionClient("unused", inference_accepted=True)
    assert client._flight.acquire(blocking=False)
    try:
        with pytest.raises(SubscriptionTransportError, match="concurrent_completion_rejected"):
            client.complete(LLMRequest("S", "U"))
    finally:
        client._flight.release()
    assert client.observed_turns == 0


def test_stdio_reader_bounds_line_before_json_parse():
    class OversizedStream:
        def __init__(self):
            self.calls = []

        def readline(self, limit):
            self.calls.append(limit)
            return "x" * limit if len(self.calls) == 1 else ""

    session = object.__new__(StdioSession)
    stream = OversizedStream()
    session.process = type("Process", (), {"stdout": stream})()
    session.events = queue.Queue()
    session._read()
    assert stream.calls == [8_000_001]
    assert session.events.get_nowait() is None


def test_failed_later_preflight_preserves_prior_complete_usage():
    class RejectedSession(FakeSession):
        def __init__(self, binary, cwd, timeout):
            super().__init__()
            self.responses["account/read"] = {"account": {"type": "apikey"}}

        def close(self):
            pass

    client = CodexSubscriptionClient("unused", session_factory=RejectedSession, inference_accepted=True)
    client.observed_turns = 1
    client.observed_tokens = 42
    with pytest.raises(SubscriptionTransportError, match="subscription_auth_required"):
        client.complete(LLMRequest("S", "U"))
    assert client.observed_turns == 1
    assert client.observed_tokens == 42
    assert client.usage_complete is True
    assert client.stopped is True


@pytest.mark.parametrize("field,value", [
    ("approvalPolicy", "on-request"), ("reasoningEffort", "high"),
    ("serviceTier", "fast"), ("runtimeWorkspaceRoots", ["/tmp/other"]),
])
def test_root_review_rejects_mismatched_thread_controls(field, value):
    fake = FakeSession()
    fake.responses["thread/start"][field] = value
    with pytest.raises(SubscriptionTransportError, match="thread_isolation_unverified"):
        inspect_preflight(fake)


@pytest.mark.parametrize("environments", [None, [{"id": "host"}]])
def test_root_review_requires_explicit_empty_environment_echo(environments):
    fake = FakeSession()
    fake.responses["thread/start"]["thread"]["environments"] = environments
    with pytest.raises(SubscriptionTransportError, match="thread_isolation_unverified"):
        inspect_preflight(fake)


@pytest.mark.parametrize("state", [
    {"observed_turns": 1200}, {"observed_tokens": 4_000_000},
    {"usage_complete": False}, {"started_at": -1e12},
    {"observed_turns": 1, "observed_tokens": None},
])
def test_root_review_budgets_stop_before_starting_process(state):
    calls = []
    client = CodexSubscriptionClient("unused", inference_accepted=True,
        session_factory=lambda *a, **k: calls.append(True))
    for field, value in state.items():
        setattr(client, field, value)
    with pytest.raises(SubscriptionTransportError, match="pilot_budget_exhausted"):
        client.complete(LLMRequest("S", "U"))
    assert not calls
    assert client.stopped


def test_root_review_quota_notification_halts_active_turn():
    event = {"method": "account/rateLimits/updated", "params": {
        "rateLimits": {"primary": {"usedPercent": 76}}}}
    with pytest.raises(SubscriptionTransportError, match="quota_floor"):
        _run_turn(FakeTurnSession(_turn_events(event)), "thr", "question")


def test_root_review_zero_placeholder_is_not_final_usage():
    events = _turn_events()
    for event in events:
        if event["method"] == "thread/tokenUsage/updated":
            event["params"]["tokenUsage"]["total"]["totalTokens"] = 0
    with pytest.raises(SubscriptionTransportError, match="incomplete_turn_or_usage"):
        _run_turn(FakeTurnSession(events), "thr", "question")


def _unstable_notice(path):
    return ("Under-development features enabled: skip_host_skill_discovery. "
            "Under-development features are incomplete and may behave unpredictably. "
            "To suppress this warning, set `suppress_unstable_features_warning = true` "
            f"in {path}.")


def test_only_exact_pinned_startup_notice_is_accepted(monkeypatch):
    import hashlib

    monkeypatch.setenv("HOME", "/home/atta")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    message = _unstable_notice("/home/atta/.codex/config.toml")
    assert len(message.encode()) == 242
    assert hashlib.sha256(message.encode()).hexdigest() == "5e1ab79fba80daaf37190e68176dca33a52dc3141d1ccdd756002009aa128614"
    event = {"method": "warning", "params": {"message": message, "threadId": "thr"}}
    assert _validate_warning(event, "thr") == "thr"
    wrong = [
        message.replace("skip_host_skill_discovery", "plugins"),
        message.replace("/home/atta/.codex/config.toml", "/other/config.toml"),
        message + " extra", message.replace("in /home", "in\n/home"),
        "Configured service tier `default` is not advertised as supported for model `gpt-6-luna` and will be omitted from requests.",
    ]
    for text in wrong:
        with pytest.raises(SubscriptionTransportError, match="warning_unapproved"):
            _validate_warning({"method": "warning", "params": {"message": text, "threadId": "thr"}}, "thr")
    with pytest.raises(SubscriptionTransportError, match="warning_thread_unverified"):
        _validate_warning({"method": "warning", "params": {"message": message, "threadId": "other"}}, "thr")
    with pytest.raises(SubscriptionTransportError, match="warning_thread_unverified"):
        _validate_warning({"method": "warning", "params": {"message": message, "threadId": None}}, "thr")


def test_only_exact_code_mode_fail_closed_notice_is_accepted():
    import hashlib

    message = ("Code Mode is unavailable because code-mode host is disabled. "
        "Code mode will fail closed; enable `features.code_mode_host` and install "
        "`codex-code-mode-host`.")
    assert len(message.encode()) == 157
    assert hashlib.sha256(message.encode()).hexdigest() == "098e801ebc95c9c7312a945849442846324dcf639365a297313248993822711b"
    event = {"method": "warning", "params": {"message": message, "threadId": "thr"}}
    assert _validate_warning(event, "thr") == "thr"
    for wrong in (
        message.replace("will fail closed", "will fall back to direct tools"),
        message.replace("host is disabled", "host is missing"),
        message.replace("host is disabled", "host crashed"),
        message.replace("features.code_mode_host", "features.code_mode"),
        message.replace("codex-code-mode-host", "codex-code-mode-fallback"),
        message + " Extra text.",
    ):
        with pytest.raises(SubscriptionTransportError, match="warning_unapproved"):
            _validate_warning({"method": "warning", "params": {"message": wrong, "threadId": "thr"}}, "thr")
    with pytest.raises(SubscriptionTransportError, match="warning_thread_unverified"):
        _validate_warning({"method": "warning", "params": {"message": message, "threadId": "other"}}, "thr")


def test_exact_warnings_before_thread_binding_and_repeated_bound_notices(monkeypatch):
    monkeypatch.setenv("HOME", "/home/atta")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    event = {"method": "warning", "params": {"message": _unstable_notice("/home/atta/.codex/config.toml"),
                                           "threadId": "thr"}}
    session = object.__new__(StdioSession)
    session.warning_targets = []
    session.bound_thread_id = None
    session.accept_warning(event, None)
    session.bind_thread_id("thr")
    session.accept_warning(event, "thr")
    assert len(session.warning_targets) == 2
    other = object.__new__(StdioSession)
    other.warning_targets = ["wrong"]
    other.bound_thread_id = None
    with pytest.raises(SubscriptionTransportError, match="warning_thread_unverified"):
        other.bind_thread_id("thr")


def test_queued_exact_warning_before_and_after_turn_started(monkeypatch):
    monkeypatch.setenv("HOME", "/home/atta")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    event = {"method": "warning", "params": {"message": _unstable_notice("/home/atta/.codex/config.toml"),
                                           "threadId": "thr"}}
    session = FakeTurnSession([event] + _turn_events())
    assert _run_turn(session, "thr", "question") == ("answer", 42)
    assert _run_turn(FakeTurnSession(_turn_events(event)), "thr", "question") == ("answer", 42)
    assert _run_turn(FakeTurnSession(_turn_events(*([event] * 16))), "thr", "question") == ("answer", 42)
    with pytest.raises(SubscriptionTransportError, match="warning_limit"):
        _run_turn(FakeTurnSession(_turn_events(*([event] * 17))), "thr", "question")


def test_rpc_warning_before_thread_reply_or_before_turn_started(monkeypatch):
    from collections import deque

    monkeypatch.setenv("HOME", "/home/atta")
    monkeypatch.delenv("CODEX_HOME", raising=False)
    warning = {"method": "warning", "params": {"message": _unstable_notice("/home/atta/.codex/config.toml"),
                                               "threadId": "thr"}}

    def session_with(events):
        session = object.__new__(StdioSession)
        session.next_id = 0
        session.stage = "startup"
        session.warning_targets = []
        session.bound_thread_id = None
        session.turn_started_seen = False
        session.pending = deque()
        session.send = lambda *args, **kwargs: 1
        iterator = iter(events)
        session.receive = lambda: next(iterator)
        return session

    early = session_with([warning, {"id": 1, "result": {"thread": {"id": "thr"}}}])
    assert early.rpc("thread/start", {})["thread"]["id"] == "thr"
    early.bind_thread_id("thr")
    late = session_with([warning, {"id": 1, "result": {"turn": {"id": "turn"}}}])
    late.bind_thread_id("thr")
    assert late.rpc("turn/start", {}, preserve_notifications=True)["turn"]["id"] == "turn"
    after_started = session_with([{"method": "turn/started", "params": {"turn": {"id": "turn"}}},
                                  warning, {"id": 1, "result": {}}])
    after_started.bind_thread_id("thr")
    assert after_started.rpc("turn/start", {}, preserve_notifications=True) == {}
    assert len(after_started.warning_targets) == 1
