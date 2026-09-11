"""Run inside the patched Hermes checkout, not against HyMem's Python package.

All sessions and peers are local fakes; tests never connect to Honcho or use an
operator's configuration. See ../hermes-session-key-retirement.md.
"""

import json
import socket
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Network forbidden in session retirement tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    for name in ("HONCHO_API_KEY", "HONCHO_BASE_URL", "HONCHO_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)
    yield
    assert not attempts


def load(tmp_path, mapping=None, *, host="hermes", root=None):
    raw = {"hosts": {"hermes": {"writeFrequency": "turn"}}}
    if mapping is not None:
        raw["hosts"]["hermes"]["sessionKeyReplacements"] = mapping
    if root:
        raw.update(root)
    path = tmp_path / "honcho.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return HonchoClientConfig.from_global_config(host=host, config_path=path), path


def make_manager(config, **kwargs):
    manager = HonchoSessionManager(config=config, **kwargs)
    manager._get_or_create_peer = Mock(side_effect=lambda peer_id: SimpleNamespace(
        id=peer_id, message=lambda content: (peer_id, content)
    ))
    physical = Mock()

    def open_session(session_id, user, assistant):
        manager._sessions_cache[session_id] = physical
        return physical, []

    manager._get_or_create_honcho_session = Mock(side_effect=open_session)
    return manager, physical


def test_retirement_survives_config_reload_and_new_managers(tmp_path):
    config, path = load(tmp_path, {"telegram-123": "telegram-123-retired-v2"})
    config_bytes = path.read_bytes()
    for _ in range(3):
        config = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
        manager, physical = make_manager(config)
        session = manager.get_or_create("telegram:123")
        assert session.key == "telegram:123"
        assert session.honcho_session_id == "telegram-123-retired-v2"
        assert session.user_peer_id == "user-telegram-123"
        assert session.assistant_peer_id == "hermes"
        assert manager.get_or_create("telegram:123") is session
        assert manager._get_or_create_honcho_session.call_count == 1
        assert manager._get_or_create_honcho_session.call_args.args[0] == session.honcho_session_id
        assert "telegram-123" not in manager._sessions_cache
        session.add_message("user", "Synthetic test message")
        assert manager._flush_session(session)
        physical.add_messages.assert_called_once_with([
            ("user-telegram-123", "Synthetic test message")
        ])
        assert path.read_bytes() == config_bytes


def test_unmapped_keys_and_other_hosts_unchanged(tmp_path):
    config, path = load(tmp_path, {"telegram-123": "fresh-123"})
    manager, _ = make_manager(config)
    assert manager.get_or_create("telegram:456").honcho_session_id == "telegram-456"
    other = HonchoClientConfig.from_global_config(host="other-host", config_path=path)
    manager, _ = make_manager(other)
    assert manager.get_or_create("telegram:123").honcho_session_id == "telegram-123"


def test_physical_ids_are_exact_not_prefix_or_resolver_aliases(tmp_path):
    config, _ = load(tmp_path, {"telegram--123": "fresh_123"})
    manager, _ = make_manager(config)
    assert manager.get_or_create("telegram::123").honcho_session_id == "fresh_123"
    assert manager.get_or_create("telegram:123").honcho_session_id == "telegram-123"
    assert manager.get_or_create("telegram::1234").honcho_session_id == "telegram--1234"


@pytest.mark.parametrize("identity", [
    {},
    {"peer_name": "stable-user", "pin_peer_name": True},
    {"user_peer_aliases": {"runtime": "stable-user"}},
    {"runtime_peer_prefix": "gateway_"},
])
def test_peer_resolution_is_identical_with_and_without_retirement(identity):
    common = dict(write_frequency="turn", ai_peer="assistant-id", **identity)
    baseline, _ = make_manager(HonchoClientConfig(**common), runtime_user_peer_name="runtime")
    mapped, _ = make_manager(HonchoClientConfig(
        **common, session_key_replacements={"logical-key": "fresh-key"}
    ), runtime_user_peer_name="runtime")
    old = baseline.get_or_create("logical:key")
    new = mapped.get_or_create("logical:key")
    assert new.key == old.key == "logical:key"
    assert (new.user_peer_id, new.assistant_peer_id) == (old.user_peer_id, old.assistant_peer_id)
    assert new.honcho_session_id == "fresh-key"


@pytest.mark.parametrize("host_value, expected", [
    (None, {"root-old": "root-new"}),
    ({}, {}),
    ({"host-old": "host-new"}, {"host-old": "host-new"}),
])
def test_host_override_is_none_aware_whole_map(tmp_path, host_value, expected):
    config, _ = load(tmp_path, root={
        "sessionKeyReplacements": {"root-old": "root-new"},
        "hosts": {"hermes": {"sessionKeyReplacements": host_value}},
    })
    assert config.session_key_replacements == expected


@pytest.mark.parametrize("value", [
    [], False, 0, "", "{\"old\":\"new\"}",
    {"": "new"}, {"old": ""}, {"old": None}, {"old": 5},
    {"old": True}, {"old": []}, {" old": "new"}, {"old ": "new"},
    {"old:raw": "new"}, {"old": "bad/slash"}, {"old": "bad space"},
    {"old": "new\n"}, {"old": "\u00e9"}, {"old": "x" * 101},
    {"old": "old"}, {"a": "same", "b": "same"},
    {"a": "b", "b": "c"}, {"a": "b", "b": "a"},
])
def test_invalid_effective_maps_fail_closed(tmp_path, value):
    with pytest.raises(ValueError, match="sessionKeyReplacements"):
        load(tmp_path, root={"hosts": {"hermes": {"sessionKeyReplacements": value}}})


def test_bad_host_override_does_not_fall_back_to_valid_root(tmp_path):
    with pytest.raises(ValueError, match="sessionKeyReplacements"):
        load(tmp_path, root={
            "sessionKeyReplacements": {"old": "new"},
            "hosts": {"hermes": {"sessionKeyReplacements": False}},
        })


def test_direct_constructor_rejects_non_string_keys_and_null():
    for value in ({5: "new"}, {True: "new"}, None):
        with pytest.raises(ValueError, match="sessionKeyReplacements"):
            HonchoClientConfig(session_key_replacements=value)


def test_max_target_length_and_detached_manager_snapshot():
    mapping = {"old": "x" * 100}
    config = HonchoClientConfig(session_key_replacements=mapping)
    mapping["old"] = "external-edit"
    manager, _ = make_manager(config)
    config.session_key_replacements["old"] = "later-config-edit"
    assert manager.get_or_create("old").honcho_session_id == "x" * 100


def test_manager_revalidates_mutated_config_before_any_io():
    config = HonchoClientConfig()
    config.session_key_replacements = {"old": "old"}
    with pytest.raises(ValueError, match="sessionKeyReplacements"):
        HonchoSessionManager(config=config)


@pytest.mark.parametrize("raw", [
    '{"apiKey": "secret-canary", "sessionKeyReplacements":',
    '{"hosts":{"hermes":{"sessionKeyReplacements":{"old":"one","old":"two"}}}}',
    '{"hosts":{"hermes":{},"hermes":{}}}',
    '[]', 'null', '{"hosts":[]}', '{"hosts":{"hermes":null}}',
])
def test_existing_bad_config_never_falls_back_or_exposes_values(tmp_path, monkeypatch, caplog, raw):
    path = tmp_path / "honcho.json"
    path.write_text(raw, encoding="utf-8")
    monkeypatch.setenv("HONCHO_API_KEY", "env-secret-canary")
    with pytest.raises(ValueError) as exc:
        HonchoClientConfig.from_global_config(host="hermes", config_path=path)
    assert "secret-canary" not in str(exc.value) + caplog.text


def test_existing_unreadable_config_fails_but_absent_config_keeps_env_fallback(tmp_path, monkeypatch):
    path = tmp_path / "honcho.json"
    monkeypatch.setenv("HONCHO_API_KEY", "env-canary")
    absent = HonchoClientConfig.from_global_config(host="hermes", config_path=path)
    assert absent.api_key == "env-canary"
    assert absent.session_key_replacements == {}
    path.write_text("{}", encoding="utf-8")

    def denied(*args, **kwargs):
        raise PermissionError("secret-canary")

    monkeypatch.setattr(type(path), "read_text", denied)
    with pytest.raises(ValueError, match="refusing environment fallback") as exc:
        HonchoClientConfig.from_global_config(host="hermes", config_path=path)
    assert "secret-canary" not in str(exc.value)


def test_schema_declares_host_scoped_operator_field():
    from plugins.memory.honcho.config_schema import CONFIG_SCHEMA

    fields = [field for field in CONFIG_SCHEMA.fields if field.key == "sessionKeyReplacements"]
    assert len(fields) == 1
    assert fields[0].scope != "root"
    assert "deleting" in fields[0].info
