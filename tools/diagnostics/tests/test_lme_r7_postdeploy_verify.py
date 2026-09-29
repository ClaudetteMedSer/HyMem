"""Offline controls for the post-deployment process/config verifier."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest


SCRIPT = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "lme_r7_postdeploy_verify.py")
)["CONTAINER_SCRIPT"]
TREE = ast.parse(SCRIPT)
PROCESS_ENV = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "process_env"
)


def _probe(honcho: dict[str, str], mcp: dict[str, str]):
    def packed(values):
        return b"\0".join(
            key.encode() + b"=" + value.encode() for key, value in values.items()
        ) + b"\0"

    data = {
        "/proc/999/cmdline": (
            b"python\0-c\0verifier source mentions b'hymem.honcho' "
            b"and b'hymem.server'\0"
        ),
        "/proc/101/cmdline": b"python\0-m\0hymem.honcho\0",
        "/proc/102/cmdline": b"python\0-m\0hymem.server\0",
        "/proc/101/environ": packed(honcho),
        "/proc/102/environ": packed(mcp),
    }

    class FakePath:
        def __init__(self, value):
            self.value = str(value)
            self.name = self.value.rsplit("/", 1)[-1]

        def iterdir(self):
            return [FakePath("/proc/999"), FakePath("/proc/101"), FakePath("/proc/102")]

        def __truediv__(self, other):
            return FakePath(self.value + "/" + str(other))

        def read_bytes(self):
            return data[self.value]

    report = {}

    def need(ok, code):
        if not ok:
            raise RuntimeError(code)

    scope = {"pathlib": SimpleNamespace(Path=FakePath), "os": os,
             "report": report, "need": need}
    exec(compile(ast.Module(body=[PROCESS_ENV], type_ignores=[]),
                 "<process_env_control>", "exec"), scope)
    return scope["process_env"], report


def _base():
    return {"HYMEM_LLM_API_KEY": "active-explicit-key",
            "HYMEM_LLM_MODEL": "deepseek-flash",
            "HYMEM_LLM_EXTRA_BODY": '{"reasoning":false}',
            "HYMEM_LLM_THINKING": "off",
            "HYMEM_EMBEDDING_ALLOW_INSECURE_INTERNAL_HTTP": "true",
            "DEEPSEEK_API_KEY": "unused-alias-a"}


def test_unused_fallback_alias_difference_is_accepted():
    honcho = _base()
    mcp = {**honcho, "DEEPSEEK_API_KEY": "unused-alias-b"}
    probe, report = _probe(honcho, mcp)
    assert probe()["HYMEM_LLM_API_KEY"] == "active-explicit-key"
    assert report["processes"]["config_fields_compared"] == 14


def test_active_explicit_key_difference_is_rejected():
    honcho = _base()
    mcp = {**honcho, "HYMEM_LLM_API_KEY": "different-active-key"}
    probe, _ = _probe(honcho, mcp)
    with pytest.raises(RuntimeError, match="^mcp_honcho_config_mismatch$"):
        probe()


def test_verifier_source_does_not_match_as_honcho_or_mcp_process():
    env = _base()
    probe, report = _probe(env, env)
    probe()
    assert report["processes"]["honcho_pid"] == 101
    assert report["processes"]["mcp_pids"] == [102]
