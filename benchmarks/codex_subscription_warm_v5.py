"""Bounded same-turn retry-progress support for the source-pinned warm transport.

Only a validated server notification can keep the existing turn's event loop
open. The original deadline, event ceiling, final-output and usage gates remain
the authority for completion.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import sys
import types
from typing import Any


_v4_path = Path(__file__).resolve().with_name("codex_subscription_warm_v4.py")
_v4_source = _v4_path.read_bytes()
if hashlib.sha256(_v4_source).hexdigest() != "43611c5b7c9b2242f8216daf1cb274c7b1f82c7c633abd11830bf5759fdb4138":
    raise RuntimeError("pinned_warm_v4_source_mismatch")
v4 = types.ModuleType("pinned_codex_subscription_warm_v5_base")
v4.__file__ = str(_v4_path)
sys.modules[v4.__name__] = v4
exec(compile(_v4_source, str(_v4_path), "exec"), v4.__dict__)

base = v4.base
concurrent = v4.concurrent
BudgetLimits = v4.BudgetLimits
ConcurrentStop = v4.ConcurrentStop
SharedBudget = v4.SharedBudget
WarmSession = v4.WarmSession
WarmSubscriptionClient = v4.WarmSubscriptionClient
serialize_failure = v4.serialize_failure

MAX_RETRY_PROGRESS = 8
_RETRY_CLASSES = frozenset({
    "httpConnectionFailed", "responseStreamConnectionFailed",
    "responseStreamDisconnected",
})


def _validated_retry_progress(event: Any, thread_id: str, turn_id: str) -> int | None:
    """Return the finite HTTP status, or -1 for absent/null, only for progress."""
    if type(event) is not dict or set(event) != {"method", "params"} or event.get("method") != "error":
        return None
    params = event.get("params")
    if (type(params) is not dict
            or set(params) != {"threadId", "turnId", "willRetry", "error"}
            or type(params.get("threadId")) is not str
            or type(params.get("turnId")) is not str
            or params["threadId"] != thread_id or params["turnId"] != turn_id
            or params.get("willRetry") is not True):
        return None
    error = params.get("error")
    if (type(error) is not dict or not {"message", "codexErrorInfo"} <= set(error)
            or set(error) - {"message", "codexErrorInfo", "additionalDetails", "misalignment"}
            or type(error.get("message")) is not str
            or ("additionalDetails" in error and error["additionalDetails"] is not None
                and type(error["additionalDetails"]) is not str)
            or error.get("misalignment") is not None):
        return None
    info = error.get("codexErrorInfo")
    if type(info) is not dict or len(info) != 1:
        return None
    kind, detail = next(iter(info.items()))
    if kind not in _RETRY_CLASSES or type(detail) is not dict or set(detail) - {"httpStatusCode"}:
        return None
    status = detail.get("httpStatusCode")
    if status is not None and (type(status) is not int or not 0 <= status <= 65535):
        return None
    return -1 if status is None else status


def _patch_pinned_parser() -> None:
    # Compile only the pinned function and insert one guarded branch in its
    # MAX_EVENTS loop. No source file or other base-module parser is edited.
    source = Path(base.__file__).read_bytes()
    if hashlib.sha256(source).hexdigest() != "387cf65be4cd7e6dabd6ba8a9ee90d9f23d0c7669702979b87926d5b71e81491":
        raise RuntimeError("pinned_warm_base_source_mismatch")
    tree = ast.parse(source, filename=str(base.__file__))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_turn"]
    if len(functions) != 1:
        raise RuntimeError("pinned_run_turn_shape_mismatch")
    function = functions[0]
    loops = [node for node in function.body if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name)
             and node.iter.func.id == "range" and len(node.iter.args) == 1
             and isinstance(node.iter.args[0], ast.Name) and node.iter.args[0].id == "MAX_EVENTS"]
    if len(loops) != 1:
        raise RuntimeError("pinned_run_turn_loop_mismatch")
    loop = loops[0]
    fallback = loop.body[-1]
    if (not isinstance(fallback, ast.Expr) or not isinstance(fallback.value, ast.Call)
            or not isinstance(fallback.value.func, ast.Name) or fallback.value.func.id != "_fail"):
        raise RuntimeError("pinned_run_turn_fallback_mismatch")
    initialization = ast.parse("retry_progress_count = 0\nsession.retry_progress_count = 0\nsession.retry_progress_http_status = None").body
    branch = ast.parse("""
if method == "error":
    retry_status = _validated_retry_progress_v5(event, thread_id, turn_id)
    if retry_status is not None and retry_progress_count < MAX_RETRY_PROGRESS_V5:
        retry_progress_count += 1
        session.retry_progress_count = retry_progress_count
        session.retry_progress_http_status = None if retry_status == -1 else retry_status
        continue
""").body[0]
    index = function.body.index(loop)
    function.body[index:index] = initialization
    loop.body.insert(len(loop.body) - 1, branch)
    global _PATCHED_PARSER_AST
    _PATCHED_PARSER_AST = function
    base.__dict__["_validated_retry_progress_v5"] = _validated_retry_progress
    base.__dict__["MAX_RETRY_PROGRESS_V5"] = MAX_RETRY_PROGRESS
    ast.fix_missing_locations(function)
    code = compile(ast.Module(body=[function], type_ignores=[]), str(base.__file__), "exec")
    exec(code, base.__dict__)


_patch_pinned_parser()
