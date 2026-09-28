"""Pinned warning diagnostic; inference requires explicit --capture-turn."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import queue
import re
import sys
import tempfile
import time

SERVICE = re.compile(r"Configured service tier `([^`]+)` is not advertised as supported "
                     r"for model `([^`]+)` and will be omitted from requests\.")
UNSTABLE = re.compile(r"Under-development features enabled: ([^.]+)\. Under-development "
                      r"features are incomplete and may behave unpredictably\. To suppress this "
                      r"warning, set `suppress_unstable_features_warning = true` in (.+)\.")
KEYWORDS = ("sandbox", "environment", "model", "reasoning", "service", "tier",
            "unsupported", "disabled", "ignored", "tools", "approval", "fallback",
            "permissions", "read", "write", "network", "profile", "requirements",
            "memory", "skill", "config", "temperature", "budget", "image", "codex",
            "websocket", "https", "transport", "stream", "request", "connection",
            "reconnect", "upgrade")


def classify(message: str, model: str) -> tuple[str, bool]:
    if not isinstance(message, str):
        return "invalid_warning", False
    match = SERVICE.fullmatch(message)
    if match:
        valid = match.groups() == ("default", model)
        return ("service_tier_omitted" if valid else "unknown_warning", valid)
    match = UNSTABLE.fullmatch(message)
    if match:
        names = [part.strip() for part in match.group(1).split(",")]
        path = match.group(2)
        valid = (names == ["skip_host_skill_discovery"] and path.startswith("/")
                 and path.endswith("/config.toml") and len(path) <= 512
                 and all(32 <= ord(char) < 127 for char in path))
        return ("unstable_feature_notice" if valid else "unknown_warning", valid)
    return "unknown_warning", False


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


class Recorder:
    def __init__(self):
        self.warnings: list[dict] = []

    def observe(self, event):
        if isinstance(event, dict) and event.get("method") == "warning":
            params = event.get("params")
            self.warnings.append(params if isinstance(params, dict) else {})

    def summaries(self, model: str, thread_id: str | None) -> list[dict]:
        result = []
        for params in self.warnings:
            message = params.get("message")
            category, validated = classify(message, model)
            data = message.encode("utf-8") if isinstance(message, str) else b""
            result.append({"category": category, "validated": validated,
                           "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                           "thread_targeted": isinstance(params.get("threadId"), str),
                           "thread_id_matches": params.get("threadId") == thread_id if thread_id else None,
                           "keywords": {word: bool(re.search(r"\b" + word + r"\b", message, re.I))
                                        if isinstance(message, str) else False for word in KEYWORDS}})
        return result

    def save(self, evidence_dir: Path) -> bool:
        if not self.warnings:
            return False
        evidence_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(evidence_dir, 0o700)
        fd = os.open(evidence_dir / "warning.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(self.warnings, output)
        return True


def run_preflight(transport, binary: str, evidence_dir: Path, *, drain_seconds=2.0,
                  capture_turn: bool = False) -> dict:
    recorder = Recorder()
    report = {"ok": False, "mode": "one-turn" if capture_turn else "preflight", "observed_turns": 0,
              "known_tokens": None, "internal_http_attempts": None,
              "usage_complete": not capture_turn,
              "inference_enabled": capture_turn, "warnings": [], "cleanup": False,
              "stop_code": None}
    with tempfile.TemporaryDirectory(prefix="hymem-luna-warning-empty-") as cwd:
        session = None
        thread_id = None
        try:
            session = transport.StdioSession(binary, cwd)
            original_receive = session.receive

            def receiving():
                event = original_receive()
                recorder.observe(event)
                return event

            session.receive = receiving
            admission = transport.inspect_preflight(session,
                base_instructions="Return only the requested JSON object." if capture_turn else "")
            thread_id = admission.get("_thread_id")
            report["config_isolation_admitted"] = admission.get("config_isolation_admitted") is True
            deadline = time.monotonic() + drain_seconds
            for _ in range(100):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    event = session.events.get(timeout=remaining)
                except queue.Empty:
                    break
                recorder.observe(event)
                if event is None:
                    break
            if capture_turn:
                if not report["config_isolation_admitted"] or not isinstance(thread_id, str) or not thread_id:
                    raise transport.SubscriptionTransportError("preflight_rejected")
                for params in recorder.warnings:
                    transport._validate_warning({"method": "warning", "params": params}, thread_id)
                report["observed_turns"] = 1
                _, used = transport._run_turn(session, thread_id, 'Return exactly {"ok":true}.')
                report["known_tokens"] = used
                report["usage_complete"] = True
            report["ok"] = report["config_isolation_admitted"]
        except transport.SubscriptionTransportError as exc:
            code = str(exc)
            report["stop_code"] = code if re.fullmatch(r"[A-Za-z0-9_:/-]{1,120}", code) else "transport_failure"
        except Exception:
            report["stop_code"] = "diagnostic_failure"
        finally:
            if session is not None:
                session.close()
                report["cleanup"] = session.process.poll() is not None
            report["warnings"] = recorder.summaries(transport.MODEL, thread_id)
            try:
                report["private_evidence_saved"] = recorder.save(evidence_dir)
            except OSError:
                report["ok"] = False
                report["stop_code"] = "evidence_save_failed"
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("binary", "transport", "candidate", "transport-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--evidence-dir")
    parser.add_argument("--capture-turn", action="store_true")
    args = parser.parse_args(argv)
    binary, source, candidate = (Path(args.binary), Path(args.transport), Path(args.candidate))
    if (not all(path.is_absolute() for path in (binary, source, candidate))
            or not binary.is_file() or not source.is_file()
            or not (candidate / "hymem/extraction/llm.py").is_file()
            or re.fullmatch(r"[0-9a-f]{64}", args.transport_sha256) is None
            or digest(source) != args.transport_sha256):
        print(json.dumps({"ok": False, "stop_code": "input_invalid"}))
        return 1
    evidence_dir = Path(args.evidence_dir) if args.evidence_dir else Path(tempfile.mkdtemp(prefix="hymem-luna-warning-"))
    if not evidence_dir.is_absolute() or evidence_dir.exists() and any(evidence_dir.iterdir()):
        print(json.dumps({"ok": False, "stop_code": "input_invalid"}))
        return 1
    try:
        sys.path.insert(0, str(candidate))
        spec = importlib.util.spec_from_file_location("luna_warning_pinned_transport", source)
        if spec is None or spec.loader is None:
            raise ImportError
        transport = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transport)
    except Exception:
        print(json.dumps({"ok": False, "stop_code": "input_invalid"}))
        return 1
    report = run_preflight(transport, str(binary), evidence_dir, capture_turn=args.capture_turn)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
