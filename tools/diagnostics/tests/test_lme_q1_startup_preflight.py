"""Isolated diagnostic controls; no production memory, real keys or API calls.

Set LME_Q1_PREFLIGHT_CURRENT_SOURCE to a reconstructed/frozen source root and
LME_Q1_PREFLIGHT_LEGACY_SOURCE to R3 to exercise both genuine source versions.
These are helper tests outside the application's ordinary test suite.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
HELPER = REPO / "tools/diagnostics/lme_q1_startup_preflight.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("q1_startup_helper_test", HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dataset(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    rows = []
    for index in range(500):
        session_id = f"invented-session-{index}"
        rows.append({
            "question_id": "09ba9854" if index == 210 else f"invented-{index:04d}",
            "question_type": "single-session-user", "question": "What color is the invented kite?",
            "answer": "blue", "question_date": "2026/01/02 (Fri) 12:00",
            "haystack_sessions": [[{"role": "user", "content": "My invented kite is blue."}]],
            "haystack_session_ids": [session_id], "answer_session_ids": [session_id],
            "haystack_dates": ["2026/01/01 (Thu) 12:00"],
        })
    (data / "longmemeval_s_cleaned.json").write_text(json.dumps(rows))
    return data


CHILD = r'''
import importlib.util, json, os, pathlib, socket, sys
helper, source, data, output, mode = map(str, sys.argv[1:])
spec = importlib.util.spec_from_file_location("q1_startup_child", helper)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
source, data, output = map(pathlib.Path, (source, data, output))
output.mkdir()
prior = dict(os.environ)
try:
    report = module.run_probe(source=source, arguments=module.stock_arguments(source, data, output), output=output)
except BaseException as exc:
    report = getattr(exc, "q1_preflight_report", {"status": "failed", "exception_type": type(exc).__name__})
report["test_environment_restored"] = dict(os.environ) == prior
if mode in {"real_dns", "synthetic_events"} and report["status"] == "passed":
    events = {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
              "socket.gethostbyaddr", "socket.getnameinfo", "socket.sendto", "socket.sendmsg"}
    # Safety backstop: if the helper misses an event, fail before any outgoing
    # lookup/connection can actually occur. Installed after the helper's hook.
    def escaped_helper(event, args):
        if event in events:
            raise AssertionError("outbound event escaped diagnostic guard")
    sys.addaudithook(escaped_helper)
    operations = ({event: lambda event=event: sys.audit(event, "synthetic-only")
                   for event in sorted(events)} if mode == "synthetic_events" else {
        "getaddrinfo": lambda: socket.getaddrinfo("example.invalid", 443),
        "gethostbyname": lambda: socket.gethostbyname("example.invalid"),
        "gethostbyname_ex": lambda: socket.gethostbyname_ex("example.invalid"),
        "gethostbyaddr": lambda: socket.gethostbyaddr("192.0.2.1"),
        "getnameinfo": lambda: socket.getnameinfo(("192.0.2.1", 443), 0),
    })
    blocked = []
    for name, operation in operations.items():
        try:
            operation()
        except RuntimeError as exc:
            assert str(exc) == "q1_preflight_outbound_forbidden"
            blocked.append(name)
    report["test_outbound_still_blocked"] = sorted(blocked) == sorted(operations)
print(json.dumps(report, sort_keys=True))
'''


def child_probe(tmp_path, source, *, mode="ordinary"):
    data = dataset(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", CHILD, str(HELPER), str(source),
         str(data), str(tmp_path / "output"), mode],
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path),
             "HYMEM_LLM_EXTRA_BODY": '{"diagnostic_ambient_override":true}',
             "DEEPSEEK_API_KEY": "ambient-synthetic-key-must-not-be-used"},
        capture_output=True, text=True, timeout=90, check=False,
    )
    assert completed.returncode == 0, (completed.returncode, completed.stderr)
    assert "ambient-synthetic-key" not in completed.stdout + completed.stderr
    assert "diagnostic-placeholder-not-a-provider-credential" not in completed.stdout + completed.stderr
    assert completed.stderr == ""
    return json.loads(completed.stdout)


def current_source():
    return Path(os.environ.get("LME_Q1_PREFLIGHT_CURRENT_SOURCE", str(REPO))).resolve()


def test_real_current_client_and_physical_stock_cli_checkpoint_agree(tmp_path):
    report = child_probe(tmp_path, current_source())
    assert report["status"] == "passed", report
    assert report["phase"] == "complete"
    for field in ("standalone_producer_identity_exact", "runtime_transport_identity_exact",
                  "checkpoint_runtime_producer_matches", "stock_cli_pre_provider_boundary_reached",
                  "cli_checkpoint_handles_closed", "canonical_extra_body_environment_absent",
                  "test_environment_restored"):
        assert report[field] is True
    for field in ("runtime_clients_constructed", "runtime_clients_closed", "cli_checkpoints_observed"):
        assert report[field] == 1
    for field in ("provider_completions", "provider_http_attempts", "provider_successful_responses",
                  "outbound_operations_blocked", "completed_questions"):
        assert report[field] == 0
    assert report["real_credentials_loaded"] is False
    assert report["benchmark_execution_verified"] is False
    assert report["cleanup_failed"] is False
    assert report["private_home_created"] is True
    private_home = tmp_path / "output/home"
    assert private_home.is_dir() and not private_home.is_symlink()
    assert private_home.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("mode,blocked", [("real_dns", 5), ("synthetic_events", 7)])
def test_diagnostic_hook_remains_outbound_denied_after_probe(tmp_path, mode, blocked):
    report = child_probe(tmp_path, current_source(), mode=mode)
    assert report["status"] == "passed", report
    assert report["test_outbound_still_blocked"] is True
    assert report["outbound_operations_blocked"] == blocked
    assert report["provider_completions"] == report["provider_http_attempts"] == 0


def test_real_r3_current_model_route_fails_before_client_or_checkpoint(tmp_path):
    raw = os.environ.get("LME_Q1_PREFLIGHT_LEGACY_SOURCE")
    if not raw:
        pytest.skip("set LME_Q1_PREFLIGHT_LEGACY_SOURCE to the reconstructed R3 source")
    report = child_probe(tmp_path, Path(raw).resolve())
    assert report["status"] == "failed"
    assert report["phase"] == "standalone_producer_declaration", report
    assert report["exception_type"] == "ValueError"
    for field in ("runtime_clients_constructed", "runtime_clients_closed", "cli_checkpoints_observed",
                  "provider_completions", "provider_http_attempts", "outbound_operations_blocked"):
        assert report[field] == 0
    assert report["canonical_extra_body_environment_absent"] is True
    assert report["test_environment_restored"] is True


@pytest.mark.parametrize("kind", ["directory", "file", "symlink"])
def test_private_home_refuses_existing_material_before_imports(tmp_path, kind):
    helper = load_helper()
    output = tmp_path / "output"
    output.mkdir()
    private_home = output / "home"
    external = tmp_path / "untouched"
    external.write_text("preserve this existing material")
    if kind == "directory":
        private_home.mkdir()
        (private_home / "keep").write_text("existing configuration")
    elif kind == "file":
        private_home.write_text("existing file")
    else:
        private_home.symlink_to(external)
    old_environment = dict(os.environ)
    with pytest.raises(FileExistsError):
        helper.run_probe(source=REPO, arguments=helper.stock_arguments(REPO, tmp_path, output), output=output)
    assert dict(os.environ) == old_environment
    assert external.read_text() == "preserve this existing material"
    if kind == "directory":
        assert (private_home / "keep").read_text() == "existing configuration"
    elif kind == "file":
        assert private_home.read_text() == "existing file"
    else:
        assert private_home.is_symlink()


@pytest.mark.parametrize("flag", ["--api-key", "--hymem-api-key", "--answer-api-key", "--judge-api-key",
                                 "--resume-from", "--retry-failures", "--skip-extraction-canary", "--no-dream",
                                 "--api-key=synthetic-only", "--resume-from=old-checkpoint"])
def test_unsafe_or_noncanonical_recipe_fails_before_imports(flag):
    helper = load_helper()
    argv = helper.stock_arguments(Path("/source"), Path("/data"), Path("/output"))
    with pytest.raises(RuntimeError, match="q1_preflight_forbidden_recipe"):
        helper.request_recipe(argv + [flag, "synthetic-only"])


def test_stock_recipe_has_no_embedding_aggregation_or_granularity_overrides():
    helper = load_helper()
    argv = helper.stock_arguments(Path("/source"), Path("/data"), Path("/output"))
    helper.request_recipe(argv)
    for flag in ("--embeddings", "--aggregation-nodes", "--episode-granularity", "--retrieval-only"):
        assert flag not in argv
    assert argv[argv.index("--indexing-max-cycles") + 1] == "100"
    assert argv[argv.index("--indexing-timeout-s") + 1] == "3600"
