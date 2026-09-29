"""Provider-free real one-question R6 producer/CLI startup diagnostic.

Run only in a dedicated process. Its permanent audit hook forbids DNS/outbound
traffic, while allowing local SDK/event-loop setup. No credential file is read.
The only CLI hooks observe its real checkpoint and stop before reader creation;
producer identity, manifest construction and checkpoint cleanup remain genuine.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys

MODEL = "deepseek-flash"
ENDPOINT = "https://api.deepseek.com"
SYNTHETIC_KEY = "diagnostic-placeholder-not-a-provider-credential"
SAMPLE = 1
SEED = 0
QUESTION_IDS = ['gpt4_483dd43c']


class PreProviderBoundaryReached(BaseException):
    """Deliberate stop, including the stock CLI's BaseException cleanup."""


def need(value, code):
    if not value:
        raise RuntimeError(code)


def stock_arguments(source: Path, data_dir: Path, output: Path) -> list[str]:
    """The fixed label-blind stock recipe; no resume, rerolls or admission bypass."""
    args = [str(source / "benchmarks/longmemeval_adapter.py"),
            "--scales", "S", "--sample", "1", "--seed", "0", "--workers", "1",
            "--top-k", "15", "--auto-ability", "--permissive-default",
            "--indexing-max-cycles", "100", "--indexing-timeout-s", "3600",
            "--indexing-require-healthy", "--keep-db", "--no-prereg",
            "--protocol-split", "full", "--judge-protocol", "legacy-custom",
            "--data-dir", str(data_dir), "--results-dir", str(output),
            "--checkpoint", str(output / "checkpoint.json")]
    for role in ("hymem", "answer", "judge"):
        args += ["--" + role + "-model", MODEL, "--" + role + "-base-url", ENDPOINT]
    args += ["--hymem-thinking", "disabled"]
    for role in ("answer", "judge"):
        args += ["--" + role + "-extra-body", '{"thinking":{"type":"disabled"}}']
    return args


def request_recipe(arguments):
    need(type(arguments) is list and all(type(item) is str for item in arguments), "q1_preflight_argv")

    def value(flag):
        need(arguments.count(flag) == 1 and arguments.index(flag) + 1 < len(arguments), "q1_preflight_flag")
        return arguments[arguments.index(flag) + 1]

    for role in ("hymem", "answer", "judge"):
        need(value("--" + role + "-model") == MODEL
             and value("--" + role + "-base-url") == ENDPOINT, "q1_preflight_route")
    need(value("--hymem-thinking") == "disabled", "q1_preflight_thinking")
    for role in ("answer", "judge"):
        need(json.loads(value("--" + role + "-extra-body")) == {"thinking": {"type": "disabled"}},
             "q1_preflight_extra_body")
    switches = {item.split("=", 1)[0] for item in arguments[1:] if item.startswith("--")}
    need(not any(flag in switches for flag in (
        "--api-key", "--answer-api-key", "--judge-api-key", "--hymem-api-key",
        "--resume-from", "--retry-failures", "--skip-extraction-canary", "--no-dream")),
        "q1_preflight_forbidden_recipe")

    expected_values = {
        "--scales": "S", "--sample": "1", "--seed": "0", "--workers": "1",
        "--top-k": "15", "--indexing-max-cycles": "100",
        "--indexing-timeout-s": "3600", "--protocol-split": "full",
        "--judge-protocol": "legacy-custom", "--hymem-thinking": "disabled",
        **{"--" + role + "-model": MODEL for role in ("hymem", "answer", "judge")},
        **{"--" + role + "-base-url": ENDPOINT for role in ("hymem", "answer", "judge")},
    }
    for flag, expected in expected_values.items():
        need(value(flag) == expected, "sample8_preflight_fixed_recipe")
    flags = {"--auto-ability", "--permissive-default", "--indexing-require-healthy",
             "--keep-db", "--no-prereg"}
    paths = {"--data-dir", "--results-dir", "--checkpoint"}
    bodies = {"--answer-extra-body", "--judge-extra-body"}
    need(switches == set(expected_values) | flags | paths | bodies,
         "sample8_preflight_unknown_flags")
    for flag in flags:
        need(arguments.count(flag) == 1, "sample8_preflight_duplicate_flag")
    for flag in paths:
        need(bool(value(flag)), "sample8_preflight_path_flag")
    need(len(arguments) == 1 + len(flags) + 2 * (len(expected_values) + len(paths) + len(bodies)),
         "sample8_preflight_extra_arguments")


def validate_expected_ids(expected_question_ids):
    need(type(expected_question_ids) is list and len(expected_question_ids) == SAMPLE
         and all(type(value) is str and value and value.strip() == value
                 for value in expected_question_ids)
         and len(set(expected_question_ids)) == SAMPLE
         and expected_question_ids == QUESTION_IDS, "sample8_preflight_expected_ids")



def standalone_binding():
    from hymem.contrib.openai_client import (
        DEFAULT_LLM_TIMEOUT_SECONDS, openai_compatible_producer_declaration,
    )
    from hymem.extraction.producer import producer_binding_from_typed_declaration
    return producer_binding_from_typed_declaration(
        openai_compatible_producer_declaration(
            model=MODEL, endpoint=ENDPOINT, thinking_mode="disabled",
            effective_extra_body={"thinking": {"type": "disabled"}},
            transport_package_version=importlib.metadata.version("openai"),
            request_timeout_seconds=DEFAULT_LLM_TIMEOUT_SECONDS,
            deployment_revision_sha256=None, deployment_tenant_sha256=None,
            require_consistent_thinking=True,
        ), declaration_hook="aggregation_producer_declaration",
    )


def runtime_binding(report):
    from hymem.contrib.openai_client import OpenAICompatibleClient
    from hymem.extraction.producer import producer_binding_from_typed_declaration
    client = None
    primary = None
    try:
        report["phase"] = "runtime_client_construction"
        client = OpenAICompatibleClient(api_key=SYNTHETIC_KEY, base_url=ENDPOINT,
                                        model=MODEL, thinking="disabled")
        report["runtime_clients_constructed"] += 1
        report["phase"] = "runtime_client_identity"
        need(client.transport_integrity_ok is True, "q1_preflight_runtime_transport_identity")
        binding = producer_binding_from_typed_declaration(
            client.aggregation_producer_declaration(), declaration_hook="aggregation_producer_declaration")
        need(binding.get("identity_exact") is True and binding.get("reuse_scope") == "durable",
             "q1_preflight_runtime_producer_inexact")
        report["runtime_transport_identity_exact"] = True
        return binding
    except BaseException as exc:
        primary = exc
        raise
    finally:
        if client is not None:
            cleanup_error = None
            try:
                for field, recorded in (("call_count", "provider_completions"),
                                        ("request_attempts", "provider_http_attempts"),
                                        ("successful_responses", "provider_successful_responses")):
                    value = getattr(client, field, None)
                    report[recorded] = value if type(value) is int and value >= 0 else None
                    need(type(value) is int and value == 0, "q1_preflight_runtime_provider_activity")
            except BaseException as exc:
                cleanup_error = exc
            try:
                client.close()
                need(client._closed is True and client._owned_http_client.is_closed is True,
                     "q1_preflight_runtime_close")
                report["runtime_clients_closed"] += 1
            except BaseException as exc:
                cleanup_error = cleanup_error or exc
            if cleanup_error is not None:
                report["cleanup_failed"] = True
                if primary is not None:
                    primary.add_note("q1_preflight_runtime_cleanup_failed")
                else:
                    raise cleanup_error


def stock_cli_boundary(source, arguments, output, runtime, report, expected_question_ids):
    from benchmarks import longmemeval_adapter as adapter
    need(Path(adapter.__file__).resolve() == source / "benchmarks/longmemeval_adapter.py",
         "q1_preflight_adapter_source")
    original_client, original_checkpoint, original_argv = adapter.LLMClient, adapter.AtomicCheckpoint, sys.argv
    ledgers, leases, reached = [], [], []
    diagnostic = output / "cli-preflight"
    diagnostic.mkdir(mode=0o700)
    arguments = list(arguments)
    for flag, value in (("--results-dir", str(diagnostic)),
                        ("--checkpoint", str(diagnostic / "checkpoint.json"))):
        arguments[arguments.index(flag) + 1] = value

    def observe_checkpoint(*args, **kwargs):
        ledger = original_checkpoint(*args, **kwargs)
        ledgers.append(ledger)
        leases.append(ledger._lease)
        return ledger

    def provider_boundary(model, api_key, **kwargs):
        need(model == MODEL and api_key == SYNTHETIC_KEY
             and kwargs == {"base_url": ENDPOINT, "extra_body": {"thinking": {"type": "disabled"}}},
             "q1_preflight_reader_constructor_recipe")
        reached.append(True)
        raise PreProviderBoundaryReached()

    adapter.AtomicCheckpoint, adapter.LLMClient = observe_checkpoint, provider_boundary
    sys.argv = arguments
    caught = primary = None
    report["phase"] = "stock_cli_startup"
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            adapter.main()
    except PreProviderBoundaryReached as exc:
        caught = exc
    except BaseException as exc:
        primary = exc
        raise
    finally:
        sys.argv = original_argv
        adapter.LLMClient, adapter.AtomicCheckpoint = original_client, original_checkpoint
        report["cli_checkpoints_observed"] = len(ledgers)
        report["cli_checkpoint_handles_closed"] = (
            len(ledgers) == 1 and ledgers[0]._lease is None
            and leases[0] is not None and leases[0]._fd is None
        )
        for ledger in reversed(ledgers):
            if ledger._lease is not None:
                report["cleanup_failed"] = True
                try:
                    ledger.close()
                except BaseException:
                    if primary is not None:
                        primary.add_note("q1_preflight_observed_checkpoint_cleanup_failed")
                    elif caught is not None:
                        caught.add_note("q1_preflight_observed_checkpoint_cleanup_failed")
                    else:
                        raise
    report["phase"] = "stock_cli_postconditions"
    need(caught is not None and len(reached) == 1, "q1_preflight_boundary_not_reached")
    need(not getattr(caught, "__notes__", None) and not report["cleanup_failed"]
         and report["cli_checkpoint_handles_closed"], "q1_preflight_cli_cleanup")
    raw = (diagnostic / "checkpoint.json").read_bytes()
    snapshot = json.loads(raw)
    need(snapshot["expected_ids"] == expected_question_ids
         and snapshot["entries"] == {} and snapshot["execution_segments"] == []
         and snapshot["status"] == "running", "q1_preflight_checkpoint_activity")
    recorded = snapshot["manifest"]["models"]["memory_pipeline"]["aggregation_producer"]
    need(recorded == runtime, "q1_preflight_checkpoint_runtime_producer_mismatch")
    need(snapshot["manifest"]["models"]["memory_pipeline"]["model"] == MODEL,
         "q1_preflight_checkpoint_model")
    report.update(stock_cli_pre_provider_boundary_reached=True,
                  checkpoint_runtime_producer_matches=True,
                  cli_checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                  cli_manifest_run_id=snapshot["manifest"]["run_id"])


def run_probe(*, source, arguments, output, expected_question_ids):
    validate_expected_ids(expected_question_ids)
    source, output = Path(source), Path(output)
    need(source.resolve() == source and source.is_dir() and output.resolve() == output
         and output.is_dir(), "q1_preflight_paths")
    request_recipe(arguments)
    need(arguments[0] == str(source / "benchmarks/longmemeval_adapter.py"), "q1_preflight_argv_source")
    # No shared /tmp home, inherited configuration, or reusable diagnostic
    # directory. Preserve this new private directory on both success/failure.
    private_home = output / "home"
    private_home.mkdir(mode=0o700)
    report = {"schema": "r6-lme-failed-question-pre-provider-startup-v1", "status": "failed", "phase": "initialization",
              "requested_model": MODEL, "endpoint": ENDPOINT,
              "standalone_producer_identity_exact": False, "runtime_transport_identity_exact": False,
              "runtime_clients_constructed": 0, "runtime_clients_closed": 0,
              "stock_cli_pre_provider_boundary_reached": False, "checkpoint_runtime_producer_matches": False,
              "cli_checkpoints_observed": 0, "cli_checkpoint_handles_closed": False,
              "cleanup_failed": False, "provider_completions": 0, "provider_http_attempts": 0,
              "outbound_operations_blocked": 0, "completed_questions": 0,
              "real_credentials_loaded": False, "benchmark_execution_verified": False,
              "private_home_created": True}

    def deny_outbound(event, _args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                     "socket.gethostbyaddr", "socket.getnameinfo", "socket.sendto", "socket.sendmsg"}:
            report["outbound_operations_blocked"] += 1
            raise RuntimeError("q1_preflight_outbound_forbidden")

    sys.addaudithook(deny_outbound)
    sys.dont_write_bytecode = True
    old_environment, old_path = dict(os.environ), list(sys.path)
    os.environ.clear()
    os.environ.update(PATH="/usr/bin:/bin", HOME=str(private_home), PYTHONDONTWRITEBYTECODE="1",
                      PYTHONNOUSERSITE="1", DEEPSEEK_API_KEY=SYNTHETIC_KEY)
    sys.path[:0] = [str(source), str(source / "benchmarks")]
    try:
        need("HYMEM_LLM_EXTRA_BODY" not in os.environ, "q1_preflight_ambient_extra_body")
        report["canonical_extra_body_environment_absent"] = True
        import hymem
        need(Path(hymem.__file__).resolve() == source / "hymem/__init__.py", "q1_preflight_import_source")
        report["phase"] = "standalone_producer_declaration"
        standalone = standalone_binding()
        need(standalone.get("identity_exact") is True and standalone.get("reuse_scope") == "durable",
             "q1_preflight_standalone_producer_inexact")
        report["standalone_producer_identity_exact"] = True
        runtime = runtime_binding(report)
        need(runtime == standalone, "q1_preflight_standalone_runtime_producer_mismatch")
        stock_cli_boundary(source, arguments, output, runtime, report, expected_question_ids)
        need(report["outbound_operations_blocked"] == 0 and report["runtime_clients_constructed"] == 1
             and report["runtime_clients_closed"] == 1, "q1_preflight_activity_or_cleanup")
        report.update(status="passed", phase="complete", producer_identity_sha256=runtime["identity_sha256"],
                      expected_question_ids=list(expected_question_ids), expected_question_count=SAMPLE)
        return report
    except BaseException as exc:
        report["exception_type"] = type(exc).__name__
        exc.q1_preflight_report = report
        raise
    finally:
        os.environ.clear()
        os.environ.update(old_environment)
        sys.path[:] = old_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh diagnostic directory; never reused")
    parser.add_argument("--question-ids-json", required=True, help="Exact predeclared one-ID JSON list")
    args = parser.parse_args()
    args.output.mkdir(mode=0o700)
    try:
        report = run_probe(source=args.source,
                           arguments=stock_arguments(args.source, args.data_dir, args.output),
                           output=args.output,
                           expected_question_ids=json.loads(args.question_ids_json))
    except BaseException as exc:
        report = getattr(exc, "q1_preflight_report", {
            "status": "failed", "exception_type": type(exc).__name__, "benchmark_execution_verified": False})
    with (args.output / "startup-report.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
