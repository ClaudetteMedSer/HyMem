"""Fresh private diagnostic adapter; no upload entry point or deployment authority.

Stage reviewed files manually after root review. Only remote-install,
remote-launch, remote-status and supervise are exposed. Full-suite success
is deliberately not asserted by the accepted focused/offline diagnosis gate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import importlib.util
import re

HEX64 = re.compile(r"[0-9a-f]{64}\Z")
FRAME = re.compile(r"hymem(?:/[A-Za-z_][A-Za-z_0-9]*)*/[A-Za-z_][A-Za-z_0-9]*\.py\Z")


def require(ok, code):
    if not ok:
        raise RuntimeError(code)


def project(raw):
    require(isinstance(raw, dict)
         and raw.get("status") in ("ready", "completed", "captured_failure",
                                    "budget_stopped", "error"), "summary_shape")
    result = {"status": raw["status"]}
    for name in ("completion_calls", "http_attempts", "llm_http_attempts",
                 "embedding_http_attempts", "llm_provider_attempts_reported",
                 "embedding_provider_attempts_reported", "chunks_processed",
                 "extractions_captured", "prepersist_captured",
                 "exception_events_captured", "exception_controlflow_skipped",
                 "prompt_tokens", "completion_tokens", "total_tokens"):
        if name in raw:
            value = raw[name]
            require(type(value) is int and 0 <= value <= 100000000,
                 "summary_count_invalid")
            result[name] = value
    require(result.get("completion_calls", 0) <= 128
         and result.get("llm_http_attempts", 0) <= 384
         and result.get("embedding_http_attempts", 0) <= 512
         and result.get("http_attempts", 0) <= 896,
         "summary_budget_exceeded")
    if all(name in result for name in ("http_attempts", "llm_http_attempts",
                                       "embedding_http_attempts")):
        require(result["http_attempts"] == result["llm_http_attempts"]
             + result["embedding_http_attempts"], "summary_attempt_disagreement")
    for name in ("cleanup_ok", "source_unchanged", "runtime_generation_verified",
                 "failure_captured", "instrumentation_capture_ok",
                 "token_usage_available", "accounting_verified",
                 "exception_capture_truncated"):
        if name in raw:
            require(type(raw[name]) is bool, "summary_boolean_invalid")
            result[name] = raw[name]
    for name in ("source_sha256", "phase1_sha256", "capture_sha256",
                 "session_sha256"):
        if name in raw:
            require(isinstance(raw[name], str) and HEX64.fullmatch(raw[name]),
                 "summary_hash_invalid")
            result[name] = raw[name]
    codes = {"error_type": {"ValueError", "RuntimeError", "TypeError",
                             "KeyError", "AssertionError", "TimeoutError",
                             "ConnectionError", "OSError", "MemoryError",
                             "BudgetStop", "InstrumentationStop", "Exception"},
             "budget_reason": {"completion_budget", "http_attempt_budget",
                               "llm_http_attempt_budget",
                               "embedding_http_attempt_budget",
                               "embedding_payload_invalid", "embedding_payload_limit",
                               "prepersist_capture_budget", "cooperative_deadline"}}
    for name, allowed in codes.items():
        if name in raw:
            require(raw[name] in allowed, "summary_static_code_invalid")
            result[name] = raw[name]
    frames = raw.get("candidate_frames", [])
    require(isinstance(frames, list) and len(frames) <= 12, "summary_frames_invalid")
    for frame in frames:
        require(isinstance(frame, dict) and set(frame) == {"path", "function", "line"}
             and isinstance(frame["path"], str) and FRAME.fullmatch(frame["path"])
             and isinstance(frame["function"], str) and frame["function"].isidentifier()
             and type(frame["line"]) is int and 1 <= frame["line"] <= 100000,
             "summary_frame_invalid")
    result["candidate_frames"] = frames
    return result



BASE = Path("/opt/stacks/hermes/instance1/home/.hermes/benchmarks/claim-conflict-20260925-zqkdtxky")
PARENT = BASE / "cold-replay-dream-v1"
ROOT = PARENT / "episode-shadow-dream-v1"
REPLAY = PARENT / "episode-shadow-replay-v1"
PARENT_HOST_SHA = "38df85589fbc561e287bc75329b3ab785af001dd0f6bc3024cbb09f629820f0c"
CANDIDATE_SHA = "5550a6e1d29692fbda28560b87120dc90eec4d6ddb01bcbc18eaea100307b576"
REPLAY_SHA = "9a05dfe4b6383b263e3d510a77e3550aab7f140036d538d0cdebf650a75ac823"
WORKER_SHA = "6edaa699b4d37c0337d41e607ebf4ee9f1c3ef028aaf3e7b9d1c1b30c5fbd380"
SUPERVISOR_SHA = "f486300d9292f86f8777b400358de7936e99ca07d65dfff85e5730b23ba82a9d"
INSTRUMENTED_SHA = "e095f534da3457c5f24f7688a8f2eb6bc6b4d6f8981b630abec26c2de49739d8"
OVERLAY = {
    "hymem/core/db.py": "ceab71516bb72424eaf1d49eed50b821fa95b37843f233959e0e051c5c469fc3",
    "hymem/dreaming/runner.py": "25387efe4ef6cf5ca6d6178f96a0c7872748cdb3758bbf68836c222027691f5a",
}


def configure_parent():
    path = PARENT / "claim_conflict_v64_dream_host.py"
    if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_HOST_SHA:
        raise RuntimeError("parent_host_pin_drift")
    spec = importlib.util.spec_from_file_location("episode_shadow_parent", path)
    parent = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parent)
    parent.ROOT = ROOT
    parent.SELF = ROOT / Path(__file__).name
    parent.WORK = ROOT / "work"
    parent.CANDIDATE = ROOT / "candidate"
    parent.PROOF_SOURCE = REPLAY / "candidate"
    parent.REFERENCE = ROOT / "reference.sqlite"
    parent.WORKER = ROOT / "claim_conflict_episode_shadow_dream.py"
    parent.OLD_HOST = ROOT / "claim_conflict_episode_shadow_dream_supervisor.py"
    parent.OLD_WORKER = ROOT / "claim_conflict_episode_shadow_dream_instrumented.py"
    parent.WORKER_SHA = WORKER_SHA
    parent.OLD_HOST_SHA = SUPERVISOR_SHA
    parent.OLD_WORKER_SHA = INSTRUMENTED_SHA
    original_dependencies = parent.dependencies

    def dependencies():
        shared, proof, helper = original_dependencies()
        shared.project = project
        return shared, proof, helper

    parent.dependencies = dependencies
    original_inventory = parent.expected_inventory

    def inventory(shared, proof, helper):
        # Reconstruct all 481 parent pins before applying the two accepted files.
        expected = original_inventory(shared, proof, helper)
        expected.update(OVERLAY)
        parent.need(hashlib.sha256(json.dumps(
            expected, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() == CANDIDATE_SHA, "new_candidate_manifest_drift")
        return expected

    def replay_gate(_proof):
        result = REPLAY / "result.json"
        parent.need(result.is_file() and not result.is_symlink()
                    and parent.sha(result) == REPLAY_SHA, "accepted_replay_pin_drift")
        value = json.loads(result.read_text())
        parent.need(value.get("status") == "verified"
                    and value.get("removed_surplus") == 8
                    and value.get("exact_vectors_preserved") is True
                    and value.get("semantic_rows_unchanged") is True
                    and value.get("repeat_noop") is True
                    and value.get("reopen_aligned") is True
                    and value.get("health_clean") is True, "accepted_replay_not_verified")
        return REPLAY_SHA

    parent.expected_inventory = inventory
    parent.proof_gate = replay_gate
    # This gate permits one diagnosis only; never reuse the parent's suite receipt.
    parent.suite_gate = lambda: None
    original_pins = parent.pins

    def pins(shared, proof, helper):
        value = original_pins(shared, proof, helper)
        value.update({"diagnostic_only": True, "new_candidate_full_suite_verified": False,
                      "candidate_sha256": CANDIDATE_SHA,
                      "accepted_parent_host_sha256": PARENT_HOST_SHA,
                      "max_completion_calls": 128, "max_llm_http": 384,
                      "max_embedding_http": 512, "max_total_http": 896,
                      "deadline_seconds": 2700})
        # Clear the inherited host identity as well as its unapplicable suite result.
        value["suite_host_sha256"] = None
        return value

    parent.pins = pins
    original_controller = parent.controller

    def controller():
        old = original_controller(parent.OLD_HOST)
        original_configure = old.configure

        def configure(helper, mode):
            command, mounts = original_configure(helper, mode)
            previous = "/diag/claim_conflict_instrumented_dream_v1.py"
            current = "/diag/claim_conflict_episode_shadow_dream_instrumented.py"
            command = [part.replace(previous, current) for part in command]
            mounts = [(src, current if dst == previous else dst, rw)
                      for src, dst, rw in mounts]
            return command, mounts

        old.configure = configure
        return old

    parent.controller = controller
    return parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("remote-install", "remote-launch",
                                          "remote-status", "supervise"))
    action = parser.parse_args().action
    try:
        parent = configure_parent()
        if action == "remote-install":
            result = parent.remote_install(*parent.dependencies())
        else:
            result = parent.controller().remote(action)
    except BaseException:
        result = {"status": "operation_failed_inspect_before_retry"}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
