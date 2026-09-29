"""Bounded paired prompt diagnostic; no benchmark score or production mutation.

The frozen extractor and canary execute unchanged. Only its three system-prefix
strings are transformed at the client boundary; all other request bytes match.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_EARLY_PINS = {
    "luna_grounding_candidate.py": "51e03ff2bec29f7517db027b103163698c18b9d34603ab950a36f5d09255e96a",
    "luna_grounding_cases.py": "56147b137c71236aa16afc8b0f2f413b90b255c79cbcb370fed50ef71879103c",
    "luna_subscription_lme_warm_v2.py": "3de8840bb70e26972228c177f99294d5aef354a5821573d13b0122e8f9cdc567",
}
for _name, _expected in _EARLY_PINS.items():
    if hashlib.sha256((HERE / _name).read_bytes()).hexdigest() != _expected:
        raise RuntimeError("diagnostic_helper_source_drift")
import luna_grounding_candidate as builder
import luna_grounding_cases as cases
import luna_subscription_lme_warm_v2 as pinned

SCHEMA = "luna-grounding-paired-prompt-v1"
CASE_SHA256 = "56147b137c71236aa16afc8b0f2f413b90b255c79cbcb370fed50ef71879103c"
GROUNDED_PROMPT_SHA256 = "17ee5017c54a1ba255e0220aa8e246766127cb71ced65fb2e8c812034f6e184c"
UNITS = tuple((rep, kind, ident, arm)
              for rep in (1, 2)
              for kind, ids in (("control", cases.CASE_IDS), ("canary", ("strict",)))
              for ident in ids
              for arm in (("baseline", "candidate") if rep == 1
                          else ("candidate", "baseline")))


class ProbeStop(RuntimeError):
    pass


class MappingFault(BaseException):
    """Cannot be converted into an extractor semantic retry."""


def need(condition: bool, code: str) -> None:
    if not condition:
        raise ProbeStop(code)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def private_write(path: Path, value: object) -> None:
    encoded = json.dumps(value, sort_keys=True, default=str).encode()
    tmp = path.with_name(path.name + ".pending")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def load_prompt_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    need(spec is not None and spec.loader is not None, "prompt_import_invalid")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    need(Path(module.__file__).resolve() == path.resolve(), "prompt_import_mismatch")
    return module


def prompt_mapping(baseline, candidate) -> dict[str, str]:
    names = ("build_chunk_extraction_system", "build_chunk_empty_verification_system",
             "build_chunk_omission_verification_system")
    old = [getattr(baseline, name)() for name in names]
    new = [getattr(candidate, name)() for name in names]
    need(len(set(old)) == len(set(new)) == 3, "prompt_mapping_ambiguous")
    # Empty and omission suffixes must remain byte-identical.
    for index in (1, 2):
        need(old[index].startswith(old[0]) and new[index].startswith(new[0])
             and old[index][len(old[0]):] == new[index][len(new[0]):],
             "prompt_suffix_drift")
    return dict(zip(old, new))


class MappingClient:
    def __init__(self, delegate, mapping: dict[str, str], arm: str, on_request=None):
        self.delegate, self.mapping, self.arm = delegate, mapping, arm
        self.on_request = on_request
        self.request_log: list[dict] = []
        self.responses: list[str] = []

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def complete(self, request):
        def fail(code):
            self.delegate.budget.halt(code)
            raise MappingFault(code)
        if request.system not in self.mapping:
            fail("unexpected_system_prompt")
        changed = replace(request, system=self.mapping[request.system]
                          if self.arm == "candidate" else request.system)
        before, after = asdict(request), asdict(changed)
        if ({k: v for k, v in before.items() if k != "system"} !=
             {k: v for k, v in after.items() if k != "system"}):
            fail("request_field_drift")
        request_meta = {
            "before_system_sha256": sha(request.system.encode()),
            "after_system_sha256": sha(changed.system.encode()),
            "before_user_sha256": sha(request.user.encode()),
            "after_user_sha256": sha(changed.user.encode()),
            "before_request_sha256": sha(json.dumps(before, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()),
            "after_request_sha256": sha(json.dumps(after, sort_keys=True,
                separators=(",", ":"), ensure_ascii=False).encode()),
            "before_request": before, "after_request": after,
        }
        self.request_log.append(request_meta)
        if self.on_request:
            try:
                self.on_request(self.request_log, self.responses)
            except BaseException:
                fail("private_evidence_write_failure")
        answer = self.delegate.complete(changed)
        self.responses.append(answer)
        return answer


def schedule() -> tuple[tuple[int, str, str, str], ...]:
    need(len(UNITS) == 44 and len(cases.CASE_IDS) == 10, "schedule_invalid")
    return UNITS


def finish_public_result(public: dict, result: dict, output: Path,
                         source_pins: dict) -> None:
    """Retain known paid-work accounting even if final private persistence fails."""
    public.update({key: result[key] for key in (
        "completed_and_clean", "completed_units", "process_groups_absent",
        "all_candidate_passed", "units", "budget")})
    public["stop_code"] = result["campaign_stop"]
    public["source_pins"] = source_pins
    try:
        private_write(output / "private-result.json", result)
    except BaseException:
        public["completed_and_clean"] = False
        public["stop_code"] = "private_result_write_failure"


def run(*, binary: str, concurrent, warm, canary, chunk,
        mapping: dict[str, str], output: Path, progress=None,
        client_factory=None) -> dict:
    limits = concurrent.BudgetLimits(192, 2_000_000, 1800)
    budget = concurrent.SharedBudget(limits, max_in_flight=1)
    results = []
    pids: list[int] = []
    started = time.monotonic()

    class TrackingSession(warm.WarmSession):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            pids.append(self.process.pid)

    def publish():
        state = {"schema": SCHEMA, "units": results, "budget": budget.snapshot(),
                 "elapsed_seconds": round(time.monotonic() - started, 3),
                 "campaign_stop": budget.snapshot()["stop_code"]}
        if progress:
            try:
                progress(state)
            except BaseException:
                budget.halt("private_progress_write_failure")
                state["budget"] = budget.snapshot()
                state["campaign_stop"] = state["budget"]["stop_code"]
        return state

    for rep, kind, ident, arm in schedule():
        state = budget.snapshot()
        if state["stopped"] or not state["usage_complete"] or state["in_flight"] != 0:
            budget.halt("accounting_or_campaign_stop")
            break
        key = f"r{rep}-{kind}-{ident}-{arm}"
        cap = concurrent.BudgetLimits(24, 300_000, 600) if kind == "canary" else concurrent.BudgetLimits(8, 100_000, 180)
        unit_start = time.monotonic()
        try:
            client = (client_factory(key, cap, budget) if client_factory else
                      warm.WarmSubscriptionClient(binary, budget, key, cap,
                         session_factory=TrackingSession, max_requests=16, max_age_seconds=300))
        except BaseException:
            budget.halt("client_creation_failure")
            break
        wrapped = MappingClient(client, mapping, arm,
            on_request=lambda requests, responses: private_write(
                output / f"private-{key}-requests.json",
                {"requests": requests, "responses": responses}))
        item = {"id": key, "repetition": rep, "kind": kind, "case_id": ident,
                "arm": arm, "passed": False, "cleanup_ok": False}
        evidence = {}
        stop_code = None
        try:
            if kind == "canary":
                grade = pinned.old.experimental_canary(canary, chunk, wrapped,
                    evidence=lambda value: evidence.update(value))
                item["grade"] = {key: grade.get(key) for key in (
                    "schema", "passed", "fixture_sha256", "matched_core_claims",
                    "expected_core_claims", "completion_calls", "execution_path_valid",
                    "core_execution_path_exact", "initial_prepartition_leaves",
                    "failure_code")}
            else:
                case = next(c for c in cases.CASES if c.case_id == ident)
                extracted = chunk.extract_chunk(wrapped, case.text,
                    source_records=case.source_records, completion_call_limit=8)
                grade = cases.grade_case(ident, extracted)
                item["grade"] = grade
                evidence["extraction_result"] = repr(extracted)
            completion_calls = (grade.get("completion_calls") if kind == "canary"
                                else extracted.completion_calls)
            item["passed"] = grade["passed"] is True
            ledger = budget.snapshot()
            need(not ledger["stopped"] and client.usage_complete
                 and client.observed_tokens is not None and ledger["reserved"] == 0
                 and ledger["in_flight"] == 0 and ledger["usage_complete"],
                 "usage_incomplete")
            need(client.observed_turns == len(wrapped.request_log)
                 == len(wrapped.responses)
                 == completion_calls,
                 "call_accounting_mismatch")
        except concurrent.ConcurrentStop:
            stop_code = budget.snapshot()["stop_code"] or "transport_failure"
        except ProbeStop as exc:
            stop_code = str(exc)
        except MappingFault as exc:
            stop_code = str(exc)
        except BaseException:
            stop_code = "unit_runtime_failure"
            evidence["traceback"] = traceback.format_exc()
        finally:
            try:
                client.close()
                item["cleanup_ok"] = True
            except BaseException:
                stop_code = "cleanup_failure"
            for pid in pids:
                try:
                    os.killpg(pid, 0)
                except ProcessLookupError:
                    pass
                except BaseException:
                    stop_code = "process_group_check_failure"
                else:
                    stop_code = "process_group_remaining"
            item.update({"completion_calls": client.observed_turns,
                         "known_tokens": client.observed_tokens,
                         "usage_complete": client.usage_complete,
                         "elapsed_seconds": round(time.monotonic() - unit_start, 3),
                         "request_count": len(wrapped.request_log),
                         "stop_code": stop_code})
            try:
                private_write(output / f"private-{key}.json", {
                    "unit": item, "requests": wrapped.request_log,
                    "responses": wrapped.responses, "evidence": evidence})
            except BaseException:
                stop_code = "private_evidence_write_failure"
                item["stop_code"] = stop_code
            results.append(item)
            if stop_code:
                budget.halt(stop_code)
            publish()
        if stop_code:
            break
    absent = True
    for pid in pids:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            pass
        except BaseException:
            absent = False
            budget.halt("process_group_check_failure")
        else:
            absent = False
    if not absent:
        budget.halt(budget.snapshot()["stop_code"] or "process_group_remaining")
    state = publish()
    state["process_groups_absent"] = absent
    state["completed_units"] = len(results)
    candidate_units = [x for x in results if x["arm"] == "candidate"]
    state["all_candidate_passed"] = (len(candidate_units) == 22
                                     and all(x["passed"] for x in candidate_units))
    state["completed_and_clean"] = (len(results) == 44 and absent
        and state["budget"]["usage_complete"] and state["budget"]["in_flight"] == 0
        and all(x["cleanup_ok"] and x["stop_code"] is None for x in results)
        and not state["budget"]["stopped"])
    return state


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    for arg in ("binary", "base-transport", "concurrent-transport", "warm-transport",
                "candidate", "inventory", "grounded-prompt", "output-dir"):
        parser.add_argument("--" + arg, required=True)
    parser.add_argument("--inventory-sha256", required=True)
    parser.add_argument("--grounded-prompt-sha256", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    public = {"schema": SCHEMA, "completed_and_clean": False,
              "paired_prompt_diagnostic": True, "full_candidate_lme_validation": False}
    try:
        candidate = Path(args.candidate)
        inventory = Path(args.inventory)
        prompt_path = Path(args.grounded_prompt)
        paths = (Path(args.binary), Path(args.base_transport),
                 Path(args.concurrent_transport), Path(args.warm_transport),
                 candidate, inventory, prompt_path, output)
        need(all(p.is_absolute() for p in paths), "path_not_absolute")
        need(not output.exists() and not output.is_symlink()
             and output.parent.is_dir(), "output_not_fresh")
        need(candidate.is_dir() and not candidate.is_symlink()
             and all(pinned.old.regular_absolute(p) for p in (
                 Path(args.binary), Path(args.base_transport),
                 Path(args.concurrent_transport), Path(args.warm_transport),
                 inventory, prompt_path)), "input_invalid")
        need(builder.sha((candidate / builder.PROMPT_RELATIVE).read_bytes()) ==
             builder.ORIGINAL_PROMPT_SHA256, "frozen_prompt_drift")
        grounded_bytes = builder.derive_prompt((candidate / builder.PROMPT_RELATIVE).read_bytes())
        need(prompt_path.read_bytes() == grounded_bytes and
             builder.sha(grounded_bytes) == args.grounded_prompt_sha256,
             "grounded_prompt_drift")
        need(sha(Path(cases.__file__).read_bytes()) == CASE_SHA256,
             "case_source_drift")
        need(args.grounded_prompt_sha256 == GROUNDED_PROMPT_SHA256,
             "grounded_prompt_pin_mismatch")
        builder.source_map(json.loads(inventory.read_text()))
        need(args.inventory_sha256 == builder.sha(inventory.read_bytes()),
             "inventory_stamp_drift")
        pinned.old.verify_inventory(candidate, inventory, args.inventory_sha256)
        need(pinned.old.digest(Path(pinned.old.__file__)) == pinned.PILOT_SHA256,
             "pilot_source_drift")
        for path, expected in ((Path(args.base_transport), pinned.old.TRANSPORT_SHA256),
                               (Path(args.concurrent_transport), pinned.CONCURRENT_SHA256),
                               (Path(args.warm_transport), pinned.WARM_SHA256)):
            need(builder.sha(path.read_bytes()) == expected, "transport_source_drift")
        base_path = Path(args.base_transport)
        concurrent_path = Path(args.concurrent_transport)
        warm_path = Path(args.warm_transport)
        need(base_path.parent == concurrent_path.parent == warm_path.parent
             and base_path.name == "codex_subscription.py"
             and concurrent_path.name == "codex_subscription_concurrent_v2.py"
             and warm_path.name == "codex_subscription_warm_v2.py",
             "transport_layout_invalid")
        for name, module in tuple(sys.modules.items()):
            if name == "hymem" or name.startswith("hymem.") or name == "benchmarks" or name.startswith("benchmarks."):
                file = getattr(module, "__file__", None)
                need(file is None or Path(file).resolve().is_relative_to(candidate.resolve()),
                     "cached_source_mismatch")
        sys.path.insert(0, str(candidate))
        from benchmarks import extraction_canary as canary
        from hymem.extraction import chunk
        from hymem.extraction import prompts as baseline
        for module in (canary, chunk, baseline):
            need(Path(module.__file__).resolve().is_relative_to(candidate.resolve()),
                 "loaded_source_mismatch")
        grounded = load_prompt_file(prompt_path, "pinned_grounded_prompt")
        mapping = prompt_mapping(baseline, grounded)
        # Import pinned transport only after frozen LLMRequest is on sys.path.
        spec = importlib.util.spec_from_file_location("pinned_grounding_warm", warm_path)
        need(spec is not None and spec.loader is not None, "transport_import_invalid")
        warm = importlib.util.module_from_spec(spec)
        sys.modules[warm.__name__] = warm
        spec.loader.exec_module(warm)
        concurrent = warm.concurrent
        output.mkdir(mode=0o700)
        def progress(value):
            private_write(output / "private-progress.json", value)
        with (output / "private-stderr.log").open("w") as log:
            os.chmod(output / "private-stderr.log", 0o600)
            with redirect_stdout(log), redirect_stderr(log):
                result = run(binary=args.binary, concurrent=concurrent, warm=warm,
                             canary=canary, chunk=chunk, mapping=mapping,
                             output=output, progress=progress)
        source_pins = {
            "source_files_verified": 508,
            "original_map_sha256": builder.ORIGINAL_MAP_SHA256,
            "original_inventory_sha256": args.inventory_sha256,
            "grounded_prompt_sha256": args.grounded_prompt_sha256,
            "case_source_sha256": sha(Path(cases.__file__).read_bytes()),
            "builder_sha256": sha(Path(builder.__file__).read_bytes()),
            "runner_sha256": sha(Path(__file__).read_bytes()),
            "pilot_sha256": pinned.PILOT_SHA256,
            "base_transport_sha256": pinned.old.TRANSPORT_SHA256,
            "concurrent_transport_sha256": pinned.CONCURRENT_SHA256,
            "warm_transport_sha256": pinned.WARM_SHA256}
        finish_public_result(public, result, output, source_pins)
    except BaseException:
        public["stop_code"] = "preparation_or_runtime_failure"
        if output.is_dir():
            try:
                private_write(output / "private-failure.json", {"traceback": traceback.format_exc()})
            except BaseException:
                public["stop_code"] = "private_evidence_write_failure"
    print(json.dumps(public, sort_keys=True))
    return 0 if public["completed_and_clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
