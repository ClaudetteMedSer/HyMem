"""Offline real-callstack repro for the frozen LME accounting classifier.

Run with ``/opt/anaconda3/bin/python3.13 -B``. Uses only the frozen candidate,
its StubLLMClient, and a disposable local SQLite store; no provider is called.
The output includes code paths/function names and finite labels, never prompts.
"""
from __future__ import annotations

import inspect
import importlib.util
import json
import logging
from pathlib import Path
import sys
import tempfile


REPO = Path(__file__).resolve().parents[3]
CANDIDATE = Path("/private/tmp/hymem-lme-diagnostic-offline-assembly-v3/candidate")
if not CANDIDATE.is_dir():
    raise SystemExit("frozen_candidate_unavailable")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(CANDIDATE))

from tools.diagnostics.luna_lme_diagnostic_v2 import AccountedClient  # noqa: E402
from hymem import HyMem, HyMemConfig  # noqa: E402
from hymem.extraction.llm import StubLLMClient  # noqa: E402
from hymem.extraction import chunk  # noqa: E402
from benchmarks import extraction_canary as canary  # noqa: E402


classifier = AccountedClient(delegate=None, candidate=CANDIDATE)
observed: dict[tuple[str, str], int] = {}


def candidate_producer() -> str:
    for frame in inspect.stack()[2:]:
        try:
            relative = Path(frame.filename).relative_to(CANDIDATE).as_posix()
        except ValueError:
            continue
        if relative.startswith(("hymem/extraction/", "hymem/dreaming/",
                                "hymem/query/")) and relative != "hymem/dreaming/runner.py":
            return relative + ":" + frame.function
    return "unknown"


class TraceClient(StubLLMClient):
    def complete(self, request):
        # Invoke the actual pinned classifier from the actual frozen call stack.
        # No AccountedClient dispatch/budget is used; StubLLM remains the only
        # completion implementation and cannot reach a provider.
        label = classifier._stage(False)
        producer = candidate_producer()
        key = (producer, label)
        observed[key] = observed.get(key, 0) + 1
        return super().complete(request)


def main() -> None:
    logging.disable(logging.CRITICAL)
    with tempfile.TemporaryDirectory(prefix="hymem-accounting-offline-") as scratch:
        llm = TraceClient(fixtures={
            "Return the JSON object now": '{"episodes":[],"summary":"","procedures":[]}',
            "Return the JSON object of narrative facts now": "[]",
        }, default="[]")
        hy = HyMem(HyMemConfig(root=Path(scratch)), llm=llm)
        try:
            hy.open_session("s")
            for index in range(10):
                hy.log_message("s", "user",
                    f"MedFlow deploy location {index} is fly.io region {index}.")
            hy.close_session("s")
            hy.dream()
            hy.augment("Where does MedFlow deploy?")
        finally:
            hy.close()
    # Exercise the real frozen staged grounding branch. A deliberately invalid
    # local stage reply is sufficient to observe the dispatch; it is never
    # published and cannot call a provider.
    spec = importlib.util.spec_from_file_location("frozen_canary_test_helpers",
        CANDIDATE / "tests/test_benchmark_extraction_canary.py")
    if spec is None or spec.loader is None:
        raise SystemExit("frozen_canary_helpers_unavailable")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)

    class StagedProbe:
        def complete(self, request):
            label = classifier._stage(False)
            key = (candidate_producer(), label)
            observed[key] = observed.get(key, 0) + 1
            return helpers._representative_response(request)

        def complete_stage(self, request, batch, stage, recheck):
            label = classifier._stage(True)
            key = (candidate_producer(), label)
            observed[key] = observed.get(key, 0) + 1
            return "{}"

    chunk.extract_chunk(StagedProbe(), canary._CANARY_CONTENT,
        source_records=canary._source_records(),
        completion_call_limit=canary.EXTRACTION_CANARY_MAX_COMPLETION_CALLS)
    rows = [{"callsite": callsite, "label": label, "calls": count}
            for (callsite, label), count in sorted(observed.items())]
    print(json.dumps(rows, sort_keys=True, separators=(",", ":")))
    expected = {
        ("hymem/extraction/chunk.py:single_attempt", "extraction"),
        ("hymem/extraction/chunk.py:grounding_call", "grounding"),
        ("hymem/dreaming/digest.py:extract_session_digest", "digest"),
        ("hymem/dreaming/user_profile.py:extract_user_profile", "unclassified"),
        ("hymem/dreaming/facts.py:_extract_facts_fresh", "unclassified"),
        ("hymem/query/rerank.py:llm_rerank", "unclassified"),
    }
    if not expected.issubset(observed):
        raise SystemExit("candidate_path_not_exercised")


if __name__ == "__main__":
    main()
