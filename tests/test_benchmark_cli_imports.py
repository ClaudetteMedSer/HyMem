"""Benchmark CLIs must run from a checkout without an installed HyMem package."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


# Ignore PYTHONPATH/PYTHONHOME and disable site initialization, including editable
# installation hooks. Keep the script directory (unlike -I) to model a real CLI.
_PYTHON = [sys.executable, "-E", "-s", "-S", "-B"]
_REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def clean_checkout(tmp_path_factory):
    work = tmp_path_factory.mktemp("benchmark-checkout")
    checkout = work / "checkout"
    outside = work / "unrelated-cwd"
    outside.mkdir()
    # Copy code/resources only: no .pth, installation metadata, or bytecode.
    for package in ("benchmarks", "hymem"):
        for source in (_REPO / package).rglob("*"):
            if source.is_file() and source.suffix in {".py", ".sql"}:
                destination = checkout / source.relative_to(_REPO)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)

    probe = subprocess.run(
        [*_PYTHON, "-c", "import importlib.util, sys; "
         "assert 'site' not in sys.modules; "
         "assert importlib.util.find_spec('hymem') is None; "
         "assert importlib.util.find_spec('benchmarks') is None"],
        cwd=outside, capture_output=True, text=True, timeout=30,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    return checkout, outside


def _run_cli(clean_checkout, script, *args, invocation="absolute"):
    checkout, outside = clean_checkout
    if invocation == "module":
        target = ["-m", f"benchmarks.{Path(script).stem}"]
    elif invocation == "relative":
        target = [str(Path("benchmarks") / script)]
    else:
        target = [str(checkout / "benchmarks" / script)]
    return subprocess.run(
        [*_PYTHON, *target, *map(str, args)],
        cwd=outside if invocation == "absolute" else checkout,
        env={key: value for key, value in os.environ.items()
             if not key.startswith(("HYMEM_", "OPENAI_", "DEEPSEEK_"))},
        capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("invocation", ["absolute", "relative", "module"])
def test_strictness_smoke_from_clean_checkout(clean_checkout, invocation):
    result = _run_cli(
        clean_checkout, "strictness.py", "--smoke", invocation=invocation,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["counts"] == {
        "expected": 3, "attempted": 3, "unique_attempted": 3,
        "total_attempts": 3, "completed": 2, "failed": 1, "missing": 0,
    }
    assert payload["accuracy"] == pytest.approx(1 / 3)
    assert payload["failure_ids"] == ["smoke-3"]


def test_sibling_import_uses_this_checkouts_shared_dependencies(clean_checkout):
    checkout, _ = clean_checkout
    result = subprocess.run(
        [*_PYTHON, "-c", "import strictness; "
         "import hymem.deadline as deadline; "
         "import hymem.dreaming.status as status; "
         "from pathlib import Path; "
         "root = Path.cwd().parent; "
         "assert Path(deadline.__file__).resolve() == root / 'hymem/deadline.py'; "
         "assert Path(status.__file__).resolve() == root / 'hymem/dreaming/status.py'; "
         "assert strictness.DeadlineExceeded is deadline.DeadlineExceeded; "
         "assert strictness.MonotonicDeadline is deadline.MonotonicDeadline"],
        cwd=checkout / "benchmarks", capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("invocation", ["absolute", "module"])
@pytest.mark.parametrize("script,expected_code", [
    ("lme_registry.py", 0), ("beam_registry.py", 1),
    ("msc_registry.py", 0), ("locomo_registry.py", 1),
    ("msc_adapter.py", 0), ("locomo_adapter.py", 0),
])
def test_benchmark_cli_help_from_clean_checkout(
    clean_checkout, script, expected_code, invocation,
):
    result = _run_cli(clean_checkout, script, "--help", invocation=invocation)
    # BEAM and LoCoMo's existing manual dispatch prints help with exit 1.
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_flipwatch_cli_help_from_clean_checkout(clean_checkout):
    result = _run_cli(clean_checkout, "flipwatch_classify.py", "--help")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize("invocation", ["absolute", "module"])
@pytest.mark.parametrize("case,expected_code,expected_text", [
    ("unevidenced", 1, "[UNEVIDENCED]"),
    ("evidenced", 0, "[EVIDENCED]"),
    ("confounded", 0, "confounded on 1 other key(s): top_k"),
])
def test_lme_arm_evidence_from_clean_checkout(
    clean_checkout, tmp_path, invocation, case, expected_code, expected_text,
):
    lever = "episode_granularity_enabled"
    arms = []
    for index, name in enumerate(("off", "on")):
        config = {"scale": "S", "sample": 0, "seed": 0, "top_k": 15}
        if case != "unevidenced":
            config[lever] = bool(index)
        if case == "confounded" and index:
            config["top_k"] = 30
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({
            "benchmark": "LongMemEval", "date": "2026-08-31T03:11:30+00:00",
            "config": config, "scores": {}, "per_question": [],
        }), encoding="utf-8")
        arms.append(path)
    result = _run_cli(
        clean_checkout, "lme_registry.py", "arms", *arms, "--lever", lever,
        invocation=invocation,
    )
    assert result.returncode == expected_code, result.stdout + result.stderr
    assert expected_text in result.stdout
