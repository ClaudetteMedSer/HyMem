"""Closed-inventory and destination controls for the R7 release freezer."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def freezer(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "r7_freeze_under_test", Path(__file__).parents[1] / "lme_r7_freeze.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tree = tmp_path / "release"
    files = {
        "hymem/minimal.py": "value = 1\n",
        "benchmarks/minimal.py": "value = 2\n",
        "tests/test_initial.py": "def test_initial(): pass\n",
        "README.md": "fixture\n",
    }
    for name, content in files.items():
        path = tree / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    baseline = {
        "source_sha256": {name: "unused" for name in files if name.startswith(("hymem/", "benchmarks/"))},
        "test_sha256": {"tests/test_initial.py": "unused"},
        "auxiliary_sha256": {"README.md": "unused"},
        "expected_skip_nodeids": [],
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline))
    monkeypatch.setattr(module, "R6_PIN", module.sha(baseline_path.read_bytes()))
    output = tmp_path / "frozen"
    manifest = tmp_path / "manifest.json"

    def run(manifest_path=manifest, mutate=None):
        def collect(*_args, **_kwargs):
            if mutate:
                mutate(tree)
            return SimpleNamespace(returncode=0, stdout="tests/test_initial.py::test_initial\n")
        monkeypatch.setattr(module.subprocess, "run", collect)
        monkeypatch.setattr(sys, "argv", ["freeze", "--tree", str(tree),
            "--output", str(output), "--baseline-manifest", str(baseline_path),
            "--manifest", str(manifest_path)])
        module.main()
    return tree, output, manifest, run


@pytest.mark.parametrize("destination", ["relative", "input", "output"])
def test_manifest_cannot_escape_inventory_boundary(freezer, destination):
    tree, output, manifest, run = freezer
    path = {"relative": Path("frozen/manifest.json"),
            "input": tree / "manifest.json", "output": output / "manifest.json"}[destination]
    with pytest.raises(AssertionError):
        run(path)
    assert not output.exists()
    assert not manifest.exists()


@pytest.mark.parametrize("name", ["tests/test_added.py", "hymem/added.py", "benchmarks/added.sql"])
def test_collection_cannot_hide_new_input_files(freezer, name):
    _, _, manifest, run = freezer
    with pytest.raises(AssertionError):
        run(mutate=lambda tree: (tree / name).write_text("# added during collection\n"))
    assert not manifest.exists()


def test_collection_cannot_hide_new_input_symlink(freezer):
    _, _, manifest, run = freezer
    with pytest.raises(AssertionError):
        run(mutate=lambda tree: (tree / "tests/alias").symlink_to(tree / "hymem", target_is_directory=True))
    assert not manifest.exists()


def test_stable_closed_inventory_freezes(freezer):
    tree, output, manifest, run = freezer
    run()
    recorded = json.loads(manifest.read_text())
    assert recorded["expected_nodeids"] == ["tests/test_initial.py::test_initial"]
    assert (output / "tests/test_initial.py").read_bytes() == (tree / "tests/test_initial.py").read_bytes()
    assert len([path for path in output.rglob("*") if path.is_file()]) == 4
