"""Synthetic checks for the isolated candidate's offline test runner."""
import importlib.util
from pathlib import Path
import socket
import json
import subprocess
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "lme_offline_gate", Path(__file__).parents[1] / "lme_offline_gate.py"
)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def maps():
    return {
        "source_sha256": {"hymem/a.py": "a" * 64},
        "test_sha256": {"tests/test_a.py": "b" * 64},
        "auxiliary_sha256": {"README.md": "c" * 64},
    }


@pytest.mark.parametrize("executable", (
    "/home/node/hymem-env/bin/python3", "/opt/anaconda3/bin/python",
))
def test_runtime_path_uses_active_interpreter_and_fixed_system_suffix(monkeypatch, executable):
    monkeypatch.setattr(gate.sys, "executable", executable)
    monkeypatch.setenv("PATH", "/synthetic/untrusted-bin")
    assert gate.runtime_search_path() == (
        str(Path(executable).parent) + ":/usr/bin:/bin:/usr/sbin:/sbin"
    )


@pytest.mark.parametrize("bad_path", ("../a.py", "/tmp/a.py", "a/../b.py", "./a.py"))
def test_manifest_rejects_unsafe_paths(bad_path):
    manifest = maps()
    manifest["source_sha256"] = {bad_path: "a" * 64}
    with pytest.raises(ValueError, match="unsafe"):
        gate.expected_files(manifest)


def test_manifest_rejects_overlapping_maps():
    manifest = maps()
    manifest["test_sha256"] = manifest["source_sha256"]
    with pytest.raises(ValueError, match="overlapping"):
        gate.expected_files(manifest)


def test_tree_identity_detects_addition_change_and_symlink(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original\n")
    expected = {"source.py": gate.digest(source)}
    gate.verify_tree(tmp_path, expected)
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="hash differs"):
        gate.verify_tree(tmp_path, expected)
    source.write_text("original\n")
    extra = tmp_path / "extra.py"
    extra.write_text("extra\n")
    with pytest.raises(ValueError, match="inventory"):
        gate.verify_tree(tmp_path, expected)
    extra.unlink()
    extra.symlink_to(source)
    with pytest.raises(ValueError, match="symlink"):
        gate.verify_tree(tmp_path, expected)


@pytest.mark.parametrize("host", ("api.deepseek.com", "example.com", "0.0.0.0", "::", "127.1.2.3"))
def test_network_guard_denies_external_and_wildcard_hosts(host):
    sock = type("Socket", (), {"family": socket.AF_INET})()
    for event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"):
        with pytest.raises(PermissionError):
            gate.audit_network(event, (sock, (host, 443)))
    for event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
        with pytest.raises(PermissionError):
            gate.audit_network(event, (host, 443))
    with pytest.raises(PermissionError):
        gate.audit_network("socket.getnameinfo", ((host, 443), 0))


@pytest.mark.parametrize("host", ("127.0.0.1", "localhost", "::1"))
def test_network_guard_allows_only_named_loopback(host):
    sock = type("Socket", (), {"family": socket.AF_INET})()
    for event in ("socket.connect", "socket.bind", "socket.sendto", "socket.sendmsg"):
        gate.audit_network(event, (sock, (host, 443)))
    for event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
        gate.audit_network(event, (host, 443))
    gate.audit_network("socket.getnameinfo", ((host, 443), 0))


def test_collection_never_silently_drops_unknown_tests():
    collection = gate.Collection(["tests/a.py::one"], ["tests/a.py::one"])
    items = [type("Item", (), {"nodeid": "tests/a.py::unexpected"})()]
    with pytest.raises(RuntimeError, match="collection differs"):
        collection.pytest_collection_modifyitems(None, None, items)


def test_collection_deselects_exact_shard():
    seen = []
    config = type("Config", (), {"hook": type("Hook", (), {
        "pytest_deselected": staticmethod(lambda items: seen.extend(items)),
    })()})()
    items = [type("Item", (), {"nodeid": f"tests/a.py::{name}"})()
             for name in ("one", "two")]
    collection = gate.Collection([item.nodeid for item in items], [items[1].nodeid])
    collection.pytest_collection_modifyitems(None, config, items)
    assert [item.nodeid for item in items] == ["tests/a.py::two"]
    assert [item.nodeid for item in seen] == ["tests/a.py::one"]


def test_late_collection_swap_is_rejected():
    collection = gate.Collection(["tests/a.py::one", "tests/a.py::two"], ["tests/a.py::one"])
    session = type("Session", (), {"items": [type("Item", (), {"nodeid": "tests/a.py::two"})()]})()
    with pytest.raises(RuntimeError, match="final selected"):
        collection.pytest_collection_finish(session)


@pytest.mark.parametrize("skip", (False, True))
@pytest.mark.parametrize("declared_skip", (False, True))
def test_real_gate_reconciles_receipt_and_explicit_skips(tmp_path, skip, declared_skip):
    tree = tmp_path / "candidate"
    (tree / "tests").mkdir(parents=True)
    files = {"source.py": "# synthetic\n", "README.md": "Synthetic only\n",
             "tests/test_synthetic.py": (
                 "import os\nimport sys\nfrom pathlib import Path\nimport pytest\n"
                 "def test_environment():\n"
                 "    assert 'SYNTHETIC_AMBIENT_KEY' not in os.environ\n"
                 "    assert os.environ['PATH'] == str(Path(sys.executable).parent) + ':/usr/bin:/bin:/usr/sbin:/sbin'\n"
                 + ("    pytest.skip('predeclared synthetic fixture')\n" if skip else "    assert True\n")
             )}
    hashes = {}
    for name, text in files.items():
        (tree / name).write_text(text)
        hashes[name] = gate.digest(tree / name)
    nodeid = "tests/test_synthetic.py::test_environment"
    manifest = {"source_sha256": {"source.py": hashes["source.py"]},
                "test_sha256": {"tests/test_synthetic.py": hashes["tests/test_synthetic.py"]},
                "auxiliary_sha256": {"README.md": hashes["README.md"]},
                "expected_nodeids": [nodeid],
                "expected_skip_nodeids": [nodeid] if declared_skip else []}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    receipt = tmp_path / "receipt.json"
    import os
    env = dict(os.environ, SYNTHETIC_AMBIENT_KEY="not-a-real-key")
    completed = subprocess.run([
        sys.executable, "-I", "-B", str(Path(gate.__file__).resolve()),
        "--tree", str(tree), "--manifest", str(manifest_path),
        "--manifest-sha256", gate.digest(manifest_path),
        "--receipt", str(receipt), "--junit", str(tmp_path / "result.xml"),
    ], env=env, capture_output=True, text=True, timeout=30)
    data = json.loads(receipt.read_text())
    assert data["candidate_verified_after"] is True
    assert data["junit_selection_reconciled"] is True
    assert data["gate_passed"] == (skip == declared_skip)
    assert (completed.returncode == 0) == (skip == declared_skip)
