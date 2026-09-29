"""Run an exact isolated LME candidate's tests, without provider credentials.

This is a diagnostic runner, not an LME result or deployment tool. It verifies
every manifest file before and after pytest. Its network audit covers this
process; subprocess tests receive a credential-free environment but must supply
their own network mocks. Use OS network isolation for an entire process tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_files(manifest: dict) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("source_sha256", "test_sha256", "auxiliary_sha256"):
        group = manifest.get(name)
        if not isinstance(group, dict) or not group:
            raise ValueError(f"missing identity map: {name}")
        for relative, checksum in group.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or str(path) != relative:
                raise ValueError("unsafe manifest path")
            if (not isinstance(checksum, str) or len(checksum) != 64
                    or any(c not in "0123456789abcdef" for c in checksum)):
                raise ValueError("invalid SHA-256")
            if relative in result:
                raise ValueError("overlapping identity maps")
            result[relative] = checksum
    return result


def verify_tree(tree: Path, expected: dict[str, str]) -> None:
    actual = set()
    for path in tree.rglob("*"):
        if path.is_symlink():
            raise ValueError("symlink in isolated candidate")
        if path.is_file():
            actual.add(path.relative_to(tree).as_posix())
    if actual != set(expected):
        raise ValueError("candidate file inventory differs from manifest")
    for relative, checksum in expected.items():
        if digest(tree / relative) != checksum:
            raise ValueError(f"candidate hash differs: {relative}")


def loopback_address(address: object) -> bool:
    return (isinstance(address, tuple) and len(address) >= 2
            and address[0] in ("127.0.0.1", "::1", "localhost"))


def audit_network(event: str, args: tuple) -> None:
    if event in ("socket.connect", "socket.bind"):
        sock, address = args
        # AF_UNIX does not reach the network, and event loops use local pairs.
        if sock.family != socket.AF_UNIX and not loopback_address(address):
            raise PermissionError("offline gate blocked non-loopback socket")
    elif event in ("socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
        if args[0] not in ("127.0.0.1", "::1", "localhost"):
            raise PermissionError("offline gate blocked non-loopback DNS")
    elif event == "socket.getnameinfo":
        if not loopback_address(args[0]):
            raise PermissionError("offline gate blocked non-loopback reverse DNS")
    elif event in ("socket.sendto", "socket.sendmsg"):
        sock, address = args
        if sock.family != socket.AF_UNIX and not loopback_address(address):
            raise PermissionError("offline gate blocked non-loopback datagram")


class Collection:
    def __init__(self, expected: list[str], selected: list[str]):
        self.expected = expected
        self.selected = selected
        self.collected: list[str] = []
        self.finished: set[str] = set()
        self.skipped: set[str] = set()

    def pytest_collection_modifyitems(self, session, config, items):
        self.collected = [item.nodeid for item in items]
        if len(self.collected) != len(set(self.collected)):
            raise RuntimeError("duplicate collected node IDs")
        if set(self.collected) != set(self.expected):
            raise RuntimeError("test collection differs from frozen manifest")
        if len(self.selected) != len(set(self.selected)):
            raise RuntimeError("duplicate selected node IDs")
        chosen = set(self.selected)
        if not chosen <= set(self.collected):
            raise RuntimeError("unknown selected test")
        deselected = [item for item in items if item.nodeid not in chosen]
        items[:] = [item for item in items if item.nodeid in chosen]
        config.hook.pytest_deselected(items=deselected)

    def pytest_collection_finish(self, session):
        final = [item.nodeid for item in session.items]
        if len(final) != len(self.selected) or set(final) != set(self.selected):
            raise RuntimeError("final selected test inventory changed")

    def pytest_runtest_logreport(self, report):
        if report.when == "teardown":
            self.finished.add(report.nodeid)
        if report.skipped:
            self.skipped.add(report.nodeid)


def runtime_search_path() -> str:
    """Use this interpreter's runtime for child commands, never ambient PATH."""
    return f"{Path(sys.executable).parent}:/usr/bin:/bin:/usr/sbin:/sbin"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--junit", required=True, type=Path)
    parser.add_argument("--shard", type=int, default=1)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--focus-file", action="append", default=[],
                        help="Exact manifest test file; repeat for a focused gate")
    args = parser.parse_args()
    tree, manifest_path = args.tree.resolve(), args.manifest.resolve()
    receipt, junit = args.receipt.resolve(), args.junit.resolve()
    if receipt.exists() or junit.exists():
        raise ValueError("refusing to overwrite an existing test receipt")
    if receipt == junit:
        raise ValueError("receipt and JUnit must be separate files")
    if receipt.is_relative_to(tree) or junit.is_relative_to(tree):
        raise ValueError("test receipts must be outside the candidate")
    if args.shards < 1 or not 1 <= args.shard <= args.shards:
        raise ValueError("invalid shard")
    if digest(manifest_path) != args.manifest_sha256:
        raise ValueError("manifest hash differs")
    manifest = json.loads(manifest_path.read_text())
    expected = expected_files(manifest)
    nodeids = manifest["expected_nodeids"]
    if not isinstance(nodeids, list) or not nodeids or not all(
        isinstance(node, str) and node.startswith("tests/") for node in nodeids
    ):
        raise ValueError("missing frozen test collection")
    if len(nodeids) != len(set(nodeids)):
        raise ValueError("duplicate frozen test node IDs")
    allowed_skips = manifest.get("expected_skip_nodeids", [])
    if (not isinstance(allowed_skips, list)
            or not all(isinstance(node, str) for node in allowed_skips)
            or len(allowed_skips) != len(set(allowed_skips))
            or not set(allowed_skips) <= set(nodeids)):
        raise ValueError("invalid explicit skip inventory")
    verify_tree(tree, expected)
    focus = args.focus_file
    if len(focus) != len(set(focus)) or any(
        path not in manifest["test_sha256"] for path in focus
    ):
        raise ValueError("invalid focused test-file inventory")
    eligible = [node for node in nodeids if not focus or node.split("::", 1)[0] in focus]
    if focus and set(focus) != {node.split("::", 1)[0] for node in eligible}:
        raise ValueError("focused file has no collected tests")
    selected = eligible[args.shard - 1::args.shards]
    if not selected:
        raise ValueError("empty selection")
    plugin = Collection(nodeids, selected)
    sandbox = tempfile.TemporaryDirectory(prefix="hymem-offline-gate-")
    # No ambient API keys, endpoint settings, HOME startup files or plugins.
    os.environ.clear()
    os.environ.update({
        "PATH": runtime_search_path(),
        "HOME": sandbox.name,
        "TMPDIR": sandbox.name,
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTHONHASHSEED": "0",
    })
    sys.dont_write_bytecode = True
    os.chdir(tree)
    sys.path.insert(0, str(tree))
    sys.addaudithook(audit_network)
    import pytest

    started = time.monotonic()
    result = {
        "manifest_sha256": args.manifest_sha256,
        "shard": args.shard, "shards": args.shards,
        "focus_files": focus,
        "selected": len(selected), "candidate_verified_before": True,
        "provider_credentials_present": False,
        "network_guard_scope": "pytest process; literal loopback only",
        "deployment_performed": False, "benchmark_completion_claimed": False,
    }
    try:
        exit_code = int(pytest.main([
            "tests", "-q", "-p", "no:cacheprovider",
            "--junitxml", str(junit),
            "--basetemp", str(Path(sandbox.name) / "pytest"),
        ], plugins=[plugin]))
        result["pytest_exit_code"] = exit_code
    except BaseException as exc:
        result["runner_error_type"] = type(exc).__name__
        raise
    finally:
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        try:
            verify_tree(tree, expected)
            result["candidate_verified_after"] = True
        except ValueError:
            result["candidate_verified_after"] = False
        result["junit_selection_reconciled"] = False
        if junit.is_file():
            result["junit_sha256"] = digest(junit)
            suites = ET.parse(junit).getroot().findall("testsuite")
            result["junit_totals"] = {
                key: sum(int(suite.get(key, "0")) for suite in suites)
                for key in ("tests", "failures", "errors", "skipped")
            }
            case_names = [
                (case.get("classname"), case.get("name"))
                for suite in suites for case in suite.findall("testcase")
            ]
            result["junit_selection_reconciled"] = (
                result["junit_totals"]["tests"] == len(selected)
                and len(case_names) == len(selected)
                and len(set(case_names)) == len(selected)
                and plugin.finished == set(selected)
            )
        result["observed_skip_nodeids"] = sorted(plugin.skipped)
        result["allowed_selected_skip_nodeids"] = sorted(set(allowed_skips) & set(selected))
        result["gate_passed"] = (
            result.get("pytest_exit_code") == 0
            and result["candidate_verified_after"]
            and result["junit_selection_reconciled"]
            and all(result.get("junit_totals", {}).get(key) == 0
                    for key in ("failures", "errors"))
            and plugin.skipped == (set(allowed_skips) & set(selected))
            and result.get("junit_totals", {}).get("skipped") == len(plugin.skipped)
        )
        with receipt.open("x") as output:
            json.dump(result, output, indent=2, sort_keys=True)
            output.write("\n")
        sandbox.cleanup()
    return 0 if result["gate_passed"] else (exit_code or 2)


if __name__ == "__main__":
    raise SystemExit(main())
