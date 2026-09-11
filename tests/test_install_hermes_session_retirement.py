"""Offline tests for the fixed-path pre-start installer; no Hermes imports."""

import copy
import difflib
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import socket
import stat
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/deployment/install_hermes_session_retirement.py"


@pytest.fixture
def installer():
    spec = importlib.util.spec_from_file_location("session_retirement_installer", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    attempts = []

    def blocked(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("network forbidden")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    yield
    assert not attempts


@pytest.fixture
def tree(tmp_path, installer):
    runtime = tmp_path / "runtime"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    before, after, manifest = {}, {}, {
        "format": "hermes-session-key-retirement-patch-v1",
        "patch": installer.PATCH_NAME,
        "files": [],
    }
    patch = ""
    for index, path in enumerate(installer.PATHS):
        before[path] = f"value = 1\n# {Path(path).name}\n".encode()
        after[path] = f"value = 2\n# {Path(path).name}\n".encode()
        target = runtime / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(before[path])
        target.chmod((0o600, 0o640, 0o755)[index])
        patch += "".join(difflib.unified_diff(
            before[path].decode().splitlines(keepends=True),
            after[path].decode().splitlines(keepends=True),
            fromfile="a/" + path, tofile="b/" + path,
        ))
        manifest["files"].append({
            "path": path,
            "before_sha256": installer._sha(before[path]),
            "after_sha256": installer._sha(after[path]),
        })
    manifest["patch_sha256"] = installer._sha(patch.encode())
    (artifacts / installer.PATCH_NAME).write_text(patch)
    (artifacts / installer.MANIFEST_NAME).write_text(json.dumps(manifest))
    unrelated = runtime / "plugins/memory/honcho/local-customization.py"
    unrelated.write_bytes(b"leave this unchanged\n")
    return dict(runtime=runtime, artifacts=artifacts, before=before, after=after,
                manifest=manifest, patch=patch, unrelated=unrelated)


def snapshot(runtime):
    return {
        str(path.relative_to(runtime)): (path.read_bytes(), path.stat().st_ino,
            stat.S_IMODE(path.stat().st_mode), path.stat().st_uid, path.stat().st_gid)
        for path in runtime.rglob("*") if path.is_file()
    }


def assert_before(tree, installer):
    for path in installer.PATHS:
        assert (tree["runtime"] / path).read_bytes() == tree["before"][path]


def write_manifest(tree, installer, manifest):
    (tree["artifacts"] / installer.MANIFEST_NAME).write_text(json.dumps(manifest))


def repin_patch(tree, installer, patch):
    manifest = copy.deepcopy(tree["manifest"])
    manifest["patch_sha256"] = installer._sha(patch.encode())
    write_manifest(tree, installer, manifest)
    (tree["artifacts"] / installer.PATCH_NAME).write_text(patch)


def test_success_preserves_owner_mode_unrelated_files_and_second_run_is_noop(tree, installer):
    original = snapshot(tree["runtime"])
    assert installer.install(tree["runtime"], artifact_dir=tree["artifacts"]) == "installed"
    installed = snapshot(tree["runtime"])
    for path in installer.PATHS:
        assert installed[path][0] == tree["after"][path]
        assert installed[path][2:] == original[path][2:]
    extra = str(tree["unrelated"].relative_to(tree["runtime"]))
    assert installed[extra] == original[extra]
    assert set(installed) == set(original)
    assert not list(tree["runtime"].rglob(".hymem-session-retirement-*"))
    assert installer.install(tree["runtime"], artifact_dir=tree["artifacts"]) == "already-installed"
    assert snapshot(tree["runtime"]) == installed


def test_check_is_read_only_and_validates_result_hash(tree, installer):
    original = snapshot(tree["runtime"])
    assert installer.install(tree["runtime"], artifact_dir=tree["artifacts"], check=True) == "ready"
    assert snapshot(tree["runtime"]) == original
    manifest = tree["manifest"]
    manifest["files"][0]["after_sha256"] = "0" * 64
    write_manifest(tree, installer, manifest)
    with pytest.raises(installer.InstallError, match="patched_source_digest_mismatch"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"], check=True)
    assert snapshot(tree["runtime"]) == original


@pytest.mark.parametrize("state", ["mixed", "unknown"])
def test_nonuniform_state_refuses_without_touching_anything(tree, installer, state):
    first = installer.PATHS[0]
    (tree["runtime"] / first).write_bytes(tree["after"][first] if state == "mixed" else b"custom code\n")
    original = snapshot(tree["runtime"])
    with pytest.raises(installer.InstallError, match="mixed_or_unknown_source_state"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert snapshot(tree["runtime"]) == original


@pytest.mark.parametrize("mutation", [
    "additional", "missing", "duplicate", "traversal", "absolute", "wrong_basename",
    "extra_entry_field", "invalid_digest", "same_states", "bad_patch_name", "extra_root_field",
    "auxiliary_traversal", "test_contract_extra_file", "bad_format", "non_object", "non_string_path",
])
def test_invalid_manifest_refuses_without_writes(tree, installer, mutation):
    manifest = copy.deepcopy(tree["manifest"])
    first = manifest["files"][0]
    if mutation == "additional":
        manifest["files"].append(dict(first, path="plugins/memory/other.py"))
    elif mutation == "missing":
        manifest["files"].pop()
    elif mutation == "duplicate":
        manifest["files"][1] = dict(first)
    elif mutation == "traversal":
        first["path"] = "../outside.py"
    elif mutation == "absolute":
        first["path"] = "/etc/passwd"
    elif mutation == "wrong_basename":
        first["path"] = "other/client.py"
    elif mutation == "extra_entry_field":
        first["execute"] = "never"
    elif mutation == "invalid_digest":
        first["before_sha256"] = "bad"
    elif mutation == "same_states":
        first["after_sha256"] = first["before_sha256"]
    elif mutation == "bad_patch_name":
        manifest["patch"] = "../outside.patch"
    elif mutation == "extra_root_field":
        manifest["extra"] = True
    elif mutation == "auxiliary_traversal":
        manifest["tests"] = "../outside.py"
    elif mutation == "test_contract_extra_file":
        manifest["test_contract_updates"] = [dict(first)]
    elif mutation == "bad_format":
        manifest["format"] = "other"
    elif mutation == "non_object":
        manifest = []
    elif mutation == "non_string_path":
        first["path"] = []
    write_manifest(tree, installer, manifest)
    original = snapshot(tree["runtime"])
    with pytest.raises(installer.InstallError):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert snapshot(tree["runtime"]) == original


def test_duplicate_json_key_rejected(tree, installer):
    path = tree["artifacts"] / installer.MANIFEST_NAME
    path.write_text('{"format":"one","format":"two"}')
    with pytest.raises(installer.InstallError, match="duplicate_manifest_key"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert_before(tree, installer)


@pytest.mark.parametrize("mutation", ["digest", "extra_file", "wrong_target", "fuzzy_context", "offset", "truncated", "duplicate_file"])
def test_patch_corruption_cannot_be_applied(tree, installer, mutation):
    patch = tree["patch"]
    if mutation == "digest":
        (tree["artifacts"] / installer.PATCH_NAME).write_text(patch + "\n")
    else:
        if mutation == "extra_file":
            patch += "--- a/extra.py\n+++ b/extra.py\n@@ -1 +1 @@\n-old\n+new\n"
        elif mutation == "wrong_target":
            patch = patch.replace("+++ b/plugins/memory/honcho/client.py", "+++ b/elsewhere.py")
        elif mutation == "fuzzy_context":
            patch = patch.replace(" # client.py", " # wrong-client.py")
        elif mutation == "offset":
            patch = patch.replace("@@ -1,2 +1,2 @@", "@@ -2,2 +2,2 @@", 1)
        elif mutation == "truncated":
            patch = patch[:-4]
        elif mutation == "duplicate_file":
            patch += patch
        repin_patch(tree, installer, patch)
    original = snapshot(tree["runtime"])
    with pytest.raises(installer.InstallError):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert snapshot(tree["runtime"]) == original


@pytest.mark.parametrize("kind", ["source", "source_parent", "runtime", "manifest", "patch", "artifact_parent"])
def test_symlinks_are_refused(tree, installer, tmp_path, kind):
    if kind == "source":
        target = tree["runtime"] / installer.PATHS[0]
    elif kind == "source_parent":
        target = tree["runtime"] / "plugins/memory/honcho"
    elif kind == "runtime":
        target = tree["runtime"]
    elif kind == "manifest":
        target = tree["artifacts"] / installer.MANIFEST_NAME
    elif kind == "patch":
        target = tree["artifacts"] / installer.PATCH_NAME
    else:
        target = tree["artifacts"]
    moved = tmp_path / "symlink-destination"
    target.rename(moved)
    target.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises((OSError, installer.InstallError)):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])


@pytest.mark.parametrize("kind", ["manifest", "patch", "source"])
def test_input_sizes_are_bounded(tree, installer, monkeypatch, kind):
    monkeypatch.setattr(installer, {"manifest": "MAX_MANIFEST", "patch": "MAX_PATCH", "source": "MAX_SOURCE"}[kind], 1)
    original = snapshot(tree["runtime"])
    with pytest.raises(installer.InstallError, match="oversized"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert snapshot(tree["runtime"]) == original


def test_hardlinked_source_is_refused(tree, installer, tmp_path):
    os.link(tree["runtime"] / installer.PATHS[0], tmp_path / "hardlink")
    with pytest.raises(installer.InstallError, match="unsafe_or_oversized_file"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert_before(tree, installer)


@pytest.mark.parametrize("failed_index", [0, 1, 2])
def test_caught_replace_failure_rolls_back_prior_files(tree, installer, monkeypatch, failed_index):
    original = snapshot(tree["runtime"])
    real_replace = os.replace
    attempted = []

    def fail_one(source, destination, **kwargs):
        if source.startswith("after-"):
            attempted.append(source)
            if len(attempted) - 1 == failed_index:
                raise OSError("synthetic failure")
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(installer.os, "replace", fail_one)
    with pytest.raises(installer.InstallError, match="install_failed_rolled_back"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    restored = snapshot(tree["runtime"])
    assert set(restored) == set(original)
    for path in installer.PATHS:
        assert restored[path][0] == original[path][0]
        assert restored[path][2:] == original[path][2:]


def test_caught_interrupt_after_rename_also_rolls_back(tree, installer, monkeypatch):
    real_replace = os.replace
    interrupted = []

    def interrupt_after(source, destination, **kwargs):
        real_replace(source, destination, **kwargs)
        if source.startswith("after-") and not interrupted:
            interrupted.append(True)
            raise KeyboardInterrupt()

    monkeypatch.setattr(installer.os, "replace", interrupt_after)
    with pytest.raises(installer.InstallError, match="install_failed_rolled_back"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert_before(tree, installer)


def test_failed_rollback_preserves_recovery_stage_and_next_run_refuses_mixed(tree, installer, monkeypatch):
    real_replace = os.replace

    def fail_install_and_restore(source, destination, **kwargs):
        if source == "after-session.py" or source.startswith("before-"):
            raise OSError("synthetic failure")
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(installer.os, "replace", fail_install_and_restore)
    with pytest.raises(installer.InstallError, match="rollback_incomplete_preserved_staging"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    stages = list(tree["runtime"].rglob(".hymem-session-retirement-*"))
    assert len(stages) == 1
    assert (stages[0] / "before-client.py").read_bytes() == tree["before"][installer.PATHS[0]]
    assert stat.S_IMODE(stages[0].stat().st_mode) == 0o700
    monkeypatch.setattr(installer.os, "replace", real_replace)
    with pytest.raises(installer.InstallError, match="mixed_or_unknown_source_state"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert stages[0].exists()


def test_concurrent_source_change_before_install_is_not_overwritten(tree, installer, monkeypatch):
    real_write = installer._write_stage
    changed = []
    target = tree["runtime"] / installer.PATHS[1]

    def mutate_after_stage(*args):
        real_write(*args)
        if not changed:
            target.write_bytes(b"concurrent customization\n")
            changed.append(True)

    monkeypatch.setattr(installer, "_write_stage", mutate_after_stage)
    with pytest.raises(installer.InstallError, match="source_changed_before_install"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert target.read_bytes() == b"concurrent customization\n"
    assert (tree["runtime"] / installer.PATHS[0]).read_bytes() == tree["before"][installer.PATHS[0]]


def test_existing_stage_name_collision_is_not_deleted(tree, installer, monkeypatch):
    monkeypatch.setattr(installer.secrets, "token_hex", lambda _: "fixed")
    stage = tree["runtime"] / "plugins/memory/honcho/.hymem-session-retirement-fixed"
    stage.mkdir()
    with pytest.raises(FileExistsError):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert stage.is_dir()
    assert_before(tree, installer)


def test_staging_failure_cleans_only_staging_and_leaves_runtime_unchanged(tree, installer, monkeypatch):
    original = snapshot(tree["runtime"])

    def denied(*args):
        raise PermissionError("synthetic metadata failure")

    monkeypatch.setattr(installer.os, "fchown", denied)
    with pytest.raises(PermissionError):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    assert snapshot(tree["runtime"]) == original
    assert not list(tree["runtime"].rglob(".hymem-session-retirement-*"))


def test_parallel_installer_lock_refuses_without_staging(tree, installer):
    directory = installer._open_directory(tree["runtime"] / "plugins/memory/honcho")
    try:
        fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(installer.InstallError, match="another_installer_holds_directory_lock"):
            installer.install(tree["runtime"], artifact_dir=tree["artifacts"])
    finally:
        os.close(directory)
    assert_before(tree, installer)
    assert not list(tree["runtime"].rglob(".hymem-session-retirement-*"))


def test_fifo_source_is_rejected_without_waiting(tree, installer):
    source = tree["runtime"] / installer.PATHS[0]
    source.unlink()
    os.mkfifo(source)
    with pytest.raises(installer.InstallError, match="unsafe_or_oversized_file"):
        installer.install(tree["runtime"], artifact_dir=tree["artifacts"])


@pytest.mark.parametrize("root", [Path("relative"), Path("/"), Path("/some/../root")])
def test_unsafe_root_refused_before_artifact_access(installer, root):
    with pytest.raises(installer.InstallError, match="unsafe_runtime_root"):
        installer.install(root)


def test_cli_uses_adjacent_artifacts_outputs_only_status_and_noops(tree, installer):
    copied = tree["artifacts"] / SCRIPT.name
    copied.write_bytes(SCRIPT.read_bytes())
    command = [sys.executable, str(copied), "--runtime-root", str(tree["runtime"])]
    for extra, status in ((["--check"], "ready"), ([], "installed"), ([], "already-installed")):
        result = subprocess.run(command + extra, capture_output=True, text=True, timeout=10)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"status": status, "runtime_files": 3}
        assert result.stderr == ""


def test_cli_failure_is_nonzero_and_secret_free(tree):
    copied = tree["artifacts"] / SCRIPT.name
    copied.write_bytes(SCRIPT.read_bytes())
    result = subprocess.run(
        [sys.executable, str(copied), "--runtime-root", str(tree["runtime"] / "secret-canary")],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["status"] == "FAIL"
    assert "secret-canary" not in result.stderr


def test_cli_deeply_nested_manifest_refuses_without_traceback(tree, installer):
    copied = tree["artifacts"] / SCRIPT.name
    copied.write_bytes(SCRIPT.read_bytes())
    (tree["artifacts"] / installer.MANIFEST_NAME).write_text("[" * 2000 + "0" + "]" * 2000)
    result = subprocess.run(
        [sys.executable, str(copied), "--runtime-root", str(tree["runtime"])],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 1
    assert result.stdout == ""
    assert json.loads(result.stderr)["status"] == "FAIL"
    assert "Traceback" not in result.stderr


def test_actual_shipped_patch_and_manifest_parse_without_optional_test_artifacts(installer):
    pins, patches = installer._load_artifacts(SCRIPT.parent)
    assert set(pins) == set(installer.PATHS)
    assert set(patches) == set(installer.PATHS)
