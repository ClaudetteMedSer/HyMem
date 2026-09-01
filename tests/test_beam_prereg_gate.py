"""The spec-hash gate: an artifact must name the spec that authorised it.

"A verdict whose spec-hash post-dates its artifact is void by construction."
That only bites if the check runs BEFORE the spend. A spec cited by filename
gives nothing — the file is mutable and the citation is not — so the run
refuses unless the spec is committed and clean, and records the commit and
blob hashes it saw. A spec edited afterwards to fit the numbers then no longer
matches what its own artifact recorded.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("requests")

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))
import beam_adapter as ba  # noqa: E402


def _run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo with one committed spec — the gate shells out to git,
    so a mocked git would test the mock rather than the refusal."""
    _run("git", "init", "-q", cwd=tmp_path)
    _run("git", "config", "user.email", "t@example.com", cwd=tmp_path)
    _run("git", "config", "user.name", "t", cwd=tmp_path)
    (tmp_path / "docs").mkdir()
    spec = tmp_path / "docs" / "spec.md"
    spec.write_text("# pre-registration\n")
    _run("git", "add", "docs/spec.md", cwd=tmp_path)
    _run("git", "commit", "-qm", "spec", cwd=tmp_path)
    monkeypatch.setattr(ba, "_repo_root", tmp_path)
    return tmp_path


def test_committed_spec_is_pinned_by_commit_and_blob(repo):
    got = ba.resolve_prereg("docs/spec.md", allow_dirty=False)
    assert got["path"] == "docs/spec.md"
    assert len(got["commit"]) == 40 and len(got["blob"]) == 40
    assert got["code_dirty"] is False
    assert got["committed_at"]


def test_the_blob_hash_is_of_the_spec_not_the_commit(repo):
    """Two runs under the same spec share a blob even if the repo moved on;
    a spec EDIT changes the blob. That is the hash doing the work."""
    first = ba.resolve_prereg("docs/spec.md", allow_dirty=False)
    (repo / "other.txt").write_text("unrelated\n")
    _run("git", "add", "other.txt", cwd=repo)
    _run("git", "commit", "-qm", "unrelated", cwd=repo)
    second = ba.resolve_prereg("docs/spec.md", allow_dirty=False)
    assert second["blob"] == first["blob"]
    assert second["code_commit"] != first["code_commit"]


def test_an_edited_spec_is_refused_before_the_run_spends(repo):
    """The failure this whole mechanism exists to prevent: editing the spec
    to fit the numbers. Post-hoc it is invisible; at run start it is a refusal."""
    (repo / "docs" / "spec.md").write_text("# pre-registration\nedited after the fact\n")
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg("docs/spec.md", allow_dirty=False)
    assert e.value.code == 2


def test_an_uncommitted_spec_is_refused(repo):
    (repo / "docs" / "draft.md").write_text("# never committed\n")
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg("docs/draft.md", allow_dirty=False)
    assert e.value.code == 2


def test_a_missing_spec_is_refused(repo):
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg("docs/nope.md", allow_dirty=False)
    assert e.value.code == 2


def test_a_spec_outside_the_repo_is_refused(repo, tmp_path_factory):
    """It could not have been committed alongside the code it authorises, so
    its 'hash' would date nothing."""
    outside = tmp_path_factory.mktemp("elsewhere") / "spec.md"
    outside.write_text("# elsewhere\n")
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg(str(outside), allow_dirty=False)
    assert e.value.code == 2


def test_dirty_tracked_code_is_refused_by_default(repo):
    """A clean spec is not enough: if the CODE has no commit, nothing names
    what actually produced the numbers."""
    (repo / "docs" / "spec.md").write_text("# pre-registration\n")  # spec itself clean
    (repo / "tracked.py").write_text("x = 1\n")
    _run("git", "add", "tracked.py", cwd=repo)
    _run("git", "commit", "-qm", "code", cwd=repo)
    (repo / "tracked.py").write_text("x = 2\n")
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg("docs/spec.md", allow_dirty=False)
    assert e.value.code == 2


def test_allow_dirty_permits_the_run_but_records_the_fact(repo):
    """The escape hatch must leave a mark in the artifact, or it is just a
    way to produce an unattributable canonical quietly."""
    (repo / "tracked.py").write_text("x = 1\n")
    _run("git", "add", "tracked.py", cwd=repo)
    _run("git", "commit", "-qm", "code", cwd=repo)
    (repo / "tracked.py").write_text("x = 2\n")
    got = ba.resolve_prereg("docs/spec.md", allow_dirty=True)
    assert got["code_dirty"] is True


def test_untracked_files_alone_do_not_block_a_run(repo):
    """Scratch files and artifacts are normal; only MODIFIED TRACKED files
    mean the run's code is unnamed."""
    (repo / "scratch.csv").write_text("junk\n")
    assert ba.resolve_prereg("docs/spec.md", allow_dirty=False)["code_dirty"] is False


def test_no_prereg_returns_none_rather_than_a_plausible_looking_stub(repo):
    """Recorded as prereg: null, so an exploratory run stays identifiable as
    one instead of blending in with the canonicals."""
    assert ba.resolve_prereg(None, allow_dirty=False) is None
