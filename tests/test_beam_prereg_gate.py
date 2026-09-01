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


def _out(*args, cwd):
    return subprocess.run(args, cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


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
    got = ba.resolve_prereg("docs/spec.md")
    assert got["path"] == "docs/spec.md"
    assert len(got["commit"]) == 40 and len(got["blob"]) == 40
    assert got["committed_at"]


def test_the_blob_hash_is_of_the_spec_not_the_commit(repo):
    """Two runs under the same spec share a blob even if the repo moved on;
    a spec EDIT changes the blob. That is the hash doing the work."""
    first = ba.resolve_prereg("docs/spec.md")
    (repo / "other.txt").write_text("unrelated\n")
    _run("git", "add", "other.txt", cwd=repo)
    _run("git", "commit", "-qm", "unrelated", cwd=repo)
    second = ba.resolve_prereg("docs/spec.md")
    assert second["blob"] == first["blob"]
    assert second["code_commit"] != first["code_commit"]


def test_an_edited_spec_is_refused_before_the_run_spends(repo):
    """The failure this whole mechanism exists to prevent: editing the spec
    to fit the numbers. Post-hoc it is invisible; at run start it is a refusal."""
    (repo / "docs" / "spec.md").write_text("# pre-registration\nedited after the fact\n")
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg("docs/spec.md")
    assert e.value.code == 2


def test_an_uncommitted_spec_is_refused(repo):
    (repo / "docs" / "draft.md").write_text("# never committed\n")
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg("docs/draft.md")
    assert e.value.code == 2


def test_a_missing_spec_is_refused(repo):
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg("docs/nope.md")
    assert e.value.code == 2


def test_a_spec_outside_the_repo_is_refused(repo, tmp_path_factory):
    """It could not have been committed alongside the code it authorises, so
    its 'hash' would date nothing."""
    outside = tmp_path_factory.mktemp("elsewhere") / "spec.md"
    outside.write_text("# elsewhere\n")
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg(str(outside))
    assert e.value.code == 2


def test_dirty_tracked_code_is_refused_with_no_escape(repo):
    """A clean spec is not enough: if the CODE has no commit, nothing names
    what produced the numbers.

    There is deliberately no --allow-dirty. Such a flag would record
    code_commit = HEAD while the tree was HEAD plus an uncommitted diff that
    nothing captures -- not an incomplete field but a WRONG one, since checking
    that commit out and re-running gives a different result. The honest path
    for a dirty tree is --no-prereg, which records prereg: null.
    """
    (repo / "tracked.py").write_text("x = 1\n")
    _run("git", "add", "tracked.py", cwd=repo)
    _run("git", "commit", "-qm", "code", cwd=repo)
    (repo / "tracked.py").write_text("x = 2\n")
    with pytest.raises(SystemExit) as e:
        ba.resolve_prereg("docs/spec.md")
    assert e.value.code == 2
    assert not hasattr(ba.resolve_prereg, "allow_dirty")


def test_a_dirty_tree_can_still_run_exploratively(repo):
    """The refusal must not make the tool unusable during iteration -- it
    redirects to the truthful label rather than blocking work."""
    (repo / "tracked.py").write_text("x = 1\n")
    _run("git", "add", "tracked.py", cwd=repo)
    _run("git", "commit", "-qm", "code", cwd=repo)
    (repo / "tracked.py").write_text("x = 2\n")
    assert ba.resolve_prereg(None) is None


def test_no_field_claims_a_tree_that_never_ran(repo):
    """code_commit is only meaningful because dirty is refused; if it is
    present at all, it must be checkoutable and complete."""
    got = ba.resolve_prereg("docs/spec.md")
    assert got["code_commit"] == _out("git", "rev-parse", "HEAD", cwd=repo)
    assert "code_dirty" not in got


def test_untracked_files_alone_do_not_block_a_run(repo):
    """Scratch files and artifacts are normal; only MODIFIED TRACKED files
    mean the run's code is unnamed."""
    (repo / "scratch.csv").write_text("junk\n")
    assert ba.resolve_prereg("docs/spec.md")["blob"]


def test_no_prereg_returns_none_rather_than_a_plausible_looking_stub(repo):
    """Recorded as prereg: null, so an exploratory run stays identifiable as
    one instead of blending in with the canonicals."""
    assert ba.resolve_prereg(None) is None


# ── the input the gate cannot see ─────────────────────────────────────────
# git witnesses the spec and the code. It cannot witness the DATASET, which is
# fetched from the Hub by a name whose referent the host can move -- the same
# hazard as `deepseek-chat` on the model axis. The rejudge path is covered by
# its 160/160 reparse guard; a full run has no stored baseline to diff against,
# so the artifact has to carry the revision or nothing does.

class _Info:
    sha = "3205395e897e7318c7b094ef4e6047b9b82dbb03"


class _Api:
    asked: list = []

    def dataset_info(self, repo):
        _Api.asked.append(repo)
        return _Info()


@pytest.fixture
def hub(monkeypatch):
    import types
    _Api.asked = []
    mod = types.ModuleType("huggingface_hub")
    mod.HfApi = _Api
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    return _Api


def test_the_resolved_revision_is_recorded_per_repo(hub):
    got = ba.resolve_dataset_revisions(["100K"])
    assert got == {"Mohammadta/BEAM": _Info.sha}


def test_the_10m_scale_is_a_different_repo_and_is_witnessed_separately(hub):
    got = ba.resolve_dataset_revisions(["100K", "10M"])
    assert set(got) == {"Mohammadta/BEAM", "Mohammadta/BEAM-10M"}


def test_scales_sharing_a_repo_are_queried_once(hub):
    ba.resolve_dataset_revisions(["100K", "500K", "1M"])
    assert _Api.asked == ["Mohammadta/BEAM"]


def test_an_explicit_pin_is_recorded_without_asking_the_hub(hub):
    got = ba.resolve_dataset_revisions(["100K"], pin="deadbeef")
    assert got == {"Mohammadta/BEAM": "deadbeef"}
    assert _Api.asked == []


def test_an_unresolvable_revision_is_recorded_as_null_not_omitted(monkeypatch):
    """"We do not know" is a fact about the run. Omitting the key would make an
    unwitnessed artifact look like one predating the field."""
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    got = ba.resolve_dataset_revisions(["100K"])
    assert got == {"Mohammadta/BEAM": None}


def test_an_offline_hub_does_not_abort_the_run(monkeypatch):
    """The revision is evidence, not a gate: losing it must not cost a run
    that is otherwise fine, it must cost the claim that the run is witnessed."""
    import types
    mod = types.ModuleType("huggingface_hub")

    class _Broken:
        def dataset_info(self, repo):
            raise OSError("no route to host")

    mod.HfApi = _Broken
    monkeypatch.setitem(sys.modules, "huggingface_hub", mod)
    assert ba.resolve_dataset_revisions(["100K"]) == {"Mohammadta/BEAM": None}
