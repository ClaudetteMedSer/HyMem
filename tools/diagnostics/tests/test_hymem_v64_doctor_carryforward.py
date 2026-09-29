"""The diagnostic overlay is exactly one reviewed file, never a gate exception."""
from pathlib import Path
import sys
import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import hymem_v64_doctor_carryforward as fix

@pytest.fixture
def files(monkeypatch):
    values = {'hymem/file%d.py' % i: 'a' * 64 for i in range(480)}
    values[fix.DOCTOR] = fix.OLD
    # Synthetic manifest with its independently sealed identity.
    monkeypatch.setattr(fix.r, 'CANDIDATE_PIN', fix.r.digest(values))
    return values

def test_exact_one_file_carryforward(files):
    actual = fix.production_files(files)
    assert {n for n in files if files[n] != actual[n]} == {fix.DOCTOR}
    assert actual[fix.DOCTOR] == fix.NEW and files[fix.DOCTOR] == fix.OLD
    info = fix.identity(files)
    assert info['production_manifest_sha256'] == fix.r.digest(actual)
    assert info['base_manifest_sha256'] == fix.r.digest(files)
    assert info['unchanged_base_files'] == 480

@pytest.mark.parametrize('path', ['hymem/contrib/openai_client.py', 'hymem/core/migrations/064.sql',
                                  'hymem/contrib/model_policy.py', 'hymem/file1.py'])
def test_any_other_manifest_change_rejected(files, path):
    changed = dict(files)
    changed[path] = 'b' * 64
    with pytest.raises(RuntimeError, match='base_manifest_drift'):
        fix.production_files(changed)

def test_stale_hotfix_rejected_before_write(files, tmp_path, monkeypatch):
    root = tmp_path / 'live'; backup = tmp_path / 'backup'; stage = tmp_path / 'stage'
    for directory in (root, backup, stage): directory.mkdir()
    (backup / 'hymem').mkdir()
    (backup / fix.DOCTOR).write_text('stale accepted source')
    monkeypatch.setattr(fix.r, 'verify_files', lambda *_: None)
    with pytest.raises(RuntimeError, match='accepted_doctor_backup_drift'):
        fix.install(root, backup, stage, files)
    assert list(stage.iterdir()) == []

def test_adapter_preserves_all_original_checks(files):
    census = {'role_profile_sha256': {'honcho': ['h'], 'mcp': ['m']},
              'processes': {'distribution_count': 86, 'distribution_sha256': 'd'},
              'preserved_file_sha256': {
                  str(fix.r.HOME / '.hermes/bin/hymem-server-wrapper'): 'w',
                  str(fix.r.HOME / '.agent37/hooks/post-restart.sh'): 'k'}}
    original = fix.post.container_script(fix.production_files(files), census)
    adapted = fix.container_script(files, census)
    addition = "\n    report.update(" + repr(fix.identity(files)) + ")"
    assert adapted.replace(addition, '', 1) == original
    assert adapted.count(addition) == 1
