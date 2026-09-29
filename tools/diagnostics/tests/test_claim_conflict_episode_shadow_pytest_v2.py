"""V2 controls retain the reviewed adapter controls and cover mounted imports."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.diagnostics import claim_conflict_episode_shadow_pytest as prior_suite
from tools.diagnostics import claim_conflict_episode_shadow_pytest_v2 as suite


# Load the prior controls in an independent namespace so collecting both test
# files cannot change the sealed v1 controls' module globals.
spec = importlib.util.spec_from_file_location(
    'episode_shadow_v2_inherited_controls',
    Path(__file__).with_name('test_claim_conflict_episode_shadow_pytest.py'))
controls = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controls)
controls.suite = suite
for name, value in vars(controls).items():
    if name.startswith('test_') and name != 'test_configuration_is_source_only_network_none_and_versioned':
        globals()[name] = value


def test_configuration_is_source_only_network_none_and_versioned():
    base = suite.configure_base('local')
    helper = SimpleNamespace(RUNTIME=Path('/runtime'), IMAGE='sha256:' + 'a' * 64)
    command, mounts = base.configure(helper)
    assert command[command.index('--name') + 1] == 'hymem-episode-shadow-pytest-v2'
    assert command[command.index('--network') + 1] == 'none'
    assert '--read-only' in command and '--cap-drop' in command
    assert [(dest, writable) for _, dest, writable in mounts if writable] == [('/work', True)]
    assert set(dest for _, dest, _ in mounts) == {'/candidate', '/overlay', '/candidate-manifest.json', '/test-overlay.json', '/diag/runner.py', '/work', '/home/node/hymem-env', '/diag/base_suite.py'}
    assert str(suite.CANDIDATE) in [src for src, _, _ in mounts]
    assert base.ROOT == suite.ROOT and base.ROOT.name == 'episode-shadow-pytest-v2'
    assert base.FULL_SECONDS == 10800 and base.HOST_SECONDS == 11200


def test_prior_adapter_reproduces_shallow_container_import_failure():
    namespace = {'__file__': '/diag/runner.py', '__name__': 'mounted_runner_control'}
    with pytest.raises(IndexError):
        exec(compile(Path(prior_suite.__file__).read_text(), '/diag/runner.py', 'exec'), namespace)


def test_v2_imports_at_actual_container_mount_without_context_fallback():
    namespace = {'__file__': '/diag/runner.py', '__name__': 'mounted_runner_control'}
    exec(compile(Path(suite.__file__).read_text(), '/diag/runner.py', 'exec'), namespace)
    assert namespace['LOCAL_MANIFEST'] == suite.LOCAL_MANIFEST
    assert namespace['BASE_CONTAINER_SUITE'] == Path('/diag/base_suite.py')
    assert namespace['BASE_SUITE'] == suite.BASE_SUITE
    assert namespace['CANDIDATE_SHA'] == suite.CANDIDATE_SHA
    assert namespace['ROOT'] == suite.ROOT
    assert namespace['SELF'] == suite.SELF

