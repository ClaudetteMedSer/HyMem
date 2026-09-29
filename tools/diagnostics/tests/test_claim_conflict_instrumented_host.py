"""Parent tests of the private-clone host's isolation and metadata boundary."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('instrumented_host', Path(__file__).parents[1] / 'claim_conflict_instrumented_host.py')
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


def project(value):
    return host.project({'completion_calls': 0, 'http_attempts': 0, **value})


def test_private_mounts_and_network():
    h = SimpleNamespace(RUNTIME=Path('/runtime'), RUNTIME_ENV=Path('/private/runtime-env.json'), IMAGE='sha256:pin')
    for mode in ('offline', 'live'):
        command, mounts = host.configure(h, mode)
        assert command[command.index('--network') + 1] == ('hermes-net' if mode == 'live' else 'none')
        assert [m for m in mounts if m[2]] == [(str(host.WORK), '/work', True)]
        assert not any(m[0].endswith('/.hermes') or m[0].endswith('/hymem.sqlite') for m in mounts)
        assert (str(host.REFERENCE), '/reference/source.sqlite', False) in mounts
        assert '--read-only' in command


def test_metadata_omits_unapproved_private_keys():
    assert project({'status': 'captured_failure', 'error_type': 'ValueError',
                         'completion_calls': 4, 'http_attempts': 5,
                         'exception': 'PRIVATE', 'request': 'PRIVATE', 'traceback': 'PRIVATE'}) == {
                             'status': 'captured_failure', 'error_type': 'ValueError',
                             'completion_calls': 4, 'http_attempts': 5, 'candidate_frames': []}


@pytest.mark.parametrize('field,value', [('completion_calls', 65), ('http_attempts', 193),
                                       ('completion_calls', -1), ('http_attempts', True)])
def test_invalid_accounting_rejected(field, value):
    with pytest.raises(RuntimeError):
        project({'status': 'completed', field: value})


def test_frames_are_static_code_only():
    good = {'path': 'hymem/dreaming/evidence.py', 'function': 'record_claim_observation', 'line': 22}
    assert project({'status': 'captured_failure', 'candidate_frames': [good]})['candidate_frames'] == [good]
    for path in ('/work/private.sqlite', 'hymem/../private.py', 'hymem/source text.py'):
        with pytest.raises(RuntimeError):
            project({'status': 'error', 'candidate_frames': [{**good, 'path': path}]})


def test_private_exception_type_is_not_echoed():
    assert project({'status': 'error', 'error_type': 'private_source'})['error_type'] == 'Exception'


def test_missing_accounting_is_not_zero_usage():
    with pytest.raises(RuntimeError, match='accounting_missing'):
        host.project({'status': 'completed'})
