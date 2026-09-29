"""Independent physical-runtime verification of the inactive policy candidate."""
import hashlib
from pathlib import Path
from types import SimpleNamespace

from tools.diagnostics import luna_semantic_candidate as old_builder
from tools.diagnostics import luna_semantic_candidate_v2 as builder


REPO = Path(__file__).resolve().parents[3]
OLD_CANDIDATE = Path('/private/tmp/hymem-semantic-step2-v2-candidate-20260929')
OLD_STAMP = Path('/private/tmp/hymem-semantic-step2-v2-map-20260929.json')


def _prepare(source, stamp, target, target_stamp, repo):
    return builder.prepare(source, stamp, OLD_CANDIDATE, OLD_STAMP,
                           target, target_stamp, repo)


TEST_FACADE = SimpleNamespace(**{
    name: getattr(old_builder, name)
    for name in ('inventory', 'derive_chunk', 'derive_contract', 'GROUNDING', 'GATE', 'CHUNK', 'CONTRACT')
}, prepare=_prepare)


def test_physical_delta_is_one_source_and_old_builder_is_unchanged(candidate):
    previous = old_builder.inventory(OLD_CANDIDATE)
    current = old_builder.inventory(candidate)
    assert set(previous) == set(current) and len(current) == 510
    assert {key for key in current if current[key] != previous[key]} == {
        'hymem/extraction/grounding.py'}
    assert current['hymem/extraction/grounding.py'] == (
        '377a688caf183f3645be246b77a445bd94d053def00fff87589061b4b2fc31ec')
    assert hashlib.sha256(Path(old_builder.__file__).read_bytes()).hexdigest() == (
        'a10ee5c5a1ba4f6a2694a88570a399c5fe081db02f0cb0ecfa5b92e5db06d7b2')


# Exercise both previously accepted integration suites against the actual v2
# candidate. Only test imports and synthetic response versions are rebound in
# memory; ordinary request fixtures and publication assertions stay unchanged.
for _file in ('test_luna_semantic_candidate.py', 'test_luna_semantic_root.py'):
    _path = Path(__file__).with_name(_file)
    _source = _path.read_text()
    _before = 'from tools.diagnostics import luna_semantic_candidate as builder'
    assert _source.count(_before) == 1
    _source = _source.replace(_before, '', 1)
    assert 'source-grounding-v1' in _source
    _source = _source.replace('source-grounding-v1', 'source-grounding-v2')
    _namespace = {'__name__': __name__ + '.' + _path.stem, '__file__': str(_path),
                  'builder': TEST_FACADE}
    exec(compile(_source, str(_path), 'exec'), _namespace)
    for _name, _value in _namespace.items():
        if _name.startswith('test_') and callable(_value):
            globals()['test_rebound_' + _path.stem + '_' + _name[5:]] = _value
        elif _name == 'candidate':
            globals()[_name] = _value
