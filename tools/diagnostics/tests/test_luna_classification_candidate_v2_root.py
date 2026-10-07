"""Root replay of the frozen physical-candidate fault matrix against v2."""
import hashlib
from pathlib import Path

from tools.diagnostics import luna_classification_candidate_v2 as builder

_old = Path(__file__).with_name('test_luna_classification_candidate_root.py')
_raw = _old.read_bytes()
assert hashlib.sha256(_raw).hexdigest() == '0c1b6f06d3b4313c828810223bd09a81e4bb0af13f09afb3555b2f051b9ea570'
_source = _raw.decode().replace(
    'from tools.diagnostics import luna_classification_candidate as builder',
    'from tools.diagnostics import luna_classification_candidate_v2 as builder')
_source = _source.replace('grounding_classification_v1', 'grounding_classification_v2')
_source = _source.replace('grounding_classification_gate_v1', 'grounding_classification_gate_v2')
_source = _source.replace('builder.old.', 'builder.prior.old.')
_old_wire = """            states=['n']*22; citations=[[] for _ in states]; pool=[]
            if pred is not None:
                pos=g.PREDICATE_ORDER.index(pred); states[pos]='e'; citations[pos]=[0]
                pool=[dict(source_message_id=71,region='owned',quote=text)]
            items.append(dict(index=i,states=states,evidence_pool=pool,citations=citations))"""
_new_wire = """            states=['not_established']*22; groups=[]
            if pred is not None:
                pos=g.PREDICATE_ORDER.index(pred); states[pos]='supported'
                groups=[dict(predicates=[pred],evidence=[dict(source_message_id=71,region='owned',quote=text)])]
            items.append(dict(index=i,states=states,support_groups=groups))"""
assert _source.count(_old_wire) == 1
_source = _source.replace(_old_wire, _new_wire)
exec(compile(_source, str(_old), 'exec'), globals())


def test_v2_physical_candidate_does_not_include_v1_classifier(candidate):
    assert not (candidate / 'hymem/extraction/grounding_classification_v1.py').exists()
    assert not (candidate / 'hymem/extraction/grounding_classification_gate_v1.py').exists()
