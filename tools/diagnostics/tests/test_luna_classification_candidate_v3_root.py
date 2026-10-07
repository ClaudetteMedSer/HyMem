"""Independent physical-candidate fault matrix rebound to claim-first v3."""
import hashlib
from pathlib import Path

from tools.diagnostics import luna_classification_candidate_v3 as builder

_old = Path(__file__).with_name('test_luna_classification_candidate_root.py')
_raw = _old.read_bytes()
assert hashlib.sha256(_raw).hexdigest() == '0c1b6f06d3b4313c828810223bd09a81e4bb0af13f09afb3555b2f051b9ea570'
_source = _raw.decode().replace(
    'from tools.diagnostics import luna_classification_candidate as builder',
    'from tools.diagnostics import luna_classification_candidate_v3 as builder')
_source = _source.replace('grounding_classification_v1', 'grounding_classification_v3')
_source = _source.replace('grounding_classification_gate_v1', 'grounding_classification_gate_v3')
_source = _source.replace('builder.old.', 'builder.prior.old.')
_source = _source.replace("assert all('predicate' not in t for t in json.loads(request.user)['batch']['candidates'])",
    "assert [t['predicate'] for t in json.loads(request.user)['batch']['candidates']] == [t.predicate for t in batch.triples]")
_old_wire = """            states=['n']*22; citations=[[] for _ in states]; pool=[]
            if pred is not None:
                pos=g.PREDICATE_ORDER.index(pred); states[pos]='e'; citations[pos]=[0]
                pool=[dict(source_message_id=71,region='owned',quote=text)]
            items.append(dict(index=i,states=states,evidence_pool=pool,citations=citations))"""
_new_wire = """            def assessment(predicate):
                if predicate != pred: return dict(state='not_established',support=None)
                checks=['attribution_and_roles','relation_and_polarity'] + [
                    k for k in ('value_text','value_numeric','value_unit','temporal_scope') if getattr(t,k) is not None]
                return dict(state='supported',support=dict(
                    evidence=[dict(source_message_id=71,region='owned',quote=text)],
                    checks={k:dict(state='supported',evidence_indices=[0]) for k in checks}))
            items.append(dict(index=i,original=assessment(t.predicate),alternatives=None if t.predicate==pred else {
                p:assessment(p) for p in g.PREDICATE_ORDER if p!=t.predicate}))"""
assert _source.count(_old_wire) == 1
_source = _source.replace(_old_wire, _new_wire)
exec(compile(_source, str(_old), 'exec'), globals())


def test_physical_candidate_does_not_include_old_classifiers(candidate):
    for version in (1,2):
        assert not (candidate / f'hymem/extraction/grounding_classification_v{version}.py').exists()
        assert not (candidate / f'hymem/extraction/grounding_classification_gate_v{version}.py').exists()
