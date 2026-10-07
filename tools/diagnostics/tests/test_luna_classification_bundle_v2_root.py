"""Independent v2 bundle verification using the pinned whole-entry fault matrix."""
import hashlib
from pathlib import Path

_old = Path(__file__).with_name('test_luna_classification_bundle_root.py')
_raw = _old.read_bytes()
assert hashlib.sha256(_raw).hexdigest() == '2fc50345424864c749360d301986bd80b1f3be349e456bc9093af898b8324a86'
_source = _raw.decode().replace(
    'from tools.diagnostics import luna_classification_bundle as derive',
    'from tools.diagnostics import luna_classification_bundle_v2 as derive')
_source = _source.replace("Path('/private/tmp/hymem-classification-root-ZP2zuU/candidate')", 'derive.CANDIDATE')
for _before, _after in (
    ('grounding_classification_gate_v1', 'grounding_classification_gate_v2'),
    ('grounding_classification_v1', 'grounding_classification_v2'),
    ('codex_subscription_classification_v1', 'codex_subscription_classification_v2'),
    ('luna-classification-probe', 'luna-classification-v2-probe'),
):
    _source = _source.replace(_before, _after)
_before = """        states=['n']*22; citations=[[] for _ in states]
        if pred is not None:
            pos=g.PREDICATE_ORDER.index(pred); states[pos]='e'
            citations[pos]=list(range(len(pool)))
        else: pool=[]
        items.append(dict(index=i,states=states,evidence_pool=pool,citations=citations))"""
_after = """        states=['not_established']*22; groups=[]
        if pred is not None:
            pos=g.PREDICATE_ORDER.index(pred); states[pos]='supported'
            groups=[dict(predicates=[pred],evidence=pool)]
        items.append(dict(index=i,states=states,support_groups=groups))"""
assert _source.count(_before) == 1
_source = _source.replace(_before, _after)
exec(compile(_source, str(_old), 'exec'), globals())


def test_all_ambiguous_table_canary_still_fails():
    script = PREAMBLE + r'''
class Ambiguous(CanaryClient):
    def complete_grounding(self,request,batch):
        raw=json.loads(super().complete_grounding(request,batch))
        for item in raw['classifications']:
            item['states']=['ambiguous']*22
            item['support_groups']=[]
        return json.dumps(raw)
client=Ambiguous()
report=check.run_canary(gold,chunk,client,candidate=Path(sys.argv[1]))
assert not report['passed']
assert sum(client.seen.values())==1
print(json.dumps(dict(rejected=True,grounding_calls=1)))
'''
    assert execute(script) == {'rejected': True, 'grounding_calls': 1}


@pytest.mark.parametrize('fault', ['duplicate', 'missing', 'extra', 'overlap', 'ambiguous', 'old_schema', 'global_limit'])
def test_v2_groups_rejected_through_actual_core(fault):
    script = PREAMBLE + r'''
case=next(c for c in cases.cases() if c.category=='supported')
class Damaged(Judge):
    def complete_grounding(self,request,batch):
        raw=json.loads(super().complete_grounding(request,batch))
        item=raw['classifications'][0]
        group=item['support_groups'][0]
        fault=sys.argv[3]
        if fault=='duplicate': group['evidence'].append(group['evidence'][0])
        elif fault=='missing': item['support_groups']=[]
        elif fault=='extra':
            group['predicates'].append(next(p for p in g.PREDICATE_ORDER if p not in group['predicates']))
        elif fault=='overlap': item['support_groups'].append(group)
        elif fault=='old_schema': raw['schema']='source-grounding-classification-v1'
        elif fault=='global_limit':
            source=next(s for s in batch.sources if s.source_message_id==batch.triples[0].source_message_id)
            assert len(source.content)>=9
            extra=next(p for p in g.PREDICATE_ORDER if p not in group['predicates'])
            item['states'][g.PREDICATE_ORDER.index(extra)]='supported'
            group['evidence']=[dict(source_message_id=source.source_message_id,region='owned',quote=source.content[:i]) for i in range(1,9)]
            item['support_groups'].append(dict(predicates=[extra],evidence=[dict(source_message_id=source.source_message_id,region='owned',quote=source.content[:9])]))
        else:
            item['states']=['ambiguous']*22
            item['support_groups']=[]
        return json.dumps(raw)
client=Damaged(case)
out=core.run_control(case,client,g,record=lambda x:None)
assert not out['passed'] and len(client.requests)==1
assert (out['malformed_code'] is not None)==(sys.argv[3]!='ambiguous'),out
print(json.dumps(dict(rejected=True)))
'''
    assert execute(script, fault) == {'rejected': True}


def test_group_overlap_in_evidence_is_valid_but_not_an_accuracy_test():
    script = PREAMBLE + r'''
case=next(c for c in cases.cases() if c.category=='supported')
class Shared(Judge):
    def complete_grounding(self,request,batch):
        raw=json.loads(super().complete_grounding(request,batch))
        item=raw['classifications'][0]
        group=item['support_groups'][0]
        extra=next(p for p in g.PREDICATE_ORDER if p not in group['predicates'])
        item['states'][g.PREDICATE_ORDER.index(extra)]='supported'
        item['support_groups'].append(dict(predicates=[extra],evidence=group['evidence']))
        return json.dumps(raw)
client=Shared(case)
out=core.run_control(case,client,g,record=lambda x:None)
assert out['passed'] and len(client.requests)==1
print(json.dumps(dict(original_retained=True)))
'''
    assert execute(script) == {'original_retained': True}
