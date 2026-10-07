"""Independent claim-first bundle whole-entry, private-replay and fault tests."""
import hashlib
from pathlib import Path

_old = Path(__file__).with_name('test_luna_classification_bundle_root.py')
_raw = _old.read_bytes()
assert hashlib.sha256(_raw).hexdigest() == '2fc50345424864c749360d301986bd80b1f3be349e456bc9093af898b8324a86'
_source = _raw.decode().replace(
    'from tools.diagnostics import luna_classification_bundle as derive',
    'from tools.diagnostics import luna_classification_bundle_v3 as derive')
_source = _source.replace("Path('/private/tmp/hymem-classification-root-ZP2zuU/candidate')", 'derive.CANDIDATE')
for _before,_after in (
    ('grounding_classification_gate_v1','grounding_classification_gate_v3'),
    ('grounding_classification_v1','grounding_classification_v3'),
    ('codex_subscription_classification_v1','codex_subscription_classification_v3'),
    ('luna-classification-probe','luna-classification-v3-probe'),
):
    _source = _source.replace(_before,_after)
_source = _source.replace("assert 'predicate' not in json.loads(request.user)['batch']['candidates'][0]",
    "assert json.loads(request.user)['batch']['candidates'][0]['predicate'] == batch.triples[0].predicate")
_before = """        states=['n']*22; citations=[[] for _ in states]
        if pred is not None:
            pos=g.PREDICATE_ORDER.index(pred); states[pos]='e'
            citations[pos]=list(range(len(pool)))
        else: pool=[]
        items.append(dict(index=i,states=states,evidence_pool=pool,citations=citations))"""
_after = """        def assessment(predicate):
            if predicate != pred: return dict(state='not_established',support=None)
            components=['attribution_and_roles','relation_and_polarity'] + [
                k for k in ('value_text','value_numeric','value_unit','temporal_scope') if getattr(triple,k) is not None]
            return dict(state='supported',support=dict(evidence=pool,checks={
                k:dict(state='supported',evidence_indices=list(range(len(pool)))) for k in components}))
        items.append(dict(index=i,original=assessment(triple.predicate),alternatives=None if pred==triple.predicate else {
            p:assessment(p) for p in g.PREDICATE_ORDER if p!=triple.predicate}))"""
assert _source.count(_before) == 1
_source = _source.replace(_before,_after)
exec(compile(_source,str(_old),'exec'),globals())


def test_ambiguous_original_canary_still_fails_without_reroll():
    script = PREAMBLE + r'''
class Ambiguous(CanaryClient):
    def complete_grounding(self,request,batch):
        raw=json.loads(super().complete_grounding(request,batch))
        for item in raw['classifications']:
            item.update(original=dict(state='ambiguous',support=None),alternatives=None)
        return json.dumps(raw)
client=Ambiguous()
report=check.run_canary(gold,chunk,client,candidate=Path(sys.argv[1]))
assert not report['passed'] and sum(client.seen.values())==1
print(json.dumps(dict(rejected=True)))
'''
    assert execute(script) == {'rejected':True}


@pytest.mark.parametrize('fault',['duplicate_quote','no_checks','actor_not_established',
    'relation_ambiguous','bool_index','orphan_quote','extra_check','old_schema','extra_alternatives'])
def test_ledger_faults_reject_in_real_control_path(fault):
    script = PREAMBLE + r'''
case=next(c for c in cases.cases() if c.category=='supported')
class Damaged(Judge):
    def complete_grounding(self,request,batch):
        raw=json.loads(super().complete_grounding(request,batch))
        item=raw['classifications'][0]; support=item['original']['support']
        fault=sys.argv[3]
        if fault=='duplicate_quote': support['evidence'].append(support['evidence'][0])
        elif fault=='no_checks': support['checks']={}
        elif fault=='actor_not_established': support['checks']['attribution_and_roles']['state']='not_established'
        elif fault=='relation_ambiguous': support['checks']['relation_and_polarity']['state']='ambiguous'
        elif fault=='bool_index': support['checks']['attribution_and_roles']['evidence_indices']=[True]
        elif fault=='orphan_quote':
            q=dict(support['evidence'][0]); q['quote']=q['quote'][:1]; support['evidence'].append(q)
        elif fault=='extra_check': support['checks']['unasked']={'state':'supported','evidence_indices':[0]}
        elif fault=='extra_alternatives': item['alternatives']={}
        else: raw['schema']='source-grounding-classification-v2'
        return json.dumps(raw)
client=Damaged(case)
out=core.run_control(case,client,g,record=lambda x:None)
assert not out['passed'] and out['malformed_code'] is not None and len(client.requests)==1
print(json.dumps(dict(rejected=True)))
'''
    assert execute(script,fault) == {'rejected':True}


def test_original_visible_does_not_weaken_trusted_batch_binding():
    # Whole-entry tests above independently tamper canonical original and replay;
    # this static guard supplements, rather than replaces, that exercised path.
    adapter=(BUNDLE/'code/benchmarks/codex_subscription_classification_v3.py').read_text()
    assert 'parents[2] / "candidate/hymem/extraction/grounding_classification_v3.py"' in adapter
    assert '435c0edf52197a5ffa9e715db24156e26109bf7445ad7c9baba2f632ba7f7a76' in adapter
    assert 'sha256:50d0ba72b0ea5290fba0541e42bc03f1880a3aa70c8014310a7b1b22be6adede' in adapter
