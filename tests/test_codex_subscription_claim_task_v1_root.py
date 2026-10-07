"""Root's pinned fake-wire matrix plus independent alternating-arm controls."""
import hashlib as _hashlib
from pathlib import Path as _Path

_template=_Path(__file__).with_name('test_codex_subscription_classification_root.py').read_bytes()
assert _hashlib.sha256(_template).hexdigest()=='2f8a10004547fd88881026d471a235c1ad7318c0443de0f9f7c81235fd0aca4f'
_source=_template.decode().replace('codex_subscription_classification_v1','codex_subscription_claim_task_v1')
_source=_source.replace('grounding_classification_v1','grounding_classification_v3')
_source=_source.replace('ClassificationSubscriptionClient','ClaimTaskSubscriptionClient')
_source=_source.replace('.complete_grounding(',".complete_arm('A', ")
_source=_source.replace('request:binding','request:arm_binding')
_source=_source.replace('c._BoundSession','c.adapter._BoundSession')
exec(compile(_source,__file__,'exec'),globals())


def test_alternating_arms_and_rotation_do_not_reuse_binding_or_schema():
    item=client(max_requests=2)
    expected=[]
    try:
        for arm,name in [('A','CairnDB'),('B','CairnDB'),('B','MapleDB'),('A','MapleDB')]:
            _,batch=request(name)
            req,newbatch=c.contract.build_arm_request(arm,batch.triples,batch.sources)
            assert newbatch==batch
            expected.append((req,c.contract.build_arm_output_schema(arm,batch)))
            assert item.complete_arm(arm,req,batch)=='{}'
            assert item._active_binding is None and item.session._binding is None
        turns=[p for s in Protocol.instances for m,p in s.calls if m=='turn/start']
        starts=[p for s in Protocol.instances for m,p in s.calls if m=='thread/start']
        assert len(turns)==len(starts)==4 and len(Protocol.instances)==2
        for (req,schema),turn,start in zip(expected,turns,starts,strict=True):
            assert turn['input']==[dict(type='text',text=req.user)]
            assert start['baseInstructions']==req.system
            assert turn['outputSchema']==schema
        assert [t['outputSchema']['properties']['schema']['enum'][0] for t in turns]==[
            g.GROUNDING_CONTRACT_VERSION,c.contract.B_SCHEMA,c.contract.B_SCHEMA,g.GROUNDING_CONTRACT_VERSION]
        assert item.observed_turns==4 and item.observed_tokens==44 and item.usage_complete
        assert item.budget.snapshot()['in_flight']==item.budget.snapshot()['reserved']==0
    finally:
        item.close()
    assert all(s.closed for s in Protocol.instances)


@pytest.mark.parametrize('arm',[None,True,1,'a','C'])
def test_unknown_arm_rejected_before_io(arm):
    item=client()
    try:
        with pytest.raises(g.GroundingContractError,match='arm:invalid'):
            item.complete_arm(arm,*request())
        assert not Protocol.instances and item.observed_turns==0
    finally: item.close()


def test_cross_arm_request_after_valid_dispatch_closes_session_without_new_turn():
    item=client()
    req,batch=request()
    try:
        item.complete_arm('A',req,batch)
        with pytest.raises(g.GroundingContractError,match='request:arm_binding'):
            item.complete_arm('B',req,batch)
        assert Protocol.instances[0].closed and item.session is None
        assert item.observed_turns==1 and item.observed_tokens==11
        assert item._active_binding is None
    finally: item.close()


def test_no_generic_dispatch_fallback():
    item=client()
    req,batch=request()
    try:
        for action in [lambda:item.complete(req),lambda:item.chat([]),
                       lambda:item.complete_grounding(req,batch)]:
            with pytest.raises(ValueError): action()
        assert not Protocol.instances and item.observed_turns==0
    finally: item.close()


def test_captured_helper_not_public_cache_and_schema_is_not_shared():
    from tools.diagnostics import luna_claim_task_contract_v1 as public
    assert c.contract is not public and c.contract.v3 is g
    req,batch=request()
    a=c.contract.build_arm_output_schema('A',batch)
    b=c.contract.build_arm_output_schema('B',batch)
    a['$defs'].clear()
    b['properties'].clear()
    assert c.contract.build_arm_output_schema('A',batch)==g.build_output_schema(batch)
    assert c.contract.build_arm_output_schema('B',batch)==public.build_arm_output_schema('B',batch)
