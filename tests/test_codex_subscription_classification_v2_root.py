"""Pinned root transport matrix rebound to v2, plus independent integration tests.

The original fake-wire checks are reused byte-for-byte except the explicit
contract/adapter module name. No live runtime or provider is started.
"""
import hashlib as _hashlib
from pathlib import Path as _Path

_template = _Path(__file__).with_name('test_codex_subscription_classification_root.py').read_bytes()
assert _hashlib.sha256(_template).hexdigest() == '2f8a10004547fd88881026d471a235c1ad7318c0443de0f9f7c81235fd0aca4f'
_rebound = _template.decode().replace('classification_v1', 'classification_v2')
exec(compile(_rebound, __file__, 'exec'), globals())


def test_old_v1_batch_is_rejected_before_dispatch_not_converted():
    from hymem.extraction import grounding_classification_v1 as prior
    req, batch = request()
    old_request, old_batch = prior.build_grounding_request(batch.triples, batch.sources)
    item = client()
    try:
        with pytest.raises(g.GroundingContractError):
            item.complete_grounding(old_request, old_batch)
        assert not Protocol.instances and item.observed_turns == 0
        assert item._active_binding is None
    finally:
        item.close()


def test_v2_schema_is_the_group_contract_without_old_pool_or_runtime_fallback():
    item = client()
    try:
        req, batch = request()
        item.complete_grounding(req, batch)
        wire = next(p for m, p in Protocol.instances[0].calls if m == 'turn/start')
        schema = wire['outputSchema']
        assert schema['properties']['schema']['enum'] == ['source-grounding-classification-v2']
        fields = schema['properties']['classifications']['items']['anyOf'][0]['properties']
        assert set(fields) == {'index', 'states', 'support_groups'}
        assert set(fields['states']['items']['enum']) == {'supported', 'not_established', 'ambiguous'}
        assert item.requested_controls[-1]['response_format_effective'] is None
    finally:
        item.close()


def test_gate_and_real_inherited_protocol_two_turns_with_invented_verdicts():
    import runpy
    support = runpy.run_path(str(Path(__file__).with_name('test_grounding_classification_gate_v2_root.py')))
    gate, reply = support['gate'], support['reply']
    original, records = support['inputs'](2, 'prefers')
    class Synthetic(Protocol):
        instances = []
        response = None
        def rpc(self, method, params, **kwargs):
            result = super().rpc(method, params, **kwargs)
            if method == 'turn/start':
                for event in self.pending:
                    if event['method'] == 'item/completed':
                        event['params']['item']['text'] = self.response
            return result
    limit = c.warm.BudgetLimits(2, 1000, 120)
    item = c.ClassificationSubscriptionClient('unused', c.warm.SharedBudget(limit), 'q', limit,
                                             session_factory=Synthetic)
    calls = []
    def invoke(req, batch, recheck):
        calls.append(recheck)
        Synthetic.response = reply(batch, ['uses'] * len(batch.triples))
        return item.complete_grounding(req, batch)
    try:
        result = gate.ground_triples(original, records, (), '', invoke)
        assert calls == [False, True]
        assert all(t.predicate == 'uses' for t in result)
        assert all(t.predicate == 'prefers' for t in original)
        assert item.observed_turns == 2 and item.observed_tokens == 22 and item.usage_complete
        wire = [p for s in Synthetic.instances for m, p in s.calls if m == 'turn/start']
        hashes = [w['outputSchema']['properties']['batch_sha256']['enum'][0] for w in wire]
        assert len(set(hashes)) == 2
        assert all(x['output_schema_sent'] and x['output_schema_acknowledged'] for x in item.requested_controls)
    finally:
        item.close()
    assert all(s.closed for s in Synthetic.instances)


def test_v1_and_v2_warm_wrappers_do_not_share_internal_module_namespace():
    from benchmarks import codex_subscription_classification_v1 as prior
    assert c.warm.__name__ != prior.warm.__name__
    assert c.classification is g and c.classification is not prior.classification
    assert c.warm.base.MODEL == prior.warm.base.MODEL
