"""Root fake-wire lifecycle checks plus independent cross-stage fault controls."""
import ast as _ast
import hashlib as _hashlib
from pathlib import Path as _Path

_template = _Path(__file__).with_name('test_codex_subscription_classification_root.py').read_bytes()
assert _hashlib.sha256(_template).hexdigest() == '2f8a10004547fd88881026d471a235c1ad7318c0443de0f9f7c81235fd0aca4f'
_source = _template.decode().replace('codex_subscription_classification_v1', 'codex_subscription_staged_v1')
_source = _source.replace('grounding_classification_v1', 'grounding_classification_v4')
_source = _source.replace('ClassificationSubscriptionClient', 'StagedSubscriptionClient')
_source = _source.replace('classification_only', 'staged_only')
_source = _source.replace('g.build_grounding_request(', 's.build_original_request(')
_source = _source.replace('g.build_output_schema(', 's.build_original_output_schema(')
_source = _source.replace('pinned_classification_import_mismatch', 'pinned_contract_import_mismatch')
_source = 'from hymem.extraction import grounding_staged_v1 as s\n' + _source


class _StageCalls(_ast.NodeTransformer):
    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, _ast.Attribute) and node.func.attr == 'complete_grounding':
            node.func.attr = 'complete_stage'
            node.keywords.extend([_ast.keyword(arg='stage', value=_ast.Constant('original')),
                                  _ast.keyword(arg='recheck', value=_ast.Constant(False))])
        return node


_tree = _StageCalls().visit(_ast.parse(_source))
exec(compile(_ast.fix_missing_locations(_tree), __file__, 'exec'), globals())


def stage_requests():
    req, batch = request()
    raw = json.dumps(dict(schema=s.ORIGINAL_SCHEMA, batch_sha256=batch.batch_sha256,
        complete=True, originals=[dict(index=0, original=dict(state='not_established', support=None))]))
    alt, bound = s.build_alternatives_request(batch, raw)
    corrected = replace(batch.triples[0], predicate='prefers')
    recheck, corrected_batch = s.build_original_request((corrected,), batch.sources)
    return [(req, batch, 'original', False), (alt, bound, 'alternatives', False),
            (recheck, corrected_batch, 'original', True)]


def test_three_stage_budget_is_shared_and_schema_changes_with_fresh_threads():
    limits = c.warm.BudgetLimits(3, 10000, 1000)
    budget = c.warm.SharedBudget(limits)
    item = c.StagedSubscriptionClient('unused', budget, 'unit', limits,
        session_factory=Protocol, max_requests=2)
    stages = stage_requests()
    try:
        for stage in stages:
            assert item.complete_stage(*stage) == '{}'
            assert item._active_binding is None and item.session._binding is None
        turns = [p for session in Protocol.instances for m, p in session.calls if m == 'turn/start']
        starts = [p for session in Protocol.instances for m, p in session.calls if m == 'thread/start']
        assert len(turns) == len(starts) == 3 and len(Protocol.instances) == 2
        for (req, batch, stage, _), turn, start in zip(stages, turns, starts, strict=True):
            schema = s.build_original_output_schema(batch) if stage == 'original' else s.build_alternatives_output_schema(batch)
            assert turn['outputSchema'] == schema
            assert turn['input'] == [dict(type='text', text=req.user)]
            assert start['baseInstructions'] == req.system
        assert item.observed_turns == 3 and item.observed_tokens == 33
        with pytest.raises(c.warm.ConcurrentStop):
            item.complete_stage(*stages[0])
        assert item.observed_turns == 3 and budget.snapshot()['turns'] == 3
        assert sum(m == 'turn/start' for session in Protocol.instances for m, _ in session.calls) == 3
    finally:
        item.close()
    assert all(x.closed for x in Protocol.instances)
    assert budget.snapshot()['in_flight'] == budget.snapshot()['reserved'] == 0


@pytest.mark.parametrize('stage,recheck', [(None,False),(True,False),('Original',False),
    ('alternatives',True),('original',0),('original',1),('original',None)])
def test_invalid_stage_control_has_no_admission(stage, recheck):
    item = client()
    req, batch = request()
    try:
        with pytest.raises((g.GroundingContractError, c.warm.base.SubscriptionTransportError, ValueError)):
            item.complete_stage(req,batch,stage,recheck)
        assert not Protocol.instances and item.observed_turns == 0
        assert item.budget.snapshot()['turns'] == 0
    finally:
        item.close()


@pytest.mark.parametrize('first,second', [(0,1),(1,0),(1,2)])
def test_cross_stage_request_after_success_closes_without_second_admission(first, second):
    stages = stage_requests()
    item = client()
    try:
        item.complete_stage(*stages[first])
        req = stages[first][0]
        _, batch, stage, recheck = stages[second]
        with pytest.raises((g.GroundingContractError, c.warm.base.SubscriptionTransportError)):
            item.complete_stage(req,batch,stage,recheck)
        assert item.observed_turns == 1 and item.observed_tokens == 11
        assert Protocol.instances[0].closed and item.session is None
    finally:
        item.close()


def test_alternatives_forged_prior_or_indices_never_admitted():
    req, bound, stage, recheck = stage_requests()[1]
    item = client()
    try:
        for corrupt in (replace(bound,original_response_sha256='0'*64),
                        replace(bound,negative_indices=(False,))):
            with pytest.raises(g.GroundingContractError):
                item.complete_stage(req,corrupt,stage,recheck)
        assert not Protocol.instances and item.observed_turns == 0
    finally:
        item.close()


def test_generic_grounding_entry_cannot_bypass_stages():
    item = client()
    try:
        for name, args in [('complete', (request()[0],)), ('chat', ([],)),
                           ('complete_grounding', request())]:
            with pytest.raises(ValueError):
                getattr(item,name)(*args)
        assert not Protocol.instances and item.observed_turns == 0
    finally:
        item.close()


@pytest.mark.parametrize('helper', ['v4_parser', 'staged_parser', 'source_mapper_module'])
def test_foreign_transitive_helper_import_rejected_before_io(helper):
    repo = Path(__file__).resolve().parents[1]
    patches = {
        'v4_parser': 'g._parse_v2=lambda *a,**k:None',
        'staged_parser': 's._parse_v2=lambda *a,**k:None',
        'source_mapper_module': 'gate._source_gate=types.SimpleNamespace(**vars(gate._source_gate))',
    }
    code = ('import sys,types;'
        f'sys.path.insert(0,{str(repo)!r});'
        'from hymem.extraction import grounding_classification_v4 as g, grounding_staged_v1 as s, grounding_staged_gate_v1 as gate;'
        + patches[helper] + ';'
        'from benchmarks import codex_subscription_staged_v1')
    out = subprocess.run([sys.executable, '-I', '-B', '-c', code], capture_output=True, text=True, timeout=20)
    assert out.returncode != 0
    assert 'pinned_contract_dependency_mismatch' in out.stderr
