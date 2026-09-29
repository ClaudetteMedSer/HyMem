"""Independent v2 mechanical checks; no model accuracy is inferred from them."""
import ast
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import pytest

from hymem.extraction import grounding as old
from hymem.extraction import grounding_v2 as new
from tools.diagnostics import luna_semantic_cases as controls


ROOT = Path(__file__).resolve().parents[1]
OLD_SHA = 'dd1a49b56abf569b4476a0b735e67e72a86723bf88e3a4739998aceeabe19c18'


def _rebind_source(source):
    data = asdict(source)
    data['contexts'] = tuple(new.GroundingContext(**item) for item in data['contexts'])
    return new.GroundingSource(**data)


def test_old_source_and_fixed_oracle_stay_immutable():
    assert hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest() == OLD_SHA
    assert controls.suite_sha256() == '511d99b361c6cce515d15cb93966d03c0118e4dc24e88b902fb9da6c9d9b9925'


def test_only_two_assignment_values_change_not_contract_mechanics():
    def normalized(path):
        tree = ast.parse(Path(path).read_text())
        changed = []
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id in ('_SYSTEM', 'GROUNDING_CONTRACT_VERSION')):
                changed.append(node.targets[0].id)
                node.value = ast.Constant('VERSIONED')
        assert set(changed) == {'_SYSTEM', 'GROUNDING_CONTRACT_VERSION'}
        return ast.dump(tree, include_attributes=False)
    assert normalized(old.__file__) == normalized(new.__file__)


@pytest.mark.parametrize('index', range(24))
def test_all_frozen_cases_preserve_every_source_and_claim_field(index):
    case = controls.cases()[index]
    prior_request, prior_batch = old.build_grounding_request(case.triples, case.sources)
    sources = tuple(_rebind_source(source) for source in case.sources)
    request, batch = new.build_grounding_request(case.triples, sources)
    prior_wire, wire = json.loads(prior_request.user), json.loads(request.user)
    assert prior_batch.batch_sha256 != batch.batch_sha256
    assert prior_wire.pop('batch_sha256') != wire.pop('batch_sha256')
    assert prior_wire['batch'].pop('version') == 'source-grounding-v1'
    assert wire['batch'].pop('version') == 'source-grounding-v2'
    assert wire == prior_wire
    assert (request.temperature, request.response_format, request.max_tokens) == (0.0, 'json', 4096)
    for item in case.expected:
        assert item.rationale not in request.user and item.rationale not in request.system
    # One fixture ID is also a required contract field, not label leakage.
    if case.case_id != 'temporal_scope':
        assert case.case_id not in request.system
    assert case.triples == batch.triples


@pytest.mark.parametrize('sender,receiver', [(old, new), (new, old)])
def test_schema_mismatch_rejected_even_with_current_batch_hash(sender, receiver):
    case = controls.cases()[0]
    sources = case.sources if receiver is old else tuple(_rebind_source(s) for s in case.sources)
    _, batch = receiver.build_grounding_request(case.triples, sources)
    verdict = {'schema': sender.GROUNDING_CONTRACT_VERSION,
               'batch_sha256': batch.batch_sha256, 'complete': True,
               'verdicts': [{'index': 0, 'status': 'supported', 'predicate': 'uses',
                   'evidence': [{'source_message_id': 101, 'region': 'owned',
                                 'quote': sources[0].content}]}]}
    with pytest.raises(receiver.GroundingContractError, match='response:schema'):
        receiver.parse_grounding_response(json.dumps(verdict), batch)


# Re-execute every prior adversarial mechanical test against v2 in memory.
# Only the module binding and hard-coded response schema are changed; no test
# assertions, cases, input text or old source files are rewritten.
for _file, _substitutions in (
    ('test_extraction_grounding_contract.py', (
        ('from hymem.extraction.grounding import (', 'from hymem.extraction.grounding_v2 import ('),
        ('"schema": "source-grounding-v1"', '"schema": "source-grounding-v2"'),
    )),
    ('test_extraction_grounding_root.py', (
        ('from hymem.extraction import grounding as g', 'from hymem.extraction import grounding_v2 as g'),
    )),
):
    _path = ROOT / 'tests' / _file
    _source = _path.read_text()
    for _before, _after in _substitutions:
        assert _source.count(_before) == 1
        _source = _source.replace(_before, _after, 1)
    _namespace = {'__name__': __name__ + '.' + _path.stem, '__file__': str(_path)}
    exec(compile(_source, str(_path), 'exec'), _namespace)
    for _name, _value in _namespace.items():
        if _name.startswith('test_') and callable(_value):
            globals()['test_rebound_' + _path.stem + '_' + _name[5:]] = _value
