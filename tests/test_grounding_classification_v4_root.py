"""Independent region-binding proof and unchanged v3 safety matrix."""
import hashlib as _hashlib
from pathlib import Path as _Path

_previous=_Path(__file__).with_name('test_grounding_classification_v3_root.py').read_bytes()
assert _hashlib.sha256(_previous).hexdigest()=='39bf91d82c06199df1aeccb29459296d802addc3a2b8dfd197a4cd01a8226f59'
_source=_previous.decode().replace('grounding_classification_v3 as g','grounding_classification_v4 as g')
exec(compile(_source,__file__,'exec'),globals())

from hymem.extraction import grounding_classification_v3 as v3


def region_enum(schema,index):
    return schema['$defs'][f'assessment_{index}']['properties']['support']['anyOf'][1][
        'properties']['evidence']['items']['properties']['region']['enum']


@pytest.mark.parametrize('index',range(24))
def test_schema_changes_only_version_binding_and_per_source_regions(index):
    case=cases.cases()[index]
    req,batch=g.build_grounding_request(case.triples,source_objects(case))
    oldreq,oldbatch=v3.build_grounding_request(case.triples,source_objects(case))
    expected=v3.build_output_schema(oldbatch)
    expected['properties']['schema']['enum']=[g.GROUNDING_CONTRACT_VERSION]
    expected['properties']['batch_sha256']['enum']=[batch.batch_sha256]
    for i,triple in enumerate(batch.triples):
        owner=next(s for s in batch.sources if s.source_message_id==triple.source_message_id)
        enum=region_enum(expected,i)
        enum[:]=sorted({'owned',*(c.region for c in owner.contexts)})
    assert g.build_output_schema(batch)==expected
    assert req.system.replace(g.GROUNDING_CONTRACT_VERSION,v3.GROUNDING_CONTRACT_VERSION)==oldreq.system
    canonical=json.loads(batch.canonical_json);canonical['version']=v3.GROUNDING_CONTRACT_VERSION
    assert canonical==json.loads(oldbatch.canonical_json)
    assert batch.batch_sha256!=oldbatch.batch_sha256


def test_observed_wrong_region_shape_rejected_by_schema_not_rewritten():
    owned='For the archive I prefer AsterDB.'
    boundary='AsterDB is the database under discussion.'
    triple=g.Triple('Iris','uses','AsterDB',1,source_message_id=7)
    source=g.GroundingSource(7,owned,contexts=(g.GroundingContext('boundary',boundary,len(owned)),))
    _,batch=g.build_grounding_request((triple,),(source,))
    row=item(triple,original='not_established',positives=['prefers'],evidence=[
        quote(7,boundary,'boundary'),quote(7,owned,'conversation_0')])
    raw=response(batch,[row]); frozen=copy.deepcopy(raw)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(raw,g.build_output_schema(batch))
    with pytest.raises(g.GroundingContractError,match='evidence:context_missing'):
        g.parse_grounding_response(json.dumps(raw),batch)
    assert raw==frozen
    row['alternatives']['prefers']['support']['evidence'][1]['region']='owned'
    jsonschema.validate(response(batch,[row]),g.build_output_schema(batch))
    assert parse(batch,[row]).verdicts[0].predicate=='prefers'


def test_regions_are_per_claim_not_batch_union_or_position():
    sources=(g.GroundingSource(8,'owned eight',contexts=(g.GroundingContext('header','header eight',11),)),
             g.GroundingSource(7,'owned seven',contexts=(g.GroundingContext('boundary','boundary seven',11),)),
             g.GroundingSource(9,'unused nine',contexts=(g.GroundingContext('prelude','unused',11),)))
    triples=(g.Triple('Iris','uses','CedarTool',1,source_message_id=7),
             g.Triple('Iris','uses','MapleTool',1,source_message_id=8))
    _,batch=g.build_grounding_request(triples,sources)
    schema=g.build_output_schema(batch)
    assert region_enum(schema,0)==['boundary','owned']
    assert region_enum(schema,1)==['header','owned']
    region_enum(schema,0).append('conversation_0')
    assert region_enum(g.build_output_schema(batch),0)==['boundary','owned']


def test_real_region_does_not_relax_prefix_guard():
    case=cases.cases()[19]
    _,batch=g.build_grounding_request(case.triples,source_objects(case))
    source=batch.sources[0]
    evidence=[quote(source.source_message_id,'Later I prefer that editor.'),
              quote(source.source_message_id,source.contexts[0].content,'conversation_0')]
    row=item(batch.triples[0],evidence=evidence)
    jsonschema.validate(response(batch,[row]),g.build_output_schema(batch))
    with pytest.raises(g.GroundingContractError,match='evidence:context_scope'):
        parse(batch,[row])


def test_legacy_source_has_owned_only_and_cross_version_cannot_dispatch():
    req,batch=g.build_grounding_request((g.Triple('Iris','uses','CedarTool',1,source_message_id=None),),
        (g.GroundingSource(None,'Iris uses CedarTool.'),))
    assert region_enum(g.build_output_schema(batch),0)==['owned']
    oldreq,oldbatch=v3.build_grounding_request(batch.triples,batch.sources)
    for request,trusted in ((oldreq,batch),(req,oldbatch)):
        with pytest.raises(g.GroundingContractError):g.validate_request(request,trusted)
    assert _hashlib.sha256(_Path(v3.__file__).read_bytes()).hexdigest()==(
        '435c0edf52197a5ffa9e715db24156e26109bf7445ad7c9baba2f632ba7f7a76')
