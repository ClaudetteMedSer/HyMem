"""Fresh invented paired controls, not live-model results or production memory.

Labels describe semantic support under the permitted-source policy. They are
never supplied in model requests. Structural ledger checks alone cannot establish
these labels; use separately held-out cases before claiming general improvement.
"""
from copy import deepcopy


def _source(cid, mid, text, role='user', context=None):
    offset = len(context) if context is not None else 0
    return {'chunk_id': cid, 'message_id': mid, 'role': role,
            'source_peer_id': None, 'source_workspace_id': None,
            'start': offset, 'end': offset + len(text), 'visible_content': text,
            'interpretation_only_context': ({'message_id': mid, 'role': role,
                'source_peer_id': None, 'source_workspace_id': None, 'start': 0,
                'end': offset, 'content': context} if context is not None else None)}


def _payload(sources, title, body, summary, prior=''):
    return {'schema': 'digest-fidelity-decisions-v9', 'source_catalog': sources,
        'items': [{'index': 0, 'candidate_title': title, 'candidate_body': body,
                   'candidate_outcome': 'informational', 'candidate_key_entities': [],
                   'cited_source_ids': [source['chunk_id'] for source in sources]}],
        'procedure_items': [], 'summary_item': {'index': 0, 'candidate_raw_summary': summary,
            'candidate_summary': summary, 'candidate_is_noop': False,
            'new_source_ids': [source['chunk_id'] for source in sources],
            'prior_derived_summary': prior}}


def build_cases():
    cases = []

    def pair(name, faithful, defective, field, reason, kind='episode'):
        for label, payload in (('faithful', faithful), ('defective', defective)):
            cases.append({'id': name + '-' + label, 'pair': name,
                          'target': {'kind': kind, 'index': 0, 'field': field,
                                     'expected': 'supported' if label == 'faithful' else 'unsupported'},
                          'reason': reason, 'payload': deepcopy(payload)})

    faithful = _payload([_source('identity-a', 201, 'I plan to attend a ceramics class.')],
        'Ceramics class plan', 'The user plans to attend a ceramics class.',
        'A ceramics class is planned.', prior='Rhea previously discussed evening classes.')
    defective = deepcopy(faithful)
    defective['items'][0]['candidate_body'] = 'Rhea plans to attend a ceramics class.'
    pair('identity-authority', faithful, defective, '/candidate_body',
         'The user role is supplied, but the own cited source and metadata do not establish Rhea; a prior summary is not episode evidence.')

    faithful = _payload([_source('ferry-a', 202, 'I considered the ferry but did not book it. I may decide next week.')],
        'Ferry booking decision', 'The user considered the ferry but did not book it.',
        'The ferry was considered but not booked.')
    defective = deepcopy(faithful)
    defective['items'][0]['candidate_body'] = 'The user booked the ferry.'
    pair('negated-outcome', faithful, defective, '/candidate_body',
         'Consideration and explicit non-booking cannot support a completed booking.')

    faithful = _payload([_source('portal-a', 203, 'The report is available in Portal A and is not available in Portal B.', role='assistant')],
        'Report available in Portal A, not Portal B',
        'The assistant said the report is available in Portal A but not Portal B.',
        'The report is available in Portal A but not Portal B.')
    defective = deepcopy(faithful)
    defective['items'][0]['candidate_title'] = 'Report available only in Portal A'
    pair('exclusivity-scope', faithful, defective, '/candidate_title',
         'Availability in one named portal and absence in another do not establish absence from every other portal.')

    faithful = _payload([_source('citation-a', 204, 'The hatch remained shut.'),
                         _source('citation-b', 205, 'The access card was rotated.')],
        'Hatch and card status', 'The hatch remained shut; the access card was rotated.',
        'The hatch remained shut and the access card was rotated.')
    defective = deepcopy(faithful)
    defective['items'][0]['cited_source_ids'] = ['citation-a']
    pair('citation-removal', faithful, defective, '/candidate_body',
         'The unchanged card claim loses its own cited support when the second citation is removed; another window record is not item authority.')

    faithful = _payload([_source('boundary-a', 206, 'I joined East Club today.',
                                context='West Club was discussed yesterday. ')],
        'East Club membership', 'The user joined East Club today.',
        'The user joined East Club today.')
    defective = deepcopy(faithful)
    defective['items'][0]['candidate_body'] = 'The user joined East Club and West Club today.'
    pair('context-not-evidence', faithful, defective, '/candidate_body',
         'Interpretation-only preceding context cannot establish a separate West Club membership or promote discussion to joining.')

    faithful = _payload([_source('procedure-a', 207,
        'First run latch inspect; then run latch verify. Never reset the latch.', role='assistant')],
        'Latch inspection procedure', 'The assistant supplied inspection and verification steps without a reset.',
        'Inspect the latch and then verify it; resetting is prohibited.')
    faithful['procedure_items'] = [{'index': 0, 'cited_source_ids': ['procedure-a'], 'candidate': {
        'name': 'Latch check', 'description': None,
        'steps': [{'order': 1, 'action': 'Run latch inspect', 'tool': 'latch'},
                  {'order': 2, 'action': 'Run latch verify', 'tool': 'latch'}],
        'triggers': [], 'entities_involved': ['latch']}}]
    defective = deepcopy(faithful)
    defective['procedure_items'][0]['candidate']['steps'][1]['action'] = 'Reset the latch'
    pair('procedure-prohibition', faithful, defective, '/candidate/steps/1/action',
         'The source instructs verification and explicitly prohibits the substituted reset.', kind='procedure')
    return cases
