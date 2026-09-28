"""Independent policy-selection checks; instructions are not model accuracy."""
from dataclasses import FrozenInstanceError

import pytest

from hymem.dreaming import summary_policy as p
from hymem.extraction.prompts import SESSION_SUMMARY_MAX_CHARS


STAGES = ('generation', 'compaction', 'content_repair', 'verification', 'diagnosis')


def test_default_is_explicit_legacy_and_format_bound_does_not_drift():
    assert p.DEFAULT_SUMMARY_POLICY == 'legacy_complete_v1'
    assert p.validate_summary_policy() == 'legacy_complete_v1'
    assert p.get_summary_policy().topic_selection_allowed is False
    assert p.get_summary_policy(p.BOUNDED_HIGHLIGHTS_V1).topic_selection_allowed is True
    for name in (p.LEGACY_COMPLETE_V1, p.BOUNDED_HIGHLIGHTS_V1):
        policy = p.get_summary_policy(name)
        assert policy.name == name
        assert policy.max_codepoints == SESSION_SUMMARY_MAX_CHARS == 500
        assert policy.sentence_count == 1


@pytest.mark.parametrize('stage', STAGES)
def test_default_delegates_without_an_alternate_prompt(stage):
    assert p.summary_prompt_component(stage=stage) is None
    assert p.summary_prompt_component('legacy_complete_v1', stage=stage) is None
    text = p.summary_prompt_component('bounded_highlights_v1', stage=stage)
    assert isinstance(text, str) and text.strip()
    assert 'bounded_highlights_v1' in text
    assert 'Entire unselected topics may be omitted' in text
    assert 'not semantic verification or publication authorization' in text


@pytest.mark.parametrize('bad', [None, True, False, 1, 0, 1.0, {}, [], (),
                                 '', 'bounded', 'BOUNDED_HIGHLIGHTS_V1',
                                 ' bounded_highlights_v1', 'bounded_highlights_v1 ',
                                 'bounded_highlights_v1\n', 'legacy_complete_v2'])
def test_no_policy_alias_coercion_or_unknown_future_version(bad):
    for resolve in (p.validate_summary_policy, p.get_summary_policy):
        with pytest.raises(ValueError):
            resolve(bad)
    with pytest.raises(ValueError):
        p.summary_prompt_component(bad, stage='generation')


@pytest.mark.parametrize('stage', [None, True, 1, {}, [], '', 'format', 'format_adjudication',
                                  'all', 'Generation', 'verification '])
@pytest.mark.parametrize('policy', ['legacy_complete_v1', 'bounded_highlights_v1'])
def test_unknown_stage_rejected_even_for_legacy_delegation(stage, policy):
    with pytest.raises(ValueError):
        p.summary_prompt_component(policy, stage=stage)


def test_string_subclass_cannot_smuggle_an_alias():
    class PretendString(str):
        pass
    with pytest.raises(ValueError):
        p.validate_summary_policy(PretendString('bounded_highlights_v1'))


def test_returned_descriptors_and_policy_table_are_immutable():
    policy = p.get_summary_policy('bounded_highlights_v1')
    with pytest.raises((AttributeError, FrozenInstanceError)):
        policy.max_codepoints = 100000
    with pytest.raises(TypeError):
        p.SUMMARY_POLICIES['invented'] = policy
    assert p.get_summary_policy('bounded_highlights_v1').max_codepoints == 500


def test_interleaved_policy_selection_never_mutates_the_other_arm():
    expected = {stage: p.summary_prompt_component('bounded_highlights_v1', stage=stage)
                for stage in STAGES}
    for _ in range(3):
        for stage in STAGES:
            assert p.summary_prompt_component('legacy_complete_v1', stage=stage) is None
            assert p.summary_prompt_component('bounded_highlights_v1', stage=stage) == expected[stage]
