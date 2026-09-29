from __future__ import annotations

import pytest

from hymem.extraction import contract, prompts


@pytest.mark.parametrize(
    "builder",
    (
        prompts.build_chunk_extraction_system,
        prompts.build_chunk_empty_verification_system,
        prompts.build_chunk_omission_verification_system,
    ),
    ids=("primary", "empty-verification", "omission-verification"),
)
def test_combined_passes_share_predicate_grounding_rule(builder):
    rendered = " ".join(builder().split())
    rule = rendered.split("- Predicate grounding (", 1)[1].split(
        "- polarity is -1", 1
    )[0]

    assert rule.startswith(prompts.PREDICATE_GROUNDING_VERSION + "):")
    assert "support each predicate independently from the cited source record" in rule
    assert "A preference does not establish use, ownership, or deployment" in rule
    assert "use does not establish preference" in rule
    assert "Intent, recommendations, and hypotheses do not establish actual adoption" in rule
    assert "Preserve both predicates when each is supported" in rule
    assert "implicit language that clearly entails the relationship" in rule
    assert "Preserve exact positive and negative claims from their respective sources" in rule


def test_grounding_version_changes_rendered_contract_without_user_schema(monkeypatch):
    baseline = contract.extraction_contract_identity()
    primary = prompts.build_chunk_extraction_system()
    assert '"triples", "markers", and "complete"' in prompts.CHUNK_EXTRACTION_USER_TEMPLATE
    assert '"triples" and "markers", plus' in prompts.CHUNK_OMISSION_VERIFICATION_USER_TEMPLATE

    monkeypatch.setattr(
        prompts, "PREDICATE_GROUNDING_VERSION", "hymem-predicate-grounding-test-change"
    )

    assert prompts.build_chunk_extraction_system() != primary
    assert contract.extraction_contract_identity() != baseline
