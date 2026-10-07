"""Offline source/scope controls for the isolated SIWC prompt repair.

These tests check prompt text and deterministic source transformations. They do
not establish model adherence or a successful benchmark run.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from tools.diagnostics import siwc_extraction_prompt_repair_v1 as repair


FROZEN_PROMPT = Path(
    "/private/tmp/hymem-siwc-pilot-source-fUk67B2v/bundle/candidate/"
    "hymem/extraction/prompts/__init__.py"
)


@pytest.fixture(scope="module")
def source_pair() -> tuple[bytes, bytes]:
    source = FROZEN_PROMPT.read_bytes()
    return source, repair.transform_prompt_source(source)


def _assignment(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return node
    raise AssertionError(f"missing assignment: {name}")


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def _render_combined(source: bytes) -> tuple[str, str, str]:
    tree = ast.parse(source.decode("utf-8"))
    names = (
        "ALLOWED_PREDICATES",
        "PREDICATE_GROUNDING_VERSION",
        "_CHUNK_EXTRACTION_SYSTEM_TEMPLATE",
        "_CHUNK_OMISSION_VERIFICATION_SUFFIX",
        "_CHUNK_EMPTY_VERIFICATION_SUFFIX",
    )
    builders = (
        "build_chunk_extraction_system",
        "build_chunk_empty_verification_system",
        "build_chunk_omission_verification_system",
    )
    selected = [_assignment(tree, name) for name in names]
    selected += [_function(tree, name) for name in builders]
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<frozen-prompt>", "exec"), namespace)
    return tuple(namespace[name]() for name in builders)  # type: ignore[misc,return-value]


def test_exact_frozen_source_and_revised_digest(source_pair: tuple[bytes, bytes]) -> None:
    original, revised = source_pair
    assert hashlib.sha256(original).hexdigest() == repair.SOURCE_SHA256
    assert hashlib.sha256(revised).hexdigest() == repair.RESULT_SHA256
    assert original != revised
    assert repair.transform_prompt_source(original) == revised


def test_exact_scope_and_untouched_builders(source_pair: tuple[bytes, bytes]) -> None:
    original, revised = (item.decode("utf-8") for item in source_pair)
    original_tree, revised_tree = ast.parse(original), ast.parse(revised)
    original_template = _assignment(original_tree, "_CHUNK_EXTRACTION_SYSTEM_TEMPLATE")
    revised_template = _assignment(revised_tree, "_CHUNK_EXTRACTION_SYSTEM_TEMPLATE")
    assert ast.dump(original_template) != ast.dump(revised_template)

    # Removing only the template assignment must leave byte-identical source.
    original_lines, revised_lines = original.splitlines(keepends=True), revised.splitlines(keepends=True)
    del original_lines[original_template.lineno - 1 : original_template.end_lineno]
    del revised_lines[revised_template.lineno - 1 : revised_template.end_lineno]
    assert original_lines == revised_lines

    for name in (
        "build_triple_system",
        "build_chunk_extraction_system",
        "build_chunk_empty_verification_system",
        "build_chunk_omission_verification_system",
    ):
        assert ast.dump(_function(original_tree, name)) == ast.dump(_function(revised_tree, name))
    for name in (
        "_TRIPLE_SYSTEM_TEMPLATE",
        "MARKER_SYSTEM",
        "PREDICATE_GROUNDING_VERSION",
        "_CHUNK_OMISSION_VERIFICATION_SUFFIX",
        "_CHUNK_EMPTY_VERIFICATION_SUFFIX",
    ):
        assert ast.dump(_assignment(original_tree, name)) == ast.dump(_assignment(revised_tree, name))


def test_all_combined_wrappers_inherit_repair(source_pair: tuple[bytes, bytes]) -> None:
    old_primary, old_empty, old_omission = _render_combined(source_pair[0])
    primary, empty, omission = _render_combined(source_pair[1])
    for old, new in ((old_primary, primary), (old_empty, empty), (old_omission, omission)):
        normalized = " ".join(new.split())
        assert old != new
        assert repair.NEW_IDENTITY_GUIDANCE in new
        assert repair.OLD_IDENTITY_GUIDANCE not in new
        assert repair.NEW_POSSESSION_EXAMPLE in new
        assert repair.OLD_POSSESSION_EXAMPLE not in new
        assert "Driving a vehicle alone does not establish owns" in new
        assert "A team's ownership or responsibility for a service alone does not" in normalized
        assert "including clearly entailed implicit wording" in normalized
    assert empty.startswith(primary)
    assert omission.startswith(primary)
    assert empty[len(primary) :] == old_empty[len(old_primary) :]
    assert omission[len(primary) :] == old_omission[len(old_primary) :]


@pytest.mark.parametrize(
    ("invented_source", "expected_mapping"),
    (
        ("Atta uses HyMem to organize notes", "(atta, uses, hymem)"),
        ("Sara is a member of the Atlas team", "(sara, part_of, atlas_team)"),
        ("The parser module is part of Atlas", "(parser_module, part_of, atlas)"),
        ("The Atlas package contains the parser module", "(atlas_package, contains, parser_module)"),
        ("I own a Ford F-150", "(user, owns, ford_f_150)"),
    ),
)
def test_invented_positive_examples_are_directly_stated(
    source_pair: tuple[bytes, bytes], invented_source: str, expected_mapping: str
) -> None:
    primary = _render_combined(source_pair[1])[0]
    assert f'"{invented_source}"' in primary
    assert expected_mapping in primary


@pytest.mark.parametrize(
    ("invented_source", "unsupported_mapping"),
    (
        ("I drive a rented Ford F-150", "(user, owns, ford_f_150)"),
        ("Sara maintains Atlas as an external contractor", "(sara, part_of, atlas)"),
        ("I'm building Atlas for a client", "(user, part_of, atlas)"),
        ("The platform team owns the auth service", "(platform_team, contains, auth_service)"),
    ),
)
def test_invented_negative_fixtures_are_not_taught_as_mappings(
    source_pair: tuple[bytes, bytes], invented_source: str, unsupported_mapping: str
) -> None:
    primary = _render_combined(source_pair[1])[0]
    normalized = " ".join(primary.split())
    assert f'"{invented_source}"' in primary
    assert f'"{invented_source}" -> {unsupported_mapping}' not in primary
    assert "Working on, building, or maintaining a project alone does not establish" in primary
    assert "Driving a vehicle alone does not establish owns" in primary
    assert "ownership or responsibility for a service alone does not" in normalized


def test_source_pin_rejects_any_mutation(source_pair: tuple[bytes, bytes]) -> None:
    original, revised = source_pair
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        repair.transform_prompt_source(original + b"\n")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        repair.transform_prompt_source(revised)
    with pytest.raises(TypeError, match="bytes"):
        repair.transform_prompt_source(original.decode("utf-8"))  # type: ignore[arg-type]
