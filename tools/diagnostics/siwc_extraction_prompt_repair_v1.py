"""Pure, source-pinned repair of the frozen combined extraction prompt.

This module creates revised source bytes only. Candidate assembly and any
runtime use are separate, independently reviewed steps.
"""

from __future__ import annotations

import hashlib


SOURCE_SHA256 = "17ee5017c54a1ba255e0220aa8e246766127cb71ced65fb2e8c812034f6e184c"
RESULT_SHA256 = "ec50ad403d9b678d3cfc62a55e4f5391b00781140d42a28802a864b3c1905048"

OLD_IDENTITY_GUIDANCE = '''- When a chunk names a person or team alongside a project, codebase, or artifact
  they own, work on, or belong to, extract the linking edge explicitly. This is
  high-priority: identity-to-artifact links are the most underrepresented and
  most useful triples in the graph. Strong examples (extract eagerly when the
  chunk supports them):
    "Atta is working on MedFlow"                   -> (atta, part_of, medflow)
    "I'm building HyMem"                            -> (atta, part_of, hymem)
    "We use HyMem for the memory layer"             -> (atta, uses, hymem)
    "The platform team owns the auth service"       -> (platform_team, contains, auth_service)
    "Sara maintains the ingest pipeline"            -> (sara, part_of, ingest_pipeline)
  When the speaker is the user themselves ("I'm working on X", "we shipped Y"),
  resolve the implicit subject to the user's canonical name when known from
  context; otherwise use a first-person handle and let canonicalization resolve
  it. Do NOT skip these just because the speaker is implicit.
  This makes identity-to-artifact relationships queryable as 1-hop graph edges
  rather than fuzzy text matches across sibling canonicals.'''

NEW_IDENTITY_GUIDANCE = '''- Extract a person, team, project, or component link when the cited source
  record supports that specific predicate. Examples with direct support:
    "Atta uses HyMem to organize notes"             -> (atta, uses, hymem)
    "Sara is a member of the Atlas team"             -> (sara, part_of, atlas_team)
    "The parser module is part of Atlas"            -> (parser_module, part_of, atlas)
    "The Atlas package contains the parser module"  -> (atlas_package, contains, parser_module)
  Working on, building, or maintaining a project alone does not establish
  part_of: the source must establish a membership or component relation,
  including clearly entailed implicit wording. A team's ownership or
  responsibility for a service alone does not establish contains: the source
  must establish that A includes B as a subcomponent, including clearly
  entailed implicit wording.
  Do not infer either component predicate from mere collaboration or stewardship.
  Counterexamples: "Sara maintains Atlas as an external contractor" and
  "I'm building Atlas for a client" do not establish part_of;
  "The platform team owns the auth service" does not establish contains.
  When the speaker is the user themselves ("I'm using X", "we use Y"),
  resolve the implicit subject to the user's canonical name when known from
  context; otherwise use a first-person handle and let canonicalization resolve
  it. Do NOT skip these just because the speaker is implicit.'''

OLD_POSSESSION_EXAMPLE = '    "I drive a Ford F-150"             -> (user, owns, ford_f_150)'
NEW_POSSESSION_EXAMPLE = '    "I own a Ford F-150"               -> (user, owns, ford_f_150)'
OLD_POSSESSION_NOTE_ANCHOR = '  Updates to these (a new car, a move, a changed metric) are exactly the'
NEW_POSSESSION_NOTE_ANCHOR = '''  Driving a vehicle alone does not establish owns; a driver may not possess it.
  For example, "I drive a rented Ford F-150" does not establish owns.
  Updates to these (a new car, a move, a changed metric) are exactly the'''


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError("Frozen combined prompt anchor missing or ambiguous")
    return source.replace(old, new, 1)


def transform_prompt_source(source: bytes) -> bytes:
    """Return revised module bytes only for the exact frozen prompt source."""
    if not isinstance(source, bytes):
        raise TypeError("source must be bytes")
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Frozen prompt source SHA256 mismatch")

    original = source.decode("utf-8", errors="strict")
    # Both anchors also occur in the standalone prompt. Limit each replacement
    # to the combined template, leaving standalone extraction unchanged.
    start_marker = '_CHUNK_EXTRACTION_SYSTEM_TEMPLATE = """'
    end_marker = '\n\ndef build_chunk_extraction_system() -> str:'
    if original.count(start_marker) != 1 or original.count(end_marker) != 1:
        raise ValueError("Frozen combined template boundaries changed")
    start = original.index(start_marker)
    end = original.index(end_marker, start)
    combined = original[start:end]
    combined = _replace_once(combined, OLD_IDENTITY_GUIDANCE, NEW_IDENTITY_GUIDANCE)
    combined = _replace_once(combined, OLD_POSSESSION_EXAMPLE, NEW_POSSESSION_EXAMPLE)
    combined = _replace_once(combined, OLD_POSSESSION_NOTE_ANCHOR, NEW_POSSESSION_NOTE_ANCHOR)
    revised = (original[:start] + combined + original[end:]).encode("utf-8")
    if hashlib.sha256(revised).hexdigest() != RESULT_SHA256:
        raise AssertionError("Revised prompt source SHA256 mismatch")
    return revised
