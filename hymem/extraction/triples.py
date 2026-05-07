from __future__ import annotations

import json
from dataclasses import dataclass

from hymem.extraction.llm import LLMClient, LLMRequest
from hymem.extraction.prompts import (
    ALLOWED_PREDICATES,
    TRIPLE_SYSTEM,
    TRIPLE_USER_TEMPLATE,
)


@dataclass(frozen=True)
class Triple:
    subject: str
    predicate: str
    object: str
    polarity: int  # +1 or -1


def extract_triples(client: LLMClient, text: str) -> list[Triple]:
    """Run the locked-vocabulary triple prompt and validate the output.

    Anything malformed or off-vocabulary is silently dropped — the LLM is allowed
    to be wrong, but we never propagate garbage into the graph.
    """
    request = LLMRequest(
        system=TRIPLE_SYSTEM,
        user=TRIPLE_USER_TEMPLATE.format(text=text),
        response_format="json",
    )
    raw = client.complete(request)
    return _parse(raw)


def _parse(raw: str) -> list[Triple]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []

    triples: list[Triple] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        polarity = item.get("polarity", 1)
        if not (isinstance(subject, str) and isinstance(predicate, str) and isinstance(obj, str)):
            continue
        if predicate not in ALLOWED_PREDICATES:
            continue
        if polarity not in (1, -1):
            continue
        if not subject.strip() or not obj.strip():
            continue
        triples.append(
            Triple(
                subject=subject.strip(),
                predicate=predicate,
                object=obj.strip(),
                polarity=polarity,
            )
        )
    return triples
