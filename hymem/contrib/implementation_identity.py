"""Cycle-free import-time source commitments for maintained transports.

The helper intentionally has no dependency on provider/endpoint modules.  An
owning module calls it while that module is being imported, so the source
commitment and the live code objects cross the same lifecycle boundary.  A
later identity lookup must reuse the frozen value rather than reread files a
rolling deployment may already have replaced.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def import_time_source_sha256(path: str) -> str:
    source = Path(path).read_text(encoding="utf-8").replace(
        "\r\n", "\n"
    ).replace("\r", "\n")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            node.body = body[1:]
    ast.fix_missing_locations(tree)
    payload = ast.dump(
        tree, annotate_fields=True, include_attributes=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def compose_import_time_sha256(*commitments: str) -> str:
    payload = json.dumps(
        list(commitments), ensure_ascii=True, separators=(",", ":"),
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()
