"""Compact, durable encoding for embedding vectors stored in TEXT columns.

The `*_embeddings` and `embedding_cache` tables keep a durable copy of every
vector (the source of truth for cold-start vec0 backfill and the no-extension
cosine fallback). Stored as JSON float text a 384-d vector is ~4-6 KB; packed
float32 + base64 is ~2.5x smaller.

`encode_vector` writes the packed form; `decode_vector` reads either the packed
form (recognised by the `b64f32:` prefix) or a legacy JSON array, so no data
migration is required — old rows keep working and re-encode whenever rewritten.

float32 matches the precision already used by the vec0 index (`db._pack_vector`
also packs `f`), so this does not lose precision relative to the search path.
"""

from __future__ import annotations

import base64
import json
import struct

# Sentinel: JSON arrays start with '[', never this prefix, so the two encodings
# are unambiguous on read.
_PREFIX = "b64f32:"


def encode_vector(vec: list[float]) -> str:
    """Pack a float vector as little-endian float32, base64, prefixed."""
    packed = struct.pack(f"<{len(vec)}f", *vec)
    return _PREFIX + base64.b64encode(packed).decode("ascii")


def decode_vector(value: str | bytes) -> list[float]:
    """Decode a stored vector. Accepts the packed form or legacy JSON text."""
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("ascii")
    if value.startswith(_PREFIX):
        raw = base64.b64decode(value[len(_PREFIX) :])
        n = len(raw) // 4
        return list(struct.unpack(f"<{n}f", raw))
    return json.loads(value)
