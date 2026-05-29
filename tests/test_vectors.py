from __future__ import annotations

import json

import pytest

from hymem.core.vectors import decode_vector, encode_vector


def test_packed_roundtrip():
    vec = [0.1, -2.5, 3.14159, 0.0, -0.0001, 42.0]
    out = decode_vector(encode_vector(vec))
    assert out == pytest.approx(vec, abs=1e-6)


def test_packed_prefix():
    # New encoding is recognisable and not valid JSON, so the two forms never
    # collide on read.
    s = encode_vector([1.0, 2.0])
    assert s.startswith("b64f32:")


def test_legacy_json_still_decodes():
    vec = [0.5, -1.25, 7.0]
    legacy = json.dumps(vec)  # what old rows hold
    assert decode_vector(legacy) == pytest.approx(vec, abs=1e-6)


def test_empty_vector():
    assert decode_vector(encode_vector([])) == []


def test_decode_accepts_bytes():
    s = encode_vector([1.0, 2.0])
    assert decode_vector(s.encode("ascii")) == pytest.approx([1.0, 2.0], abs=1e-6)
