"""Recorder retains malformed provider bytes even when verdict envelopes recover."""
from copy import deepcopy
import hashlib

import pytest

from benchmarks import episode_probe as probe
from hymem.dreaming import digest
from hymem.extraction.jsonio import loads_exact_or_fenced
from tests.test_episode_probe_multicall import _run
from tests.test_episode_probe_targeted_root import _root_backend


@pytest.mark.parametrize('repair,semantic,grammar,calls,failed', [
    (False, 'supported', 'supported', 3, False),
    (True, 'supported', 'supported', 6, False),
    (False, 'uncertain', 'supported', 3, True),
    (True, 'unsupported', 'supported', 5, True),
    (False, 'supported', 'unsupported', 3, True),
])
def test_root_recorded_raw_reply_is_never_replaced_with_repaired_json(
    tmp_path, monkeypatch, repair, semantic, grammar, calls, failed,
):
    backend = _root_backend(monkeypatch, repair=repair, semantic=semantic, grammar=grammar)
    returned = []
    def cut_verdict(system, user):
        raw = backend(system, user)
        if system in {digest._DIGEST_FIDELITY_SYSTEM, digest._DIGEST_FORMAT_ADJUDICATION_SYSTEM}:
            assert raw.endswith(']}')
            raw = raw[:-2]
            assert loads_exact_or_fenced(raw) is None
        returned.append(raw)
        return raw
    (row,), client = _run(tmp_path, cut_verdict)
    assert row['digest_failed'] is failed and row['calls'] == calls
    assert [r['reply'] for r in row['completion_records']] == returned
    assert row['completion_records'] == client.sent
    for record in row['completion_records']:
        assert record['reply_sha256'] == hashlib.sha256(record['reply'].encode()).hexdigest()
        assert record['reply_chars'] == len(record['reply'])
    before = deepcopy(row)
    stats = probe.summarize([row], [], faithfulness=None)
    assert row == before
    assert stats['attempted_session_digests'] == 1
    assert stats['digest_failure_rate'] == int(failed)
    assert stats['calls'] == calls
    if failed:
        assert row['failure_reply_chars'] == len(returned[-1])
        assert row['failure_reply_head'] == returned[-1][:240]
