"""Parent's independent recovery boundary controls."""
import pytest

from hymem.core import db
from hymem.dreaming.summary_state import classify_summary_state
from hymem.session import append_message
from tests.test_summary_recovery_v63 import Replies, _seed, _items, _public, _no_lease


def test_public_api_repairs_only_summary_under_its_own_budget(hy):
    _seed(hy.conn)
    before = _items(hy.conn)
    llm = Replies({'summary': 'Both exact source messages are retained.'})
    hy.set_llm(llm)
    result = hy.recover_summaries(max_calls=1, session_id='x')
    assert result['calls'] == result['published'] == 1
    assert result['remaining'] == 0
    assert _items(hy.conn) == before
    assert classify_summary_state(hy.conn, 'x')['summary_healthy']
    _no_lease(hy.conn)


def test_new_raw_input_during_completion_cannot_be_claimed_as_summarized(hy):
    _, old_target, _ = _seed(hy.conn)
    before = _items(hy.conn)
    def append_while_waiting(request):
        append_message(hy.conn, 'x', 'user', 'A NEW unrelated claim after the captured input.')
        return {'summary': 'Only the originally supplied messages are summarized.'}
    hy.set_llm(Replies(append_while_waiting))
    result = hy.recover_summaries(max_calls=1)
    assert result['published'] == 1 and result['remaining'] == 1
    row = hy.conn.execute("SELECT auto_summary_message_id,digest_published_message_id FROM sessions WHERE id='x'").fetchone()
    assert tuple(row) == (old_target, old_target)
    assert classify_summary_state(hy.conn, 'x')['degraded']
    assert _items(hy.conn) == before
    _no_lease(hy.conn)


def test_operator_text_change_in_flight_rejects_old_publication(hy):
    _seed(hy.conn)
    before = _items(hy.conn)
    def update_operator(request):
        hy.conn.execute("UPDATE sessions SET summary='Operator changed this during recovery.',summary_source='operator' WHERE id='x'")
        return {'summary': 'No right to overwrite the changed base.'}
    hy.set_llm(Replies(update_operator))
    with pytest.raises(RuntimeError, match='public target changed') as caught:
        hy.recover_summaries()
    assert caught.value.summary_recovery_report['calls'] == 1
    row = hy.conn.execute("SELECT summary,auto_summary FROM sessions WHERE id='x'").fetchone()
    assert tuple(row) == ('Operator changed this during recovery.', 'Prior accepted summary.')
    assert _items(hy.conn) == before
    _no_lease(hy.conn)


def test_current_summary_does_not_spend_provider_calls(hy):
    from tests.test_summary_frontier_v62 import _published
    _published(hy.conn)
    llm = Replies(AssertionError('provider must not run'))
    hy.set_llm(llm)
    result = hy.recover_summaries()
    assert result['calls'] == result['provider_attempts'] == result['remaining'] == 0
    assert llm.calls == []
    _no_lease(hy.conn)


def test_api_missing_llm_fails_before_creating_store(cfg):
    from hymem import HyMem
    hy = HyMem(cfg)
    try:
        with pytest.raises(RuntimeError, match='requires an LLMClient'):
            hy.recover_summaries()
        assert not cfg.db_path.exists()
    finally:
        hy.close()
