"""Independent public-health checks for summary/index separation."""
from types import SimpleNamespace

import pytest

from hymem.dreaming.status import summary_health_projection_is_valid
from hymem.dreaming.summary import effective_session_summary
from tests.test_honcho_server import client, _open_external_session, _log_external


def projection(**changes):
    value = dict(summary_degraded_sessions=0, summary_missing_sessions=0,
                 malformed_summaries=0, summary_healthy=True)
    value.update(changes)
    return value


@pytest.mark.parametrize('changes,valid', [
    ({}, True),
    ({'summary_degraded_sessions': 2, 'summary_missing_sessions': 1, 'summary_healthy': False}, True),
    ({'malformed_summaries': 1, 'summary_healthy': False}, True),
    ({'summary_missing_sessions': 1}, False),
    ({'summary_healthy': False}, False),
    ({'summary_healthy': 1}, False),
    ({'summary_degraded_sessions': True}, False),
    ({'summary_degraded_sessions': -1}, False),
    ({'summary_degraded_sessions': 2**31, 'summary_healthy': False}, False),
    ({'malformed_summaries': False}, False),
    ({'malformed_summaries': '0'}, False),
    ({'summary_degraded_sessions': 1, 'summary_healthy': True}, False),
])
def test_summary_projection_requires_exact_types_and_cross_field_consistency(changes, valid):
    assert summary_health_projection_is_valid(projection(**changes)) is valid


@pytest.mark.parametrize('name', list(projection()))
def test_summary_projection_missing_field_is_not_a_healthy_zero(name):
    state = projection()
    del state[name]
    assert not summary_health_projection_is_valid(state)


def health(**changes):
    state = dict(summary_healthy=False, degraded=True, missing=False, malformed=False)
    state.update(changes)
    return state


@pytest.mark.parametrize('source', ['operator', 'legacy', 'auto'])
def test_stale_context_labels_without_modifying_prior_text(source):
    text = 'Accepted historical text with exact spelling, numbers 123 and punctuation.'
    row = {'summary': text, 'auto_summary': text if source == 'auto' else '', 'summary_source': source}
    saved = dict(row)
    result = effective_session_summary(row, summary_health=health())
    assert result.startswith('[Automatic summary is stale or missing;')
    assert text in result and row == saved


def test_unbounded_legacy_history_is_preserved_without_clipping():
    text = 'sole surviving historical context ' * 80
    row = dict(summary=text, auto_summary='', summary_source='legacy')
    assert text in effective_session_summary(row, summary_health=health())


def test_malformed_automatic_context_is_withheld_without_losing_operator_text():
    row = dict(summary='Curated exact operator context.', auto_summary='Unverified automatic context.',
               summary_source='operator')
    result = effective_session_summary(row, summary_health=health(malformed=True, degraded=False))
    assert 'state is unverified' in result
    assert row['summary'] in result and row['auto_summary'] not in result
    assert row['auto_summary'] == 'Unverified automatic context.'


def test_malformed_auto_alias_is_not_reexposed_as_fallback():
    row = dict(summary='Unverified content.', auto_summary='Unverified content.', summary_source='auto')
    result = effective_session_summary(row, summary_health=health(malformed=True, degraded=False))
    assert 'withheld' in result and 'Unverified content.' not in result


def test_invalid_health_shape_cannot_render_unqualified_context():
    with pytest.raises(ValueError):
        effective_session_summary(dict(summary='History.', auto_summary='', summary_source='legacy'),
                                  summary_health={})


@pytest.mark.parametrize('state', [
    health(summary_healthy=True), health(malformed=True),
    health(degraded=False, missing=True), health(degraded=False),
])
def test_inconsistent_health_cannot_render_context(state):
    with pytest.raises(ValueError):
        effective_session_summary(dict(summary='History.', auto_summary='', summary_source='legacy'),
                                  summary_health=state)


def test_mcp_degraded_completion_is_not_a_clean_or_incomplete_claim():
    from hymem import server
    from hymem.dreaming.runner import DreamReport
    from tests.test_mcp_server import _clean_dream_status
    state = _clean_dream_status(**projection(summary_degraded_sessions=1, summary_healthy=False))
    hy = SimpleNamespace(dream_status=lambda: state)
    result = server._format_dream_completion(hy, DreamReport(), targeted=False)
    assert 'item indexing complete with summary degradation' in result
    assert 'finished cleanly' not in result and 'dreaming incomplete' not in result
    assert 'actual coverage frontier' in result


@pytest.mark.parametrize('mutations', [
    {'summary_healthy': True}, {'summary_missing_sessions': 2},
    {'malformed_summaries': 1}, {'pending_digests': 1}, {'coverage_integrity_failures': 1},
    {'summary_missing_context': 1},
])
def test_mcp_summary_degradation_does_not_hide_other_failures(mutations):
    from hymem import server
    from hymem.dreaming.runner import DreamReport
    from tests.test_mcp_server import _clean_dream_status
    state = _clean_dream_status(**projection(summary_degraded_sessions=1, summary_healthy=False),
                               **{k: v for k, v in mutations.items() if k not in projection()})
    state.update(mutations)
    result = server._format_dream_completion(SimpleNamespace(dream_status=lambda: state), DreamReport(), targeted=True)
    assert 'item indexing complete with summary degradation' not in result
    assert 'finished cleanly' not in result


def test_honcho_summary_text_and_health_share_a_snapshot(client, hy_with_embed, monkeypatch):
    from hymem.core import db
    from hymem.dreaming.lossless import materialize_message_coverage
    from hymem.dreaming import summary_state
    from tests.test_summary_frontier_v62 import _generation
    hy = hy_with_embed
    sid = 'summary-publication-race'
    _open_external_session(hy, sid)
    last = _log_external(hy, sid, 'user', 'The exact historical source.')
    generation = _generation()
    with db.transaction(hy.conn):
        materialize_message_coverage(hy.conn, sid)
        hy.conn.execute('UPDATE sessions SET digest_published_generation=?,digest_published_message_id=? WHERE id=?',
                        (generation, last, sid))
        summary_state.mark_summary_current(hy.conn, sid, 'Old accepted summary.',
                                           generation=generation, covered_message_id=last)
        summary_state.record_summary_failure(hy.conn, sid, 'summary_output_cap')
    classify = summary_state.classify_summary_state
    published = []

    def publish_between_read_and_health(conn, session_id, **kwargs):
        if not published:
            writer = db.connect(hy.config.db_path)
            try:
                with db.transaction(writer):
                    summary_state.mark_summary_current(writer, sid, 'New accepted summary.',
                                                       generation=generation, covered_message_id=last)
                published.append(True)
            finally:
                writer.close()
        return classify(conn, session_id, **kwargs)

    monkeypatch.setattr(summary_state, 'classify_summary_state', publish_between_read_and_health)
    response = client.get(f'/v3/workspaces/hermes/sessions/{sid}/context?tokens=1000')
    assert response.status_code == 200
    body = response.json()
    assert body['summary_health'] == health()
    assert 'Old accepted summary.' in str(body['summary'])
    assert 'New accepted summary.' not in str(body['summary'])
    second = client.get(f'/v3/workspaces/hermes/sessions/{sid}/context?tokens=1000').json()
    assert second['summary_health']['summary_healthy']
    assert 'New accepted summary.' in str(second['summary'])


@pytest.mark.parametrize('mode,expected', [('healthy', 'OK'), ('degraded', 'WARN'), ('malformed', 'FAIL')])
def test_doctor_summary_health_is_read_only_and_truthful(hy, mode, expected):
    from hymem import doctor
    from hymem.core import db
    from hymem.dreaming.summary_state import record_summary_failure
    from tests.test_summary_frontier_v62 import _published
    _published(hy.conn)
    if mode == 'degraded':
        with db.transaction(hy.conn):
            record_summary_failure(hy.conn, 'x', 'summary_output_cap')
    elif mode == 'malformed':
        hy.conn.execute("UPDATE sessions SET auto_summary_message_id=999999 WHERE id='x'")
    before = list(hy.conn.iterdump())
    result = doctor._check_summary_health(SimpleNamespace(root=hy.config.root))
    assert result.status == expected, result.detail
    assert list(hy.conn.iterdump()) == before


def test_doctor_missing_store_is_not_created(tmp_path):
    from hymem import doctor
    result = doctor._check_summary_health(SimpleNamespace(root=tmp_path))
    assert result.status == doctor.FAIL
    assert not list(tmp_path.iterdir())
