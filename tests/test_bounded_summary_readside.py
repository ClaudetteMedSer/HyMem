"""Published-policy labels are read-side metadata, never new evidence."""
import sqlite3

import pytest

from hymem.dreaming.digest import digest_config_version
from hymem.dreaming.summary import effective_session_summary
from hymem.dreaming.summary_policy import BOUNDED_HIGHLIGHTS_V1, LEGACY_COMPLETE_V1


LABEL = "Automatic highlights (non-exhaustive): "


def generation(policy=BOUNDED_HIGHLIGHTS_V1, *, semantic=False):
    value = digest_config_version(
        prompt_version="v20", episode_prompt_version=None,
        max_chars=12000, max_tokens=3072, max_episodes=None,
        summary_policy=policy,
    )
    if semantic:
        value += "|semantic=sha256:" + "1" * 64
    return value + "|walk=" + "2" * 32


def row(**values):
    # Exercise the real sqlite Row API, including historical three-column
    # callers, rather than depending on a dict-only accessor.
    with sqlite3.connect(":memory:") as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT " + ", ".join(f"? AS {key}" for key in values),
            tuple(values.values()),
        ).fetchone()


@pytest.mark.parametrize("semantic", [False, True])
def test_only_recognized_published_bounded_automatic_content_is_qualified(semantic):
    source = row(summary="Checks passed.", auto_summary="Checks passed.",
                 summary_source="auto", digest_published_generation=generation(semantic=semantic))
    assert effective_session_summary(source) == LABEL + "Checks passed."
    assert source["auto_summary"] == source["summary"] == "Checks passed."


@pytest.mark.parametrize("published", [
    None, "", 1, "unknown", "summary-policy=bounded_highlights_v1",
    "prefix|summary-policy=bounded_highlights_v1|suffix",
    generation().replace("bounded_highlights_v1", "bounded_highlights_v2"),
    generation() + "junk", generation() + "\n",
    generation().replace("|walk=", "|summary-policy=bounded_highlights_v1|walk="),
    generation().replace("|walk=", "|semantic=sha256:BAD|walk="),
    generation(LEGACY_COMPLETE_V1),
])
def test_unknown_malformed_or_legacy_published_generation_keeps_old_bytes(published):
    source = row(summary="Exact old summary. ", auto_summary="Exact old summary. ",
                 summary_source="auto", digest_published_generation=published,
                 digest_cursor_prompt_version=generation())
    assert effective_session_summary(source) == "Exact old summary. "


@pytest.mark.parametrize("source", ["operator", "legacy", None])
def test_mixed_curated_and_automatic_content_qualifies_only_auto(source):
    value = row(summary="Exact curated wording.  ", auto_summary="New checks passed.",
                summary_source=source, digest_published_generation=generation())
    assert effective_session_summary(value) == (
        "Operator/legacy summary: Exact curated wording.  \n\n"
        + LABEL + "New checks passed."
    )


@pytest.mark.parametrize("source", ["operator", "legacy", None])
@pytest.mark.parametrize("auto", [None, "", "Exact curated wording.  "])
def test_curated_or_legacy_only_output_is_not_relabelled(source, auto):
    value = row(summary="Exact curated wording.  ", auto_summary=auto,
                summary_source=source, digest_published_generation=generation())
    assert effective_session_summary(value) == "Exact curated wording.  "


@pytest.mark.parametrize("source", ["auto", "operator", "legacy", None])
def test_actual_auto_only_content_uses_published_policy(source):
    value = row(summary=None, auto_summary="Checks passed.", summary_source=source,
                digest_published_generation=generation())
    assert effective_session_summary(value) == LABEL + "Checks passed."


@pytest.mark.parametrize("source", ["auto", "operator", "legacy", None])
def test_legacy_three_column_callers_keep_existing_bytes(source):
    value = row(summary="Old summary.", auto_summary="New automatic summary.", summary_source=source)
    expected = ("New automatic summary." if source == "auto" else
                "Operator/legacy summary: Old summary.\n\nAutomatic rolling summary: New automatic summary.")
    assert effective_session_summary(value) == expected


@pytest.mark.parametrize("published,staged,expected", [
    (generation(LEGACY_COMPLETE_V1), generation(), "Checks passed."),
    (generation(), generation(LEGACY_COMPLETE_V1), LABEL + "Checks passed."),
    (None, generation(), "Checks passed."),
])
def test_staging_cursor_never_qualifies_published_output(published, staged, expected):
    value = row(summary="Checks passed.", auto_summary="Checks passed.", summary_source="auto",
                digest_published_generation=published, digest_cursor_prompt_version=staged)
    assert effective_session_summary(value) == expected


def test_empty_auto_and_absent_row_do_not_gain_a_label():
    assert effective_session_summary(None) == ""
    assert effective_session_summary(row(summary=None, auto_summary="", summary_source="auto",
                                         digest_published_generation=generation())) == ""
    assert effective_session_summary(row(summary="Legacy fallback.", auto_summary="", summary_source="auto",
                                         digest_published_generation=generation())) == "Legacy fallback."


@pytest.fixture
def context_client(hy_with_embed):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from hymem.honcho import app as hsrv
    from tests.test_honcho_server import _NoopScheduler

    hsrv.set_hy(hy_with_embed)
    if hsrv._scheduler is not None:
        hsrv._scheduler.stop()
    hsrv.set_scheduler(_NoopScheduler())
    with TestClient(hsrv.app) as client:
        yield client


def seed_context(hy, *, text="Checks passed.", published=None, source="auto", summary=None):
    hy.open_session("qualified-summary", source_workspace_id="workspace")
    hy.conn.execute(
        "UPDATE sessions SET summary=?, auto_summary=?, summary_source=?, "
        "digest_published_generation=?, digest_cursor_prompt_version=? WHERE id=?",
        (text if summary is None else summary, text, source, published,
         generation(), "qualified-summary"),
    )


def test_honcho_reads_actual_published_policy_not_config_or_private_cursor(context_client, hy_with_embed):
    seed_context(hy_with_embed, published=generation())
    # The runtime remains the default legacy policy; the published value wins.
    assert hy_with_embed.config.digest_summary_policy == LEGACY_COMPLETE_V1
    route = "/v3/workspaces/workspace/sessions/qualified-summary/context"
    result = context_client.get(route).json()
    assert result["summary"]["content"] == LABEL + "Checks passed."
    hy_with_embed.conn.execute(
        "UPDATE sessions SET digest_published_generation=? WHERE id='qualified-summary'",
        (generation(LEGACY_COMPLETE_V1),),
    )
    assert context_client.get(route).json()["summary"]["content"] == "Checks passed."
    assert context_client.get(route + "?summary=false").json()["summary"] is None


@pytest.mark.parametrize("text", ["Checks passed.", "x" * 500, "🌿" * 500])
def test_full_label_counts_toward_context_budget_without_mutating_stored_cap(context_client, hy_with_embed, text):
    from hymem.honcho import app as hsrv

    seed_context(hy_with_embed, text=text, published=generation())
    rendered = LABEL + text
    cost = hsrv.estimate_tokens(f"role:system\ncontent:<summary>{rendered}</summary>") + 4
    route = "/v3/workspaces/workspace/sessions/qualified-summary/context"
    exact = context_client.get(route, params={"tokens": cost})
    tight = context_client.get(route, params={"tokens": cost - 1})
    assert exact.status_code == tight.status_code == 200
    assert exact.json()["summary"]["content"] == rendered
    assert exact.json()["context_token_count"] == cost
    assert exact.json()["summary"]["token_count"] == max(1, hsrv.estimate_tokens(rendered))
    assert tight.json()["summary"] is None
    assert tight.json()["context_token_count"] == 0
    assert tight.json()["context_truncated"] is True
    assert tight.json()["context_omitted_items"] == 1
    assert tuple(hy_with_embed.conn.execute(
        "SELECT summary,auto_summary FROM sessions WHERE id='qualified-summary'",
    ).fetchone()) == (text, text)
