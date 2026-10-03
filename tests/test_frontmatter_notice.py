"""Operational notices must never displace or rewrite skill frontmatter."""

import pytest

from tools.deployment.frontmatter_notice import insert_notice


NOTICE = b"> VERIFIED-NOTICE-v1\n> Current operational guidance.\n\n"


@pytest.mark.parametrize("closing", [b"---", b"..."])
@pytest.mark.parametrize("ending", [b"\n", b"\r\n"])
def test_frontmatter_stays_first_and_byte_identical(closing, ending):
    metadata = ending.join((b"---", b"name: private-skill", b"description: |", b"  unchanged", closing)) + ending
    body = b"\n# Original\n\x80 unmodified bytes\n"
    notice = NOTICE.replace(b"\n", ending)
    original = metadata + body

    result = insert_notice(original, notice, marker=b"VERIFIED-NOTICE-v1")

    assert result == metadata + notice + body
    assert result.replace(notice, b"", 1) == original


@pytest.mark.parametrize("original", [b"", b"# Plain Markdown", b"# Plain Markdown\n", b" ---\nnot leading YAML\n"])
def test_plain_markdown_is_prefixed_without_changing_original(original):
    assert insert_notice(original, NOTICE) == NOTICE + original


def test_only_exact_standalone_delimiter_ends_frontmatter():
    metadata = b'---\ninline: "---"\nblock: |\n  ---\n  ...\n--- # comment\n...\n'
    original = metadata + b"body"
    assert insert_notice(original, NOTICE) == metadata + NOTICE + b"body"


def test_utf8_bom_and_frontmatter_remain_at_document_start():
    metadata = b"\xef\xbb\xbf---\nname: example\n---\n"
    assert insert_notice(metadata + b"body", NOTICE) == metadata + NOTICE + b"body"


def test_utf8_bom_of_plain_markdown_remains_first():
    assert insert_notice(b"\xef\xbb\xbfbody", NOTICE) == b"\xef\xbb\xbf" + NOTICE + b"body"


@pytest.mark.parametrize("original", [b"---\n", b"---\nname: example\n", b"---\r\nname: example\r\n  ---\r\n"])
def test_unterminated_frontmatter_fails_closed(original):
    with pytest.raises(ValueError, match="unterminated"):
        insert_notice(original, NOTICE)


def test_eof_closing_delimiter_uses_supplied_separator_without_modifying_original():
    original = b"---\r\nname: example\r\n..."
    notice = b"\r\n" + NOTICE.replace(b"\n", b"\r\n")
    result = insert_notice(original, notice)
    assert result == original + notice
    assert result.replace(notice, b"", 1) == original
    with pytest.raises(ValueError, match="start with a line ending"):
        insert_notice(original, NOTICE)


def test_exact_duplicate_and_same_marker_changed_notice_fail_closed():
    original = insert_notice(b"---\nname: example\n---\nbody", NOTICE)
    with pytest.raises(ValueError, match="notice already exists"):
        insert_notice(original, NOTICE)
    changed_notice = NOTICE.replace(b"Current", b"Revised")
    with pytest.raises(ValueError, match="marker already exists"):
        insert_notice(original, changed_notice, marker=b"VERIFIED-NOTICE-v1")


def test_notice_and_marker_validation():
    for notice in (b"", b"notice", b"notice\n"):
        with pytest.raises(ValueError, match="blank line"):
            insert_notice(b"body", notice)
    for marker in (b"", b"not-in-notice"):
        with pytest.raises(ValueError, match="occur in notice"):
            insert_notice(b"body", NOTICE, marker=marker)
    with pytest.raises(TypeError, match="must be bytes"):
        insert_notice("body", NOTICE)
    with pytest.raises(TypeError, match="marker must be bytes"):
        insert_notice(b"body", NOTICE, marker="marker")


def test_non_utf8_bom_is_explicitly_rejected():
    with pytest.raises(ValueError, match="only UTF-8"):
        insert_notice(b"\xff\xfe-\x00-\x00-\x00\n\x00", NOTICE)
