"""Build a Markdown notice update without rewriting existing document bytes.

This is a byte-splice helper, not a YAML parser or a file publisher. Callers
remain responsible for pinning the original file and publishing atomically.
"""

from __future__ import annotations


_UTF8_BOM = b"\xef\xbb\xbf"


def insert_notice(
    original: bytes, notice: bytes, *, marker: bytes | None = None
) -> bytes:
    """Insert a complete notice after leading YAML frontmatter, if present.

    ``notice`` is inserted verbatim and must end in a blank line. Consequently,
    removing the inserted notice recovers ``original`` exactly. A closing YAML
    delimiter at EOF additionally requires the notice to start with the opening
    delimiter's line ending, so the delimiter remains a standalone line.

    UTF-8 BOMs and all existing line endings are preserved. Only exact ``---``
    or ``...`` lines terminate frontmatter; indented or inline spellings do not.
    An unterminated leading block fails closed. A duplicate notice, or an
    existing optional unique ``marker``, also raises ``ValueError`` rather than
    silently adding another notice. A marker must occur in the supplied notice.
    """
    if not isinstance(original, bytes) or not isinstance(notice, bytes):
        raise TypeError("original and notice must be bytes")
    if not notice or not notice.endswith((b"\n\n", b"\r\n\r\n")):
        raise ValueError("notice must be nonempty and end in a blank line")
    if marker is not None:
        if not isinstance(marker, bytes):
            raise TypeError("marker must be bytes")
        if not marker or marker not in notice:
            raise ValueError("marker must be nonempty and occur in notice")
        if marker in original:
            raise ValueError("notice marker already exists")
    if notice in original:
        raise ValueError("notice already exists")

    if original.startswith((b"\xff\xfe", b"\xfe\xff", b"\x00\x00\xfe\xff")):
        raise ValueError("only UTF-8 documents are supported")
    start = len(_UTF8_BOM) if original.startswith(_UTF8_BOM) else 0
    if original.startswith(b"---\r\n", start):
        line_ending = b"\r\n"
    elif original.startswith(b"---\n", start):
        line_ending = b"\n"
    else:
        return original[:start] + notice + original[start:]

    position = start + 3 + len(line_ending)
    while position < len(original):
        newline = original.find(b"\n", position)
        end = len(original) if newline == -1 else newline + 1
        line = original[position:end]
        if line.endswith(b"\r\n"):
            delimiter = line[:-2]
        elif line.endswith(b"\n"):
            delimiter = line[:-1]
        else:
            delimiter = line
        if delimiter in (b"---", b"..."):
            if newline == -1 and not notice.startswith(line_ending):
                raise ValueError("notice must start with a line ending after EOF delimiter")
            return original[:end] + notice + original[end:]
        position = end
    raise ValueError("leading YAML frontmatter is unterminated")
