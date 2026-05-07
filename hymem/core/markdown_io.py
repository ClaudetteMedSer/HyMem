from __future__ import annotations

import contextlib
import os
import re
import tempfile
from pathlib import Path

# Sections are delimited by HTML comments so they survive surrounding edits and
# parse unambiguously. Format:
#
#   <!-- HyMem:auto:<section>:start -->
#   ...managed content...
#   <!-- HyMem:auto:<section>:end -->

_START = "<!-- HyMem:auto:{name}:start -->"
_END = "<!-- HyMem:auto:{name}:end -->"


def _pattern(name: str) -> re.Pattern[str]:
    start = re.escape(_START.format(name=name))
    end = re.escape(_END.format(name=name))
    return re.compile(rf"{start}\n?(.*?)\n?{end}", re.DOTALL)


def read_section(path: Path, section: str) -> str | None:
    """Return the content between the section delimiters, or None if absent."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    match = _pattern(section).search(text)
    return match.group(1) if match else None


def write_section(path: Path, section: str, content: str, *, header: str | None = None) -> None:
    """Atomically replace (or insert) a managed section with `content`.

    `header` (e.g. '## Behavioral Profile') is written immediately above a freshly
    inserted section. Existing files keep whatever heading the user already has.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else ""

    block = f"{_START.format(name=section)}\n{content.rstrip()}\n{_END.format(name=section)}"
    pattern = _pattern(section)

    if pattern.search(existing):
        new_text = pattern.sub(lambda _m: block, existing, count=1)
    else:
        prefix = existing
        if prefix and not prefix.endswith("\n"):
            prefix += "\n"
        if header:
            prefix += f"\n{header}\n" if prefix else f"{header}\n"
        elif prefix:
            prefix += "\n"
        new_text = prefix + block + "\n"

    _atomic_write(path, new_text)


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
