from __future__ import annotations

from pathlib import Path

from hymem.core.markdown_io import read_section, write_section


def test_section_round_trip(tmp_path: Path):
    f = tmp_path / "MEMORY.md"
    write_section(f, "project_insights", "- foo\n- bar", header="## Project Insights")
    assert read_section(f, "project_insights") == "- foo\n- bar"


def test_section_replaces_in_place_without_disturbing_surroundings(tmp_path: Path):
    f = tmp_path / "MEMORY.md"
    write_section(f, "project_insights", "v1", header="## Project Insights")

    # Simulate a manual edit ABOVE and BELOW the managed block.
    text = f.read_text(encoding="utf-8")
    text = "# My memory file\n\n" + text + "\n## Manually maintained\nhand-written notes\n"
    f.write_text(text, encoding="utf-8")

    write_section(f, "project_insights", "v2")

    final = f.read_text(encoding="utf-8")
    assert "# My memory file" in final
    assert "hand-written notes" in final
    assert "v1" not in final
    assert "v2" in final
    assert read_section(f, "project_insights") == "v2"


def test_missing_section_returns_none(tmp_path: Path):
    f = tmp_path / "MEMORY.md"
    f.write_text("nothing managed here\n", encoding="utf-8")
    assert read_section(f, "project_insights") is None


def test_missing_file_returns_none(tmp_path: Path):
    assert read_section(tmp_path / "absent.md", "x") is None


def test_atomic_write_does_not_leave_tempfiles(tmp_path: Path):
    f = tmp_path / "USER.md"
    write_section(f, "behavioral_profile", "hello", header="## Behavioral Profile")
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith("USER.md.")]
    assert leftovers == []
