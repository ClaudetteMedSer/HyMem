"""Reapply independent installer privacy controls to the fresh pilot-only version."""
from pathlib import Path

_fixture = Path(__file__).with_name("test_luna_delta_installer_root.py")
_source = _fixture.read_text().replace("luna_delta_source_install_v1", "luna_instrumented_source_install_v1")
_source = _source.replace("luna_lme_diagnostic_launch_v7.py", "luna_lme_diagnostic_launch_v8.py")
exec(compile(_source, str(_fixture), "exec"), globals())
