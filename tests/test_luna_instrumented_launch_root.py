"""Independent prior launch controls adapted only for version and accepted policy."""
from pathlib import Path

_fixture = Path(__file__).resolve().parents[1] / "tools/diagnostics/tests/test_luna_lme_diagnostic_launch_v1.py"
_source = _fixture.read_text().replace("luna_lme_diagnostic_launch_v1", "luna_lme_diagnostic_launch_v8")
_source = _source.replace("luna_lme_diagnostic_v1.py", "luna_lme_diagnostic_v8.py")
_source = _source.replace("TasksMax=128", "TasksMax=256")
_source = _source.replace("PINS={", 'BILLING_POLICY="included_allowance_or_existing_finite_positive_credits_per_window_v2",\n        NOTIFICATION_POLICY="agent_message_delta_optout_v1",\n        PINS={')
exec(compile(_source, str(_fixture), "exec"), globals())
