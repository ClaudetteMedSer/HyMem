"""Offline, finite pilot-only installer controls; no host or model access."""
import hashlib
from pathlib import Path

from tools.diagnostics import luna_instrumented_source_install_v1 as installer


def test_installer_is_pilot_only_and_source_pinned():
    assert set(installer.FILES) == {"luna_lme_diagnostic_launch_v8.py"}
    assert "access-check" not in installer.REMOTE
    assert '"kind":"pilot"' in installer.REMOTE
    repo = Path(__file__).resolve().parents[1]
    relative, digest = installer.FILES["luna_lme_diagnostic_launch_v8.py"]
    assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == digest


def test_installer_projection_accepts_only_exact_finite_pilot_receipt():
    root = "/home/atta/.hymem-lme-diagnostic-preflight-instrumented1"
    unit = "hymem-luna-lme-diagnostic-preflight-instrumented1.service"
    value = {"schema": installer.SCHEMA, "prepared": True, "model_calls": 0,
        "root": root, "unit": unit, "kind": "pilot", "receipt_sha256": "a" * 64}
    assert installer._success_projection(value, root=root, kind="pilot", returncode=0) == value
    for mutation in ({**value, "kind": "access"}, {**value, "private": "secret"},
                     {**value, "model_calls": 1}, {**value, "receipt_sha256": "bad"}):
        assert installer._success_projection(mutation, root=root, kind="pilot", returncode=0) is None
    assert installer._success_projection(value, root=root, kind="pilot", returncode=1) is None
