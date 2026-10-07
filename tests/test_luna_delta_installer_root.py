"""Root verifies finite source-only installer output before any host writes."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.diagnostics import luna_delta_source_install_v1 as install

ROOT = "/home/atta/.hymem-lme-diagnostic-preflight-invented"
PRIVATE = "invented private provider detail"


def success():
    return {"schema": install.SCHEMA, "prepared": True, "model_calls": 0,
        "root": ROOT, "unit": "hymem-luna-lme-diagnostic-preflight-invented.service", "kind": "pilot",
        "receipt_sha256": "a" * 64}


def test_root_installer_reconstructs_only_expected_success():
    value = success()
    projected = install._success_projection(value, root=ROOT, kind="pilot", returncode=0)
    assert projected == value and projected is not value
    assert install._success_projection(value, root=ROOT, kind="pilot", returncode=1) is None
    assert install._success_projection(value, root=ROOT + "x", kind="pilot", returncode=0) is None


@pytest.mark.parametrize("field", list(success()))
@pytest.mark.parametrize("bad", [None, True, False, 1, [], {"private": PRIVATE}, PRIVATE])
def test_root_installer_never_exports_arbitrary_response(field, bad):
    value = copy.deepcopy(success())
    value[field] = bad
    out = install._success_projection(value, root=ROOT, kind="pilot", returncode=0)
    assert out is None or out == success()
    assert PRIVATE not in json.dumps(out)


def test_root_installer_exact_sources_and_prepare_only():
    repo = Path(__file__).resolve().parents[1]
    for _, (relative, digest) in install.FILES.items():
        assert hashlib.sha256((repo / relative).read_bytes()).hexdigest() == digest
    assert "--prepare-root" in install.REMOTE
    assert "--launch-root" not in install.REMOTE
    assert "luna_lme_diagnostic_launch_v7.py" in install.REMOTE
    assert "luna_lme_diagnostic_launch_v6.py" not in install.REMOTE
    assert install.ROOT.fullmatch(ROOT)
    assert not install.ROOT.fullmatch(ROOT + "/../production")


def test_root_installer_drops_unknown_keys():
    value = success()
    value["private"] = PRIVATE
    assert install._success_projection(value, root=ROOT, kind="pilot", returncode=0) is None
