"""Independent metadata export controls for source-only installation."""
import copy
import json

import pytest

from tools.diagnostics import luna_existing_credit_source_install_root as install


ROOT = "/home/atta/.hymem-lme-diagnostic-preflight-invented"
PRIVATE = "invented private provider detail"


def success(kind):
    prefix = "hymem-luna-access-check-" if kind == "access" else "hymem-luna-lme-diagnostic-"
    return {"schema": install.SCHEMA, "prepared": True, "model_calls": 0,
        "root": ROOT, "unit": prefix + "preflight-invented.service", "kind": kind,
        "receipt_sha256": "a" * 64}


@pytest.mark.parametrize("kind", ["access", "pilot"])
def test_expected_success_is_reconstructed(kind):
    value = success(kind)
    projected = install._success_projection(value, root=ROOT, kind=kind, returncode=0)
    assert projected == value and projected is not value
    assert install._success_projection(value, root=ROOT, kind=kind, returncode=1) is None


@pytest.mark.parametrize("field", list(success("access")))
@pytest.mark.parametrize("bad", [None, True, False, 1, [], {"private": PRIVATE}, PRIVATE])
def test_untrusted_value_does_not_escape(field, bad):
    value = copy.deepcopy(success("access"))
    value[field] = bad
    out = install._success_projection(value, root=ROOT, kind="access", returncode=0)
    assert out is None or out == success("access")
    assert PRIVATE not in json.dumps(out)


def test_unknown_nested_keys_never_escape():
    value = success("access")
    value["status"] = {"private": PRIVATE}
    assert install._success_projection(value, root=ROOT, kind="access", returncode=0) is None
    assert install._success_projection({"status": {"private": PRIVATE}},
        root=ROOT, kind="access", returncode=0) is None


def test_root_and_kind_must_match_actual_dispatch_intent():
    value = success("access")
    assert install._success_projection(value, root=ROOT + "x", kind="access", returncode=0) is None
    assert install._success_projection(value, root=ROOT, kind="pilot", returncode=0) is None
