"""Root checks that the offline host wrapper exports no arbitrary text."""
import copy
import json

import pytest

from tools.diagnostics import luna_delta_mock_host_verify_root as wrapper


def metadata():
    row = {"case": "baseline", "status": "completed",
           **{key: 1 for key in wrapper.COUNTS},
           **{key: True for key in wrapper.FLAGS},
           "usage_digest": "a" * 64, "failure_code": None}
    return {"schema": wrapper.SCHEMA, "source_sha256": "b" * 64,
            "network_policy_verified": True, "cleanup_verified": True,
            "status": "observed", "observation": {
                "schema": wrapper.SOURCE_SCHEMA, "verified": False, "results": [row]}}


def test_projection_is_numeric_and_fixed_status_only():
    value = metadata()
    value["private"] = "DO-NOT-EXPORT"
    value["observation"]["results"][0]["failure_method"] = "DO-NOT-EXPORT"
    value["observation"]["results"][0]["failure_code"] = "DO-NOT-EXPORT"
    result = wrapper.project(value, "b" * 64)
    assert result["observation"]["results"][0]["failure_present"] is True
    assert "DO-NOT-EXPORT" not in json.dumps(result)


@pytest.mark.parametrize("key,bad", [("http_requests", True), ("http_requests", -1),
    ("delta_notifications", 6001), ("status", "private"),
    ("same_turn_identity", "private"), ("usage_digest", "private"),
    ("case", "private"), ("cleanup_verified", 1)])
def test_malformed_row_never_exports(key, bad):
    value = copy.deepcopy(metadata())
    value["observation"]["results"][0][key] = bad
    with pytest.raises((ValueError, TypeError)):
        wrapper.project(value, "b" * 64)


@pytest.mark.parametrize("key,bad", [("schema", "private"), ("source_sha256", "c" * 64),
    ("status", "private"), ("cleanup_verified", None), ("network_policy_verified", "yes")])
def test_malformed_outer_metadata_never_exports(key, bad):
    value = metadata()
    value[key] = bad
    with pytest.raises(ValueError):
        wrapper.project(value, "b" * 64)
