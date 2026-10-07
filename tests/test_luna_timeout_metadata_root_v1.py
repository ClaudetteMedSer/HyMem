"""Invented finite metadata controls; no SSH, model calls or private data."""
import copy
import json
import pytest
from tools.diagnostics import luna_timeout_metadata_root_v1 as probe

def sample():
    return {"budget": {"first_failure": {"code":"timeout", "phase":"run", "rpc":"turn/events",
        "process_age_seconds":200.5}, "timings": {"preflight_seconds":2.5,"model_seconds":125.0,
        "cleanup_seconds":1.0}, "questions": {"q-0000":{"turns":15,"known_tokens":12345,
            "in_flight":0,"usage_complete":False,"stopped":False}}}}

def record():
    error={"error_class":"responseStreamDisconnected","http_status_code":403,"will_retry":True,
           "message":"PRIVATE_SENTINEL","additional_details":"PRIVATE_SENTINEL"}
    return {"schema":"warm_private_failure_v1","failure_code":"timeout","error_count":1,
            "first":error,"last":error}

def test_safe_projection_drops_private_messages_and_unknown_fields():
    value=sample(); value["raw"]="PRIVATE_SENTINEL"
    out=probe.projection(value,[("q-0000",0,record())])
    assert "PRIVATE_SENTINEL" not in json.dumps(out)
    assert out["private_error_slots"][0]["first"]["http_status_code"]==403
    assert out["first_failure_process_age_seconds"]==200.5

@pytest.mark.parametrize("value", [None,True,False,"PRIVATE_SENTINEL",{},[],float("nan"),float("inf"),-1,1000001])
def test_invalid_numeric_age_is_never_exported(value):
    data=sample();data["budget"]["first_failure"]["process_age_seconds"]=value
    with pytest.raises((ValueError,TypeError)):probe.projection(data,[])

@pytest.mark.parametrize("field,value", [("error_class","PRIVATE_SENTINEL"),("error_class",{}),
    ("http_status_code",True),("http_status_code","PRIVATE_SENTINEL"),("will_retry",1)])
def test_private_or_malformed_classification_fails_closed(field,value):
    rec=copy.deepcopy(record());rec["first"][field]=value
    with pytest.raises((ValueError,TypeError)):probe.projection(sample(),[("q-0000",0,rec)])

def test_paths_and_question_keys_are_finite():
    with pytest.raises(ValueError):probe.projection(sample(),[("../other",0,record())])
    data=sample();data["budget"]["questions"]["PRIVATE_SENTINEL"]={}
    with pytest.raises(ValueError):probe.projection(data,[])

def test_zero_error_slots_is_not_invented_error_evidence():
    assert probe.projection(sample(),[])["private_error_slots"]==[]
