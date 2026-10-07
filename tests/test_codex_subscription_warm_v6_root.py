"""Independent confidentiality/path/bounds controls; no real provider calls."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import stat

import pytest

from benchmarks import codex_subscription_warm_v6 as warm


PRIVATE = "INVENTED_PRIVATE_REASON_ONLY"


def notice():
    return {"method": "error", "params": {"threadId": "private-thread",
        "turnId": "private-turn", "willRetry": False, "error": {
            "message": PRIVATE, "additionalDetails": PRIVATE,
            "codexErrorInfo": {"responseStreamDisconnected": {"httpStatusCode": 403}}}}}


def projected(event=None):
    return warm._private_error(event or notice(), "private-thread", "private-turn")


def record(event=None):
    detail = projected(event)
    assert detail is not None
    return {"schema": "warm_private_failure_v1", "failure_code": "turn_failed",
            "error_count": 1, "first": detail, "last": detail}


def directory(tmp_path):
    path = tmp_path.resolve() / "private"
    path.mkdir(mode=0o700)
    return path


@pytest.mark.parametrize("field", ["params", "error", "codexErrorInfo"])
@pytest.mark.parametrize("value", [None, True, False, [], {}, 1, 1.5, PRIVATE])
def test_error_projection_is_total_for_malformed_json(field, value):
    item = notice()
    if field == "params":
        item[field] = value
    elif field == "error":
        item["params"][field] = value
    else:
        item["params"]["error"][field] = value
    result = projected(item)
    assert result is None or isinstance(result, dict)
    assert "private-thread" not in json.dumps(result)
    assert "private-turn" not in json.dumps(result)


@pytest.mark.parametrize("text", ["\0" * 20000, "😀" * 6000, "\\" * 20000,
                                  "\ud800" * 20000, "\ud800", "\n" * 20000])
def test_hostile_unicode_and_json_escaping_stay_within_record_limit(tmp_path, text):
    item = notice()
    item["params"]["error"].update(message=text, additionalDetails=text)
    data = record(item)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    assert len(payload) <= warm.MAX_PRIVATE_RECORD_BYTES
    path = directory(tmp_path)
    name = warm.PrivateFailureSink(path).write(data)
    target = path / name
    assert target.stat().st_size <= warm.MAX_PRIVATE_RECORD_BYTES
    assert json.loads(target.read_text()) == data


def test_private_prose_cannot_enter_public_failure_projection():
    data = record()
    public = warm.serialize_failure({"code": "turn_failed", "phase": "run",
        "rpc": "turn/events", "private_record": data, **data})
    encoded = json.dumps(public)
    assert PRIVATE not in encoded and "message" not in encoded and "additional" not in encoded


def test_record_limit_and_exclusive_publish_survive_concurrent_writers(tmp_path):
    path = directory(tmp_path)
    def write(_):
        try:
            return warm.PrivateFailureSink(path).write(copy.deepcopy(record()))
        except ValueError:
            return None
    with ThreadPoolExecutor(max_workers=8) as executor:
        outputs = list(executor.map(write, range(warm.MAX_PRIVATE_RECORDS + 12)))
    successful = [name for name in outputs if name is not None]
    assert len(successful) == len(set(successful)) == warm.MAX_PRIVATE_RECORDS
    files = list(path.iterdir())
    assert len(files) == warm.MAX_PRIVATE_RECORDS
    for item in files:
        assert item.name in successful
        assert stat.S_IMODE(item.stat().st_mode) == 0o600
        assert item.stat().st_nlink == 1
        assert json.loads(item.read_text()) == record()


def test_restrictive_umask_does_not_claim_unreadable_capture(tmp_path):
    path = directory(tmp_path)
    before = os.umask(0o777)
    try:
        name = warm.PrivateFailureSink(path).write(record())
    finally:
        os.umask(before)
    assert stat.S_IMODE((path / name).stat().st_mode) == 0o600
    assert json.loads((path / name).read_text()) == record()


def test_directory_and_slot_symlinks_never_follow_or_overwrite(tmp_path):
    path = directory(tmp_path)
    link = tmp_path.resolve() / "link"
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises((OSError, ValueError)):
        warm.PrivateFailureSink(link).write(record())
    with pytest.raises((OSError, ValueError)):
        warm.PrivateFailureSink(link / "child").write(record())
    target = tmp_path.resolve() / "sentinel"
    target.write_text("unchanged")
    target.chmod(0o600)
    (path / "warm-private-failure-00.json").symlink_to(target)
    with pytest.raises((OSError, ValueError)):
        warm.PrivateFailureSink(path).write(record())
    assert target.read_text() == "unchanged"
    assert {p.name for p in path.iterdir()} == {"warm-private-failure-00.json"}


def test_world_readable_directory_is_rejected_without_chmod(tmp_path):
    path = directory(tmp_path)
    path.chmod(0o755)
    with pytest.raises((OSError, ValueError)):
        warm.PrivateFailureSink(path).write(record())
    assert stat.S_IMODE(path.stat().st_mode) == 0o755
    assert list(path.iterdir()) == []


def test_observer_fault_cannot_replace_the_original_event(monkeypatch):
    instance = object.__new__(warm.WarmSession)
    instance.reset_private_errors()
    instance._observation_thread = "private-thread"
    instance._observation_turn = "private-turn"
    original = notice()
    monkeypatch.setattr(warm.v5.WarmSession, "next_event", lambda self: original)
    def fault(*args):
        raise RuntimeError(PRIVATE)
    monkeypatch.setattr(warm, "_private_error", fault)
    assert instance.next_event() is original


def test_record_and_reset_faults_cannot_skip_original_failure_accounting(monkeypatch):
    observed = []
    def inherited(self, *args):
        observed.append(args)
    monkeypatch.setattr(warm.v5.WarmSubscriptionClient, "_record_failure", inherited)
    class BrokenObserver:
        def private_failure_record(self, code):
            raise RuntimeError(PRIVATE)
        def reset_private_errors(self):
            raise RuntimeError(PRIVATE)
    client = object.__new__(warm.WarmSubscriptionClient)
    client.session = BrokenObserver()
    client.private_failure_sink = None
    client.private_sink_status = None
    client._record_failure("turn_failed", "run", True, False)
    assert observed == [("turn_failed", "run", True, False)]


@pytest.mark.parametrize("misalignment", [{}, {"errorType": PRIVATE},
    {"detailedExplanation": PRIVATE, "steer": {"message": PRIVATE}},
    {"errorType": None, "detailedExplanation": None, "steer": None}])
def test_documented_misalignment_is_presence_only(misalignment):
    item = notice()
    item["params"]["error"] = {"message": "safe", "misalignment": misalignment}
    result = projected(item)
    assert result["error_class"] == "unspecified"
    assert result["misalignment_present"] is True
    assert PRIVATE not in json.dumps(result)


@pytest.mark.parametrize("misalignment", [PRIVATE, [], True, {"steer": PRIVATE},
    {"steer": {}}, {"steer": {"message": []}}, {"errorType": []}])
def test_malformed_misalignment_is_not_recorded(misalignment):
    item = notice()
    item["params"]["error"]["misalignment"] = misalignment
    assert projected(item) is None


@pytest.mark.parametrize("kind", [[], {}, True, 12, None])
def test_unhashable_turn_kind_cannot_break_observer(kind):
    item = notice()
    item["params"]["error"]["codexErrorInfo"] = {"activeTurnNotSteerable": {"turnKind": kind}}
    assert projected(item) is None
