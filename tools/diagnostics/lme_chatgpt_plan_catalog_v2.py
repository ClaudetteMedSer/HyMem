"""One-shot, read-only SIWC listing of models marked visible in the catalog.

Run with ``python -I -B ... --state-dir ABSOLUTE_PRIVATE_DIRECTORY``. This
tool never refreshes a token, writes state, or calls an inference endpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import ssl
import stat
import sys
import time
from urllib import error as url_error, request

sys.dont_write_bytecode = True

CATALOG_SHA256 = "599a74dd6c8010f37f7f304ec34a695930a461c1b0b05afa6eba5a6cfe68bc92"
CATALOG_URL = "https://api.openai.com/v1/models"
MAX_RESPONSE = 1024 * 1024
REQUEST_TIMEOUT = 15
ABSOLUTE_TIMEOUT = 30
MAX_VISIBLE = 128
SLUG_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")


class ListError(Exception):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _AbsoluteTimeout(BaseException):
    pass


def _load_catalog_module():
    source = Path(__file__).with_name("lme_chatgpt_plan_catalog_v1.py")
    try:
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > 256 * 1024:
                raise ListError("source_mismatch")
            raw = os.read(fd, 256 * 1024 + 1)
        finally:
            os.close(fd)
        if hashlib.sha256(raw).hexdigest() != CATALOG_SHA256:
            raise ListError("source_mismatch")
        spec = importlib.util.spec_from_file_location("_pinned_lme_catalog_v1", source)
        if spec is None:
            raise ListError("source_unavailable")
        module = importlib.util.module_from_spec(spec)
        exec(compile(raw, str(source), "exec"), module.__dict__)
        return module
    except ListError:
        raise
    except Exception:
        raise ListError("source_unavailable") from None


def _catalog_get(token: str, catalog) -> list[dict[str, str]]:
    opener = request.build_opener(request.ProxyHandler({}), catalog._NoRedirect(),
                                  request.HTTPSHandler(context=ssl.create_default_context()))
    req = request.Request(CATALOG_URL, headers={"Accept": "application/json", "Authorization": "Bearer " + token},
                          method="GET")
    try:
        with opener.open(req, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:
                raise ListError("catalog_http_error")
            raw = response.read(MAX_RESPONSE + 1)
    except ListError:
        raise
    except catalog.CheckError as exc:
        raise ListError(exc.code) from None
    except url_error.HTTPError:
        raise ListError("catalog_http_error") from None
    except Exception:
        raise ListError("catalog_unavailable") from None
    if len(raw) > MAX_RESPONSE:
        raise ListError("catalog_too_large")
    try:
        value = json.loads(raw, object_pairs_hook=catalog._no_duplicate_keys,
                           parse_constant=lambda _value: (_ for _ in ()).throw(ValueError("nonstandard JSON value")))
    except (ValueError, UnicodeError):
        raise ListError("catalog_invalid") from None
    if type(value) is not dict or type(value.get("models")) is not list or len(value["models"]) > 10000:
        raise ListError("catalog_invalid")
    visible = []
    seen = set()
    for entry in value["models"]:
        if type(entry) is not dict:
            raise ListError("catalog_invalid")
        slug = entry.get("slug")
        name = entry.get("display_name")
        visibility = entry.get("visibility")
        if (type(slug) is not str or SLUG_RE.fullmatch(slug) is None or slug in seen
                or type(name) is not str or not 1 <= len(name) <= 128 or not name.isprintable()
                or type(visibility) is not str or not 1 <= len(visibility) <= 64
                or not visibility.isascii() or not visibility.isprintable()):
            raise ListError("catalog_invalid")
        seen.add(slug)
        if visibility == "list":
            if len(visible) == MAX_VISIBLE:
                raise ListError("catalog_invalid")
            visible.append({"slug": slug, "display_name": name})
    return visible


def check(state_dir: Path) -> dict:
    prior = signal.getsignal(signal.SIGALRM)
    def expire(_signum, _frame):
        raise _AbsoluteTimeout()
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, ABSOLUTE_TIMEOUT)
    requests = 0
    try:
        catalog = _load_catalog_module()
        signin = catalog._load_signin_module()
        fd = catalog._open_private_dir(state_dir)
        try:
            token = catalog._validated_access_token(fd, signin, int(time.time()))
        finally:
            os.close(fd)
        requests = 1
        models = _catalog_get(token, catalog)
        return {"models": models, "model_calls": 0, "catalog_requests": requests}
    except _AbsoluteTimeout:
        code = "deadline_exceeded"
    except ListError as exc:
        code = exc.code
    except Exception as exc:
        code = exc.code if "catalog" in locals() and isinstance(exc, catalog.CheckError) else "internal_error"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
    return {"error": code, "models": [], "model_calls": 0, "catalog_requests": requests}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    result = check(args.state_dir)
    print(json.dumps(result, sort_keys=True, ensure_ascii=True), flush=True)
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
