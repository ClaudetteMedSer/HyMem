"""No-inference diagnostic; prints fixed codes and non-sensitive metadata only."""
from __future__ import annotations

import argparse
import json

from benchmarks.codex_subscription import CodexSubscriptionClient, SubscriptionTransportError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True)
    args = parser.parse_args()
    try:
        metadata = CodexSubscriptionClient(args.binary).preflight()
    except SubscriptionTransportError as exc:
        print(json.dumps({"ok": False, "stop_code": str(exc), "observed_turns": 0,
                          "internal_http_attempts": None}))
        return 1
    except Exception:
        print(json.dumps({"ok": False, "stop_code": "diagnostic_failure", "observed_turns": 0,
                          "internal_http_attempts": None}))
        return 1
    print(json.dumps(metadata, sort_keys=True))
    return 0 if (metadata.get("config_isolation_admitted") is True
                 and metadata.get("runtime_probe_verified") is False
                 and metadata.get("inference_enabled") is False
                 and metadata.get("observed_turns") == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
