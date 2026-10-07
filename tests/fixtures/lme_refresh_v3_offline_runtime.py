#!/opt/anaconda3/bin/python3.13
"""Invented-response runtime for the owner3 subprocess contract; never network."""

import importlib.util
import json
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 7 or sys.argv[1:3] != ["-I", "-B"] or sys.argv[4:6] != ["refresh", "--state-dir"]:
        return 2
    source = Path(sys.argv[3])
    if source.name != "lme_chatgpt_plan_refresh_v3.py":
        return 2
    spec = importlib.util.spec_from_file_location("offline_refresh_v3_child", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def invented_exchange(url, form):
        assert url == "https://auth.openai.com/api/accounts/oauth/token"
        assert form == {"grant_type": "refresh_token", "client_id": "oaiapp_invented123",
                        "refresh_token": "invented-refresh-old",
                        "resource": "https://api.openai.com/v1"}
        return {"token_type": "Bearer", "expires_in": 3600,
                "access_token": "invented-access-new", "refresh_token": "invented-refresh-new"}

    module._http_json = invented_exchange
    result = module.run(Path(sys.argv[6]))
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
