"""Read-only finite stack/source metadata; never print locals, logs or env."""
import hashlib
import json
import pathlib
import re
import subprocess


def main():
    command = ["docker", "exec", "hermes-1", "/home/node/hymem-env/bin/py-spy",
               "dump", "--pid", "12381", "--json", "--native"]
    result = subprocess.run(command, capture_output=True, timeout=15)
    if result.returncode:
        print(json.dumps({"stack_error": "profiler_failed", "rc": result.returncode}))
        return
    raw = json.loads(result.stdout)
    rows = raw if isinstance(raw, list) else raw.get("threads", [])
    output = []
    for row in rows:
        frames = []
        for frame in row.get("frames", []):
            filename = str(frame.get("filename", ""))
            name = str(frame.get("name", ""))
            # Profiler function/file metadata only, never source lines/locals.
            frames.append({"function": name[:150] if re.fullmatch(r"[\w .<>:-]+", name) else "other",
                           "file": pathlib.PurePosixPath(filename).name[:100],
                           "line": frame.get("line")})
        output.append({"thread_id": row.get("thread_id"), "active": row.get("active"),
                       "owns_gil": row.get("owns_gil"), "frames": frames})
    print(json.dumps({"stack_sha256": hashlib.sha256(result.stdout).hexdigest(),
                      "thread_count": len(output), "threads": output}, sort_keys=True))


if __name__ == "__main__":
    main()
