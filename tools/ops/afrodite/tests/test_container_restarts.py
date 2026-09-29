"""Offline regression tests of the actual section-1 shell/Python branch.

Run: python3 -m unittest discover -s tools/ops/afrodite/tests -v
No Docker, network, stack probes, notification state, or deployed files touched.
"""
import copy
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "stack-health-check.sh"
SECTION = SOURCE.read_text().split("# 1. container state:", 1)[1].split(
    "# 2. service reachability,", 1)[0]
SECTION = SECTION.split("\n", 1)[1]
SECTION = re.sub(r"EXPECTED=\(.*?\)", "EXPECTED=(fixture)", SECTION, count=1, flags=re.S)
# macOS system Bash 3 lacks associative arrays. One fixture container uses
# index zero; only the storage declaration changes, not production check logic.
SECTION = SECTION.replace("declare -A C_STATE C_HEALTH C_RESTARTS C_IP",
                          "fixture=0; declare -a C_STATE C_HEALTH C_RESTARTS C_IP")
NOW = 1790438400  # 2026-09-26T16:00:00Z
BASE = {"Id": "a" * 64, "RestartCount": 1,
        "State": {"Status": "running", "Restarting": False,
                  "StartedAt": "2026-09-25T17:53:00.123456789Z",
                  "Health": {"Status": "healthy"}},
        "NetworkSettings": {"Networks": {"hermes-net": {"IPAddress": "172.18.0.2"}}}}


class ContainerRestarts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.directory = Path(self.tmp.name)
        self.baseline = self.directory / "container-fixture.json"

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, obj=None, *, raw=None, now=NOW, failed=False, state_dir=None):
        env = dict(os.environ, MOCK_INSPECT=json.dumps([BASE if obj is None else obj]) if raw is None else raw,
                   MOCK_RC="1" if failed else "0", STATE_DIR=str(state_dir or self.directory), START_EPOCH=str(now))
        prelude = '''set -uo pipefail
FINDINGS=(); OK_COUNT=0
add_fail() { FINDINGS+=("FAIL"$'\\t'"$1"$'\\t'"$2"); }
add_warn() { FINDINGS+=("WARN"$'\\t'"$1"$'\\t'"$2"); }
add_ok() { OK_COUNT=$((OK_COUNT + 1)); }
docker() { printf '%s' "$MOCK_INSPECT"; return "$MOCK_RC"; }
date() { printf '%s' "$START_EPOCH"; }
'''
        result = subprocess.run(["bash"], input=prelude + SECTION + '\nprintf "%s\\n" ${FINDINGS+"${FINDINGS[@]}"}\n',
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, "")
        return result.stdout.strip()

    def test_historical_count_initial_warning_then_stable(self):
        self.assertIn("WARN\tCONTAINER RESTART BASELINE INITIAL", self.run_check())
        self.assertEqual(self.run_check(), "")
        self.assertEqual(json.loads(self.baseline.read_text())["count"], 1)

    def test_zero_count_initial_observation_is_honest(self):
        obj = copy.deepcopy(BASE); obj["RestartCount"] = 0
        self.assertIn("INITIAL", self.run_check(obj))
        self.assertEqual(self.run_check(obj), "")

    def test_changes_fail_once_then_recover(self):
        for change in ("increase", "decrease", "recreate", "started"):
            with self.subTest(change=change):
                self.run_check(BASE)
                obj = copy.deepcopy(BASE)
                if change == "increase": obj["RestartCount"] = 2
                if change == "decrease": obj["RestartCount"] = 0
                if change == "recreate": obj["Id"] = "b" * 64; obj["RestartCount"] = 0
                if change == "started": obj["State"]["StartedAt"] = "2026-09-26T10:00:00Z"
                self.assertIn("FAIL\tCONTAINER RESTART OBSERVED", self.run_check(obj))
                self.assertEqual(self.run_check(obj), "")
                self.baseline.unlink()

    def test_active_restart_always_fails(self):
        for status, flag in (("running", True), ("restarting", False), ("restarting", True)):
            obj = copy.deepcopy(BASE); obj["State"].update(Status=status, Restarting=flag)
            self.assertIn("FAIL\tCONTAINER RESTARTING", self.run_check(obj))
            self.assertIn("FAIL\tCONTAINER RESTARTING", self.run_check(obj))

    def test_down_and_health_findings_preserved(self):
        self.run_check()
        for status in ("exited", "dead", "paused", "created", "removing"):
            obj = copy.deepcopy(BASE); obj["State"]["Status"] = status
            self.assertIn("FAIL\tCONTAINER DOWN", self.run_check(obj))
        for health, expected in (("unhealthy", "FAIL\tCONTAINER UNHEALTHY"),
                                 ("starting", "WARN\tCONTAINER STARTING"), ("healthy", ""), (None, "")):
            obj = copy.deepcopy(BASE)
            if health is None: del obj["State"]["Health"]
            else: obj["State"]["Health"]["Status"] = health
            self.assertEqual(self.run_check(obj).split("\t")[0:2], expected.split("\t") if expected else [""])

    def test_stable_restart_history_does_not_clear_a_sick_service(self):
        self.run_check()
        for down in (False, True):
            obj = copy.deepcopy(BASE); obj["RestartCount"] = 2
            if down: obj["State"]["Status"] = "exited"
            else: obj["State"]["Health"]["Status"] = "unhealthy"
            self.run_check(obj)
            finding = "CONTAINER DOWN" if down else "CONTAINER UNHEALTHY"
            self.assertIn("FAIL\t" + finding, self.run_check(obj))
            self.assertNotIn("CONTAINER RESTART OBSERVED", self.run_check(obj))

    def test_inspect_failures_and_malformed_data_never_green(self):
        self.run_check(); original = self.baseline.read_text()
        self.assertIn("CONTAINER MISSING", self.run_check(failed=True))
        self.assertIn("CONTAINER MISSING", self.run_check(raw=""))
        for raw in ("junk", "[]", "{}", '[{}]'):
            self.assertIn("CONTAINER STATE UNREADABLE", self.run_check(raw=raw))
        for field, value in (("RestartCount", -1), ("RestartCount", "1"), ("RestartCount", True), ("Id", "bad")):
            obj = copy.deepcopy(BASE); obj[field] = value
            self.assertIn("CONTAINER STATE UNREADABLE", self.run_check(obj))
        for field, value in (("Restarting", "false"), ("Status", "unknown"), ("StartedAt", "nonsense"),
                             ("StartedAt", "2026-09-27T10:00:00Z")):
            obj = copy.deepcopy(BASE); obj["State"][field] = value
            self.assertIn("CONTAINER STATE UNREADABLE", self.run_check(obj))
        obj = copy.deepcopy(BASE); obj["State"]["Health"]["Status"] = "bogus"
        self.assertIn("CONTAINER STATE UNREADABLE", self.run_check(obj))
        self.assertEqual(self.baseline.read_text(), original)

    def test_corrupt_baseline_fails_then_rebaselines(self):
        for raw in ("junk", "{}", "[]"):
            self.baseline.write_text(raw)
            self.assertIn("FAIL\tCONTAINER RESTART BASELINE UNREADABLE", self.run_check())
            self.assertEqual(self.run_check(), "")

    def test_missing_baseline_warns_again(self):
        self.run_check(); self.baseline.unlink()
        self.assertIn("INITIAL", self.run_check())

    def test_future_baseline_fails_and_recovers(self):
        self.run_check()
        previous = json.loads(self.baseline.read_text()); previous["observed"] = NOW + 100
        self.baseline.write_text(json.dumps(previous))
        self.assertIn("FAIL\tCONTAINER RESTART BASELINE UNREADABLE", self.run_check())
        self.assertEqual(self.run_check(), "")

    def test_current_second_nanoseconds_are_valid(self):
        from datetime import datetime, timezone
        obj = copy.deepcopy(BASE)
        obj["State"]["StartedAt"] = datetime.fromtimestamp(NOW, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S") + ".999999999Z"
        self.assertIn("INITIAL", self.run_check(obj))
        self.assertEqual(self.run_check(obj), "")

    def test_docker_projects_only_required_metadata_before_helper_argv(self):
        # Regression guard at the actual invocation boundary: Docker must
        # receive the narrow format before its output reaches the helper.
        invocation = SECTION.split('if ! info="$(docker inspect', 1)[1].split('2>/dev/null)', 1)[0]
        self.assertIn("--format", invocation)
        self.assertNotIn("Config", invocation)
        self.assertNotIn("Env", invocation)
        self.assertNotIn(".State.Health", invocation)
        self.assertIn('with index .State "Health"', invocation)
        self.assertIn('"Health":{"Status":{{json .Status}}}', invocation)

    def test_persist_failure_fails_even_for_stable_container(self):
        self.run_check()
        before = self.baseline.read_text()
        self.assertIn("FAIL\tCONTAINER RESTART BASELINE UNREADABLE",
                      self.run_check(state_dir=self.baseline))  # file cannot be a directory, even as root
        self.assertEqual(self.baseline.read_text(), before)


if __name__ == "__main__":
    unittest.main()
