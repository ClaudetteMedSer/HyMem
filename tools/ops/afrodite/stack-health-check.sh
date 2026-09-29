#!/usr/bin/env bash
#
# Afrodite stack health check — one pane of glass for all nine stacks.
# Written 18 August 2026.
#
#   /opt/stacks/hermes/stack-health-check.sh            run the checks
#   /opt/stacks/hermes/stack-health-check.sh --check    report on the LAST run
#                                                       without probing anything
#   /opt/stacks/hermes/stack-health-check.sh --quiet    cron mode: print nothing
#                                                       to stdout, only log
#   /opt/stacks/hermes/stack-health-check.sh --no-notify
#                                                       run everything, never
#                                                       touch Telegram (testing)
#   ... --deadline-utc HH:MM   override the "daily jobs should be done by" time
#   ... --meta FILE            use a different TTSP meta.json  (fixture testing)
#   ... --summary-dir DIR      use a different richtlijn dir   (fixture testing)
#   ... --episodes DIR         use a different episodes dir    (fixture testing)
#   ... --hermes-home DIR      use a different ~/.hermes tree  (fixture testing)
#   ... --state-dir DIR        put lock/log/state somewhere else (fixture testing)
#   ... --searxng-base URL     point section 7b at a fixture instead of SearXNG
#
# The fixture options exist so the stale-output detection can be PROVED against
# a constructed copy of the 17 August situation without touching real data.
# They are not for normal use; cron passes none of them.
#
# ===========================================================================
# WHY THIS EXISTS — the failure it is built to catch
# ===========================================================================
# On 17 August 2026 we found that the TTSP podcast job had reported `ok` every
# weekday for SEVEN WEEKS while building each episode from the PREVIOUS day's
# summary. Both it and the summariser it depends on recorded success. Nothing
# was watching the OUTPUT, only the exit status.
#
# A check that asks "is the container up" would have missed it. So would one
# that asks "did the job exit 0". The whole point of this script is the third
# question: **is the thing it produced actually right and actually today's?**
#
# The second lesson already recorded on this box points the same way: SearXNG's
# /healthz returned 200 in 3 ms straight through a TOTAL engine outage. On this
# host a liveness probe proves nothing. Every reachability probe below therefore
# asserts on RESPONSE CONTENT, not just on the status code.
#
# ===========================================================================
# QUIET WHEN HEALTHY
# ===========================================================================
# A watchdog you learn to ignore is worse than none. So:
#   * healthy run  -> writes one line to the log, sends nothing.
#   * new fault    -> ONE Telegram message, with a specific headline naming the
#                     failure class (never a generic "something is wrong").
#   * same fault   -> silence, re-sent at most every REALERT_HOURS.
#   * recovery     -> one RECOVERED message, so a silence is never ambiguous.
# The signature/cooldown machinery is copied deliberately from
# ~/.hermes/scripts/search-watchdog.py so both watchdogs behave identically.
#
# ===========================================================================
# WHY THIS IS A HOST SCRIPT AND NOT A `hermes cron` JOB
# ===========================================================================
# hermes-1 has no Docker socket, no docker binary and no view of /opt/stacks.
# Container state, restart counts, the edge stack and the Paperless stack are
# all invisible from inside it. So this has to run on the host, from atta's
# crontab, and it cannot use `hermes cron`'s Telegram delivery — it talks to the
# Bot API directly (see notify() for how the token is kept out of `ps`).
#
# ===========================================================================
# IT IS READ-ONLY WITH RESPECT TO THE STACKS
# ===========================================================================
# `docker inspect`, `docker stats --no-stream`, `docker top`, HTTP GETs, and
# reads of bind-mounted files. No exec, no restart, no compose, nothing that
# writes into a container. The only files it writes are its own under
# $STATE_DIR plus one published status copy the agent can read.
#
# ===========================================================================
# IT NEVER RUNS A SEARCH QUERY
# ===========================================================================
# Cumulative automated probing already got this host's egress IP rate-limited
# and knocked out several engines for hours: diagnosing search degrades search.
# Search health here is read from the existing hourly watchdog's state file and
# log. This script issues exactly ONE request that leaves the building — the
# public edge probe to podcast.hermeshosting.cc, which is our own Cloudflare
# hostname and hits no third-party index.
#
# ===========================================================================
# EXIT CODES  (same shape as backup-hermes-state.sh / paperless/backup.sh)
# ===========================================================================
#   0  healthy
#   2  warnings only (degraded, nothing down)
#   3  one or more failures
#   4  the check itself could not complete (no docker, etc.)
#  64  bad usage
#  75  another run holds the lock
#
# --check reads the recorded result instead of probing:
#   0 healthy   2 warnings   3 failures   4 stale/unreadable   5 never run

# `set -e` is deliberately NOT used. This script's whole job is to run commands
# that are EXPECTED to fail and to report on them; -e would make the first sick
# service abort the run and hide every check after it. Errors are handled
# explicitly instead. pipefail stays on so a broken pipeline cannot look like
# success.
set -uo pipefail

# ---------------------------------------------------------------------------
# paths and tunables
# ---------------------------------------------------------------------------

STACKS=/opt/stacks
HERMES_HOME_HOST="$STACKS/hermes/instance1/home/.hermes"

STATE_DIR="$STACKS/hermes/health"

TTSP_META="$STACKS/ttsp/meta.json"
TTSP_EPISODES="$STACKS/ttsp/episodes"
HERMES_BACKUP="$STACKS/hermes/backup-hermes-state.sh"
PAPERLESS_BACKUP="$STACKS/paperless/backup.sh"
PAPERLESS_CONSUME="$STACKS/paperless/consume"

# Everything below hangs off HERMES_HOME_HOST, so it is derived AFTER argument
# parsing (see derive_paths) rather than here — otherwise --hermes-home could
# only ever move some of the paths and would silently leave the rest pointing at
# real data, which is the worst possible outcome for a fixture flag.
SUMMARY_DIR=""; ENV_FILE=""; CRON_JOBS=""; SEARCH_STATE=""; SEARCH_LOG=""
RICHTLIJN_WATCHDOG=""; NETWATCH=""

derive_paths() {
    : "${ENV_FILE:=$HERMES_HOME_HOST/.env}"
    : "${SUMMARY_DIR:=$HERMES_HOME_HOST/cron/output}"
    : "${CRON_JOBS:=$HERMES_HOME_HOST/cron/jobs.json}"
    : "${SEARCH_STATE:=$HERMES_HOME_HOST/cron/search-watchdog-state.json}"
    : "${SEARCH_LOG:=$HERMES_HOME_HOST/cron/search-watchdog.log}"
    : "${RICHTLIJN_WATCHDOG:=$HERMES_HOME_HOST/scripts/ttsp-richtlijn-watchdog.py}"
    : "${NETWATCH:=$HERMES_HOME_HOST/scripts/netwatch.py}"
    : "${PUBLISHED_STATUS:=$HERMES_HOME_HOST/stack-health-status.json}"
}

# The status copy the agent inside hermes-1 can read at
# ~/.hermes/stack-health-status.json. Exactly the trick backup-hermes-state.sh
# uses: the container cannot see /opt/stacks, so hand it a copy in the one
# directory that is bind-mounted. Derived with the rest, so a fixture run
# publishes into the fixture tree and never overwrites the real status file.
PUBLISHED_STATUS=""

# Telegram. There is no origin to inherit on the host, so the chat id is
# pinned. 5437100263 is atta, and matches TELEGRAM_HOME_CHANNEL in ~/.hermes/.env.
TG_CHAT=5437100263

# Re-alert cadence while a fault persists, in hours. Same value and same reason
# as search-watchdog.py: without it, a 15-minute watchdog sends 96 identical
# messages a day during one multi-day outage. A CHANGED signature always alerts
# immediately regardless of the cooldown.
REALERT_HOURS=6

# The daily-output checks (richtlijn summary, podcast episode) only mean
# something once the jobs that produce them have had their chance. In UTC —
# which is what `hermes cron` expressions use — the chain is (moved here on
# 19 Aug 2026 to keep the DeepSeek spend outside peak pricing):
#   04:15 summariser        b28fc7d18fa1  (observed running up to 59 min)
#   05:15 TTSP podcast      8d3c80e70295  (observed up to 27 min once running,
#                                          but ttsp-precondition.py will wait
#                                          MAX_WAIT_SECONDS=3300 for a late
#                                          summariser, so the job itself cannot
#                                          still be alive after ~06:40)
#   07:00 TTSP richtlijn backstop watchdog  74d146df2027
# 08:00 UTC (10:00 local) is one clear hour after the last of them. Before that
# time the daily checks report "not due yet" and stay silent, which is why this
# script can run every 15 minutes without screaming at 02:00 every night.
#
# These three numbers move together. If a job time changes again, this deadline
# and the backstop's cron expression have to change with it — otherwise the
# 15-minute run that lands between the deadline and the job produces exactly
# one false failure and then one false recovery, which is what happened on the
# morning of 19 Aug 2026 (07:07 local fail, 07:22 local recovery).
DEADLINE_UTC="08:00"

# Smallest genuine richtlijn summary ever seen here is 10,948 bytes. 5,000
# rejects a stub without ever rejecting a real summary — the same floor
# ttsp-precondition.py and ttsp-richtlijn-watchdog.py already use. Kept
# identical on purpose: three components disagreeing about what "a real
# summary" means is its own bug.
MIN_SUMMARY_BYTES=5000

# Headroom thresholds. Memory is the binding constraint on this box: 12 GB
# total, ~7.8 GB available at rest, Camofox capped at 2 GB, and PDF parsing in
# web-extract peaks around 1 GB. So "available" must stay comfortably above one
# PDF peak plus normal churn.
MEM_WARN_MB=2000      # under this, one PDF parse plus a Camofox page is tight
MEM_FAIL_MB=900       # under this, the next PDF parse is an OOM kill
# Swap is judged by FLOW, not by how much is parked (see section 10 for the
# measurement that forced this change). Both rates are deltas of cumulative
# kernel counters taken between consecutive runs, so a 15-minute interval
# cannot miss a burst the way a spot sample can.
SWAP_RATE_WARN_MBMIN=10   # sustained swap-OUT per minute between runs. The
                          # 6-day average on this box — including the heavy
                          # HyMem re-embed that filled swap in the first place
                          # — was 1.1 MB/min, so this is ~9x the worst real
                          # week on record.
MEM_PSI_WARN_PCT=10       # % of wall clock with at least one task stalled on
                          # memory reclaim, from /proc/pressure/memory. This is
                          # the kernel's own answer to "is memory hurting
                          # throughput". Measured at rest: 0.003%.
SWAP_FULL_WARN_PCT=85     # of swap_total. The one legitimate use of the stock:
                          # swap that is nearly full turns the next spike into
                          # an OOM. Below this, a large residue is normal.
DISK_WARN_PCT=85
DISK_FAIL_PCT=93
CAMOFOX_CAP_WARN_PCT=85   # of its own 2 GB cgroup limit, not of host RAM

# netwatch --check exits 4 on ANY single lossy minute in 24 h. On Starlink that
# is normal weather, not an incident — 2 lossy minutes out of 241 was the state
# of a perfectly healthy link when this script was written. Taking that exit
# code at face value would make this watchdog cry wolf daily, which is the one
# thing it must not do. So exit 2/3 (no data / stale = the every-minute cron job
# has stopped) are treated as real, and degradation is judged against these
# thresholds instead, read off netwatch's own summary.
NET_LOSS_MINUTES_WARN_PCT=8   # >8% of sampled minutes showing any loss
NET_DNS_FAIL_WARN=3           # minutes with BOTH resolvers failing
NET_TLS_FAIL_WARN=3           # failed IMAP greeting probes

STALE_CHECK_MIN=60            # --check calls the recorded result stale past this

# --- SearXNG engine pool (section 7b) --------------------------------------
# Added 29 August 2026, the day after four of ten enabled general engines were
# found delivering ZERO results while /healthz returned 200 throughout. These
# numbers are set FROM THE MEASURED BASELINE of that day, not from taste:
# 8 enabled general engines, of which duckduckgo and startpage were 100%
# CAPTCHA (deliberately left enabled, see settings.yml), nothing at 403, and
# google cse 15% rate-limited. So the healthy pool here is SIX, and a floor
# above six would have alarmed on day one — the one thing this script must not
# do.
SEARXNG_STATS_BASE="http://127.0.0.1:8888"
# A single 403 rounds to 0% in SearXNG's integer percentages, so this must sit
# well above 0 or a one-off blip reads as a dead engine. 100% is what infospace
# and wikidata actually showed.
ENGINE_DEAD_PCT=50
# google cse sat at 15% on a healthy day and is on a borrowed, revocable shared
# CX token, so it is the engine most likely to move. 40 is comfortably clear of
# its normal noise while still catching a token going bad.
ENGINE_THROTTLE_WARN_PCT=40
ENGINE_POOL_WARN=5     # one engine below the measured healthy six
ENGINE_POOL_FAIL=4     # half the enabled general pool gone
# /stats/errors counts cumulatively SINCE THE CONTAINER STARTED. On a fresh
# container a handful of requests make every percentage meaningless, so below
# this uptime the section reports "not enough data" instead of guessing. Same
# instinct as the daily-output checks' "not due".
ENGINE_STATS_MIN_UPTIME_MIN=120

# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------

MODE=run
QUIET=0
NOTIFY=1
while [ $# -gt 0 ]; do
    case "$1" in
        --check)        MODE=check ;;
        --quiet)        QUIET=1 ;;
        --no-notify)    NOTIFY=0 ;;
        --deadline-utc) DEADLINE_UTC="${2:?}"; shift ;;
        --meta)         TTSP_META="${2:?}"; shift ;;
        --summary-dir)  SUMMARY_DIR="${2:?}"; shift ;;
        --episodes)     TTSP_EPISODES="${2:?}"; shift ;;
        --hermes-home)  HERMES_HOME_HOST="${2:?}"; shift ;;
        --state-dir)    STATE_DIR="${2:?}"; shift ;;
        --searxng-base) SEARXNG_STATS_BASE="${2:?}"; shift ;;
        --help|-h)
            sed -n '3,20p' "$0"; exit 0 ;;
        *)
            echo "usage: $0 [--check] [--quiet] [--no-notify] [fixture options]" >&2
            exit 64 ;;
    esac
    shift
done

derive_paths

LOG="$STATE_DIR/stack-health.log"
LAST_RUN="$STATE_DIR/last-run.json"
ALERT_STATE="$STATE_DIR/alert-state.json"
LOCK="$STATE_DIR/.health.lock"

mkdir -p "$STATE_DIR" 2>/dev/null || true

# ---------------------------------------------------------------------------
# --check: report on the recorded result, probe nothing
# ---------------------------------------------------------------------------
# Deliberately does no I/O against any service. This is the mode another script
# — or a future dashboard — calls to ask "is the box healthy" without adding a
# second round of probes on top of the scheduled ones.

if [ "$MODE" = check ]; then
    if [ ! -s "$LAST_RUN" ]; then
        echo "stack health: NEVER RUN (no $LAST_RUN)" >&2
        exit 5
    fi
    python3 - "$LAST_RUN" "$STALE_CHECK_MIN" <<'PY'
import json, sys, time
path, stale_min = sys.argv[1], int(sys.argv[2])
try:
    d = json.load(open(path))
except Exception as exc:
    print(f"stack health: last-run.json unreadable ({exc})", file=sys.stderr)
    raise SystemExit(4)
age_min = (time.time() - d.get("finished_epoch", 0)) / 60
print("stack health: state=%s  %d ok / %d warn / %d fail  (last run %.0f min ago, %.1fs)"
      % (d.get("state"), d.get("ok_count", -1), d.get("warn_count", -1),
         d.get("fail_count", -1), age_min, d.get("run_seconds", -1)))
for line in d.get("findings", []):
    print("  %-5s %s: %s" % (line.get("severity"), line.get("headline"), line.get("detail")))
if age_min > stale_min:
    print(f"  -> STALE: no run in {age_min:.0f} min (expected every 15)", file=sys.stderr)
    raise SystemExit(4)
raise SystemExit({"ok": 0, "warn": 2, "fail": 3}.get(d.get("state"), 4))
PY
    exit $?
fi

# ---------------------------------------------------------------------------
# one run at a time
# ---------------------------------------------------------------------------
# 15-minute cadence against a run that normally takes ~5 s leaves enormous
# slack, but a wedged curl behind a dead network could still overlap the next
# tick. flock -n makes the overlapping run a no-op instead of piling up.

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "stack health: another run holds the lock — doing nothing" >&2
    exit 75
fi

START_EPOCH=$(date +%s)
NOW_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
TODAY_UTC="$(date -u +%Y-%m-%d)"
DOW_UTC="$(date -u +%u)"          # 1=Mon .. 7=Sun
HHMM_UTC="$(date -u +%H:%M)"

# ---------------------------------------------------------------------------
# finding accumulator
# ---------------------------------------------------------------------------
# Findings are collected as SEVERITY<TAB>HEADLINE<TAB>DETAIL. The HEADLINE is
# the failure CLASS and is what goes in the Telegram subject line and in the
# de-duplication signature — so "SERVICE NOT ANSWERING" firing for a different
# container counts as a new fault and alerts immediately, while the identical
# fault repeating stays silent.

FINDINGS=()
OK_COUNT=0
add_fail() { FINDINGS+=("FAIL"$'\t'"$1"$'\t'"$2"); }
add_warn() { FINDINGS+=("WARN"$'\t'"$1"$'\t'"$2"); }
add_ok()   { OK_COUNT=$((OK_COUNT + 1)); }

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG"; }

# ---------------------------------------------------------------------------
# 0. docker itself
# ---------------------------------------------------------------------------

if ! docker info >/dev/null 2>&1; then
    echo "stack health: docker is not reachable — cannot check anything" >&2
    log "ABORT: docker unreachable"
    exit 4
fi

# ---------------------------------------------------------------------------
# 1. container state: running, healthy, and NOT restarting
# ---------------------------------------------------------------------------
# RestartCount is lifetime history, not current activity. Compare each valid
# observation with the last one under our existing whole-run lock. A change
# fails this observation; the next stable observation clears that finding.
# First sight warns once because no previous observation exists to compare.
#
# EXPECTED is the full inventory. Listing it explicitly rather than iterating
# `docker ps` is the point: a container that has VANISHED is the failure mode
# `docker ps` cannot show you.

EXPECTED=(hermes-1 embedding-server searxng ddgs-search web-extract camofox
          caddy cloudflared ttsp paperless paperless-db paperless-redis
          site stirling)

declare -A C_STATE C_HEALTH C_RESTARTS C_IP

# Kept inline so deployment remains a single script. The tests extract this
# entire production section and provide only Docker and accumulator fixtures.
container_observation() {
    python3 - "$STATE_DIR/container-$1.json" "$(date +%s)" "$2" <<'PY'
import json, os, re, sys, tempfile
from datetime import datetime
path, now, raw = sys.argv[1], int(sys.argv[2]), sys.argv[3]
def emit(st, he, rc, ip, severity, detail):
    print("|".join(map(str, (st, he, rc, ip, severity, detail))))
def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("StartedAt is missing")
    # Docker uses nanosecond RFC3339; datetime accepts and truncates fractions.
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("StartedAt lacks timezone")
    return dt.timestamp()
def validate(d, previous=False):
    if not isinstance(d, dict) or not re.fullmatch(r"[0-9a-f]{64}", d.get("id", "")):
        raise ValueError("invalid container ID")
    if type(d.get("count")) is not int or d["count"] < 0:
        raise ValueError("invalid RestartCount")
    started = timestamp(d.get("started"))
    # Shell date samples whole seconds, Docker StartedAt carries nanoseconds.
    if started >= now + 1:
        raise ValueError("StartedAt is in the future")
    if previous and (type(d.get("observed")) is not int or d["observed"] < 0 or d["observed"] > now):
        raise ValueError("invalid/future observation time")
try:
    docs = json.loads(raw)
    if not isinstance(docs, list) or len(docs) != 1:
        raise ValueError("expected one inspect object")
    obj = docs[0]; state = obj["State"]
    st, restarting = state["Status"], state["Restarting"]
    if st not in ("created", "running", "paused", "restarting", "removing", "exited", "dead") or type(restarting) is not bool:
        raise ValueError("invalid container state")
    health = state.get("Health")
    he = "none" if health is None else health["Status"]
    if he not in ("none", "starting", "healthy", "unhealthy"):
        raise ValueError("invalid health state")
    ip = obj["NetworkSettings"]["Networks"].get("hermes-net", {}).get("IPAddress", "")
    if not isinstance(ip, str) or any(ch in ip for ch in "|\n\r"):
        raise ValueError("invalid network address")
    current = dict(id=obj["Id"], count=obj["RestartCount"], started=state["StartedAt"], observed=now)
    validate(current)
except Exception as exc:
    emit("unknown", "none", 0, "", "INVALID", "inspect data invalid: " + str(exc)); sys.exit(0)
severity, detail = "OK", "stable restart observation"
try:
    with open(path) as stream:
        previous = json.load(stream)
    validate(previous, previous=True)
except FileNotFoundError:
    severity, detail = "WARN", "initial observation; restart history is not yet comparable (RestartCount=%s)" % current["count"]
except Exception as exc:
    severity, detail = "STATE", "restart baseline invalid; re-baselining: " + str(exc)
else:
    if any(current[key] != previous[key] for key in ("id", "count", "started")):
        severity, detail = "EVENT", "restart observation changed: ID %s -> %s, count %s -> %s, StartedAt %s -> %s" % (previous["id"][:12], current["id"][:12], previous["count"], current["count"], previous["started"], current["started"])
# Commit atomically before accepting the observation. A failed write fails the
# check and preserves the old baseline so an event is retried next run.
temporary = None
try:
    with tempfile.NamedTemporaryFile(mode="w", dir=os.path.dirname(path), prefix=".container-", delete=False) as stream:
        temporary = stream.name
        json.dump(current, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path); temporary = None
    directory = os.open(os.path.dirname(path), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
except Exception as exc:
    severity, detail = "STATE", "cannot persist restart baseline: " + str(exc)
finally:
    if temporary:
        os.unlink(temporary)
if restarting or st == "restarting":
    severity, detail = "RESTARTING", "Docker reports active restarting (RestartCount=%s)" % current["count"] + ("; " + detail if severity == "STATE" else "")
emit(st, he, current["count"], ip, severity, detail)
PY
}

for c in "${EXPECTED[@]}"; do
    # Project metadata inside Docker: full inspect includes Config.Env secrets
    # and healthcheck logs and must never be forwarded to a helper's argv.
    if ! info="$(docker inspect --format \
        '[{"Id":{{json .Id}},"RestartCount":{{json .RestartCount}},"State":{"Status":{{json .State.Status}},"Restarting":{{json .State.Restarting}},"StartedAt":{{json .State.StartedAt}}{{with index .State "Health"}},"Health":{"Status":{{json .Status}}}{{end}}},"NetworkSettings":{"Networks":{"hermes-net":{"IPAddress":{{with index .NetworkSettings.Networks "hermes-net"}}{{json .IPAddress}}{{else}}""{{end}}}}}}]' \
        "$c" 2>/dev/null)" || [ -z "$info" ]; then
        add_fail "CONTAINER MISSING" "$c cannot be inspected — missing container or Docker inspect failed"
        C_STATE[$c]=absent; C_HEALTH[$c]=none; C_RESTARTS[$c]=0; C_IP[$c]=""
        continue
    fi
    if ! observation="$(container_observation "$c" "$info")" || [ -z "$observation" ]; then
        observation='unknown|none|0||INVALID|restart observation helper failed'
    fi
    IFS='|' read -r st he rc ip restart_severity restart_detail <<<"$observation"
    C_STATE[$c]="$st"; C_HEALTH[$c]="$he"; C_RESTARTS[$c]="$rc"; C_IP[$c]="$ip"
    case "$restart_severity" in
        INVALID) add_fail "CONTAINER STATE UNREADABLE" "$c $restart_detail"; continue ;;
        STATE) add_fail "CONTAINER RESTART BASELINE UNREADABLE" "$c $restart_detail" ;;
        WARN) add_warn "CONTAINER RESTART BASELINE INITIAL" "$c $restart_detail" ;;
        EVENT) add_fail "CONTAINER RESTART OBSERVED" "$c $restart_detail" ;;
        RESTARTING) add_fail "CONTAINER RESTARTING" "$c $restart_detail" ;;
        OK) ;;
        *) add_fail "CONTAINER STATE UNREADABLE" "$c invalid observation result"; continue ;;
    esac

    if [ "$st" != running ]; then
        add_fail "CONTAINER DOWN" "$c is '$st' (restarts=$rc)"
        continue
    fi
    if [ "$he" = unhealthy ]; then
        add_fail "CONTAINER UNHEALTHY" "$c reports its own healthcheck as unhealthy"
    elif [ "$he" = starting ]; then
        add_warn "CONTAINER STARTING" "$c is still in its healthcheck start period"
    fi
    add_ok
done

# ---------------------------------------------------------------------------
# 2. service reachability, ASSERTING ON CONTENT
# ---------------------------------------------------------------------------
# Two vantage points, and neither needs `docker exec`:
#   host   — the three services published on 127.0.0.1
#   netip  — the container's hermes-net address, resolved fresh from `docker
#            inspect` every run. Docker bridge networks are routable from the
#            host, so an unpublished service is still probeable without going
#            through hermes-1. That matters: if the probe ran inside hermes-1,
#            a wedged gateway would black out every other service's result too.
#
# expect_body is a plain substring, matched with bash's own [[ == * ]]. There is
# no `grep -q` anywhere in this script: `grep -q` in a pipeline under pipefail
# exits at the first match, the producer takes SIGPIPE, and the pipeline reports
# failure on success. That has already caused two real bugs on this box.
#
# Notes on the two endpoints that look wrong but are not:
#   ttsp       has no /health at all — /feed.xml IS its health endpoint, and
#              asserting on <rss also proves the feed generator works.
#   paperless  answers 302 on /api/ (redirect to login). That is CORRECT and
#              proves Django is serving; a 200 there would mean auth is off.

probe() {
    # probe <label> <container> <host|netip> <port> <path> <expect_http> <expect_body> [host_header]
    local label="$1" cont="$2" how="$3" port="$4" path="$5" want="$6" body_want="$7" hosthdr="${8:-}"
    local base code body tmp

    [ "${C_STATE[$cont]:-absent}" = running ] || return 0   # already reported as down

    if [ "$how" = host ]; then
        base="http://127.0.0.1:$port"
    else
        if [ -z "${C_IP[$cont]:-}" ]; then
            add_fail "SERVICE UNADDRESSABLE" "$label ($cont) has no hermes-net address"
            return 0
        fi
        base="http://${C_IP[$cont]}:$port"
    fi

    tmp="$(mktemp)"
    if [ -n "$hosthdr" ]; then
        code="$(curl -s -m 5 -o "$tmp" -w '%{http_code}' -H "Host: $hosthdr" "$base$path" 2>/dev/null)"
    else
        code="$(curl -s -m 5 -o "$tmp" -w '%{http_code}' "$base$path" 2>/dev/null)"
    fi
    body="$(head -c 4096 "$tmp" 2>/dev/null)"
    rm -f "$tmp"

    if [ "$code" != "$want" ]; then
        if [ "$code" = 000 ]; then
            add_fail "SERVICE NOT ANSWERING" \
                "$label: no response from $base$path (container is up, the service inside it is not)"
        else
            add_fail "SERVICE NOT ANSWERING" \
                "$label: $base$path returned HTTP $code, expected $want"
        fi
        return 0
    fi
    if [ -n "$body_want" ] && [[ "$body" != *"$body_want"* ]]; then
        add_fail "SERVICE ANSWERING WRONG" \
            "$label: HTTP $want from $base$path but the body does not contain '$body_want' — $label is answering, but with the WRONG content. (Failure class: 200-with-wrong-body. SearXNG's /healthz once did this during a total outage; that name is the textbook example, NOT the subject of this alert — the subject is named at the start of this line.)"
        return 0
    fi
    add_ok
}

probe "hermes web UI"   hermes-1         netip 9120 /            200 "Hermes"
probe "embedding-server" embedding-server host  8766 /health     200 '"status":"ok"'
# LIVENESS ONLY, and deliberately so. SearXNG's /healthz returned 200 in 3 ms
# throughout a total engine outage, so this line proves the process is up and
# NOTHING about whether search works. The content-level assertion lives in
# section 7 below, which reads search-watchdog's own verdict (status, age and
# fault signature) — that is where "web search is degraded" is decided. Do not
# "strengthen" this probe by giving it a query: an extra search on every health
# run is how this host's egress IP got blocked before.
probe "searxng (liveness)" searxng       host  8888 /healthz     200 "OK"
probe "ddgs-search"      ddgs-search     netip 8080 /healthz     200 '"service": "ddgs-search"'
probe "web-extract"      web-extract     netip 3002 /health      200 '"service": "web-extract"'
probe "camofox"          camofox         netip 9377 /            200 '"ok":true'
probe "ttsp"             ttsp            netip 8081 /feed.xml    200 "<rss"
probe "caddy"            caddy           netip 80   /edge-health 200 "edge ok" podcast.hermeshosting.cc
# The corporate site's OWN health endpoint, not the edge's. Deliberately
# /site-health and not /edge-health: this asserts the `site` container is
# serving, independently of whether the edge in front of it is healthy.
probe "site"             site            netip 80   /site-health 200 "site ok"
# Content assertion on the real page, so an empty or clobbered public/ directory
# is caught. A 200 serving nothing is the failure mode a port check cannot see.
probe "site (index)"     site            netip 80   /            200 "MedSer PBAS CommV"

# The real corporate domain, served from here since 12 Sep 2026 (phase 2 of the
# medserpbas.eu zone move). Probed through caddy with a Host header rather than
# over the public internet, so this still passes when the tunnel or Cloudflare
# is the thing that is broken — and fails when the Caddyfile hostname list is.
# That list is the failure mode with no other symptom: drop a hostname and Caddy
# keeps answering happily for the others while the real domain 404s.
# Both hostnames are asserted separately because a typo in either is invisible
# from the other.
probe "caddy (medserpbas.eu)"     caddy netip 80 / 200 "MedSer PBAS CommV" medserpbas.eu
probe "caddy (www.medserpbas.eu)" caddy netip 80 / 200 "MedSer PBAS CommV" www.medserpbas.eu

# Stirling-PDF, added 22 Aug 2026. Probed TWICE, because it is reached by two
# different paths that fail independently and the difference is the diagnosis:
#
#   host  8085 -> the published binding (Tailscale + loopback). This is the
#                 path a browser on the MacBook or the iPhone uses. It breaks
#                 on its own if the tailscale0 bind loses the boot race.
#   netip 8080 -> the hermes-net path, `stirling:8080`. This is the path the
#                 Hermes agents use via ~/.hermes/bin/pdf-tool, and the one
#                 hermes-2 and hermes-3 will use. It breaks on its own if the
#                 container falls off hermes-net.
#
# Both assert on the body, not just the status code: the React frontend answers
# 200 with the SPA shell for almost any path, so a code-only check would pass
# while the Java backend behind it was dead.
probe "stirling (published)" stirling   host  8085 /api/v1/info/status 200 '"status":"UP"'
probe "stirling (hermes-net)" stirling  netip 8080 /api/v1/info/status 200 '"status":"UP"'

# paperless-db and paperless-redis are on paperless_paperless-net, not
# hermes-net, and speak Postgres and RESP rather than HTTP. Their compose
# healthchecks are `pg_isready` and `redis-cli ping`, which ARE content
# assertions — they prove the database answers a query, not merely that a port
# accepts a connection. Section 1 already asserts on those. Adding a raw TCP
# connect here would be strictly weaker, so it is deliberately not done.

# cloudflared publishes no port and has no healthcheck. Its real assertion is
# section 3: if the public hostname serves the feed, the tunnel is up. There is
# no cheaper honest test.

# ---------------------------------------------------------------------------
# 3. the public edge, end to end
# ---------------------------------------------------------------------------
# Two probes because they fail differently and the difference is the diagnosis:
#   /edge-health  is answered by Caddy itself     -> proves cloudflared -> Caddy
#   /feed.xml     is answered by TTSP behind it   -> proves the whole chain
# A 502 on the second with a 200 on the first means the tunnel is fine and TTSP
# is the problem; no response at all on either means the tunnel is down.
# The feed body is captured here and reused in section 6 — one fetch, not two.

EDGE_FEED=""
edge_code="$(curl -s -m 20 -o /dev/null -w '%{http_code}' https://podcast.hermeshosting.cc/edge-health 2>/dev/null)"
if [ "$edge_code" != 200 ]; then
    add_fail "PUBLIC EDGE DOWN" \
        "https://podcast.hermeshosting.cc/edge-health returned '$edge_code' — the Cloudflare tunnel or Caddy is not serving"
else
    add_ok
    edge_tmp="$(mktemp)"
    feed_code="$(curl -s -m 30 -o "$edge_tmp" -w '%{http_code}' https://podcast.hermeshosting.cc/feed.xml 2>/dev/null)"
    EDGE_FEED="$(cat "$edge_tmp" 2>/dev/null)"
    rm -f "$edge_tmp"
    if [ "$feed_code" != 200 ]; then
        add_fail "PODCAST FEED NOT PUBLIC" \
            "edge-health is 200 but /feed.xml returned '$feed_code' — the tunnel is fine, TTSP behind it is not"
    elif [[ "$EDGE_FEED" != *"<rss"* ]] || [[ "$EDGE_FEED" != *"<enclosure"* ]]; then
        add_fail "PODCAST FEED EMPTY" \
            "the public feed is 200 but contains no <enclosure> — it is serving a feed with no episodes in it"
    else
        add_ok
    fi
fi

# ---------------------------------------------------------------------------
# 4. Hermes itself: is the AGENT running, not just the container?
# ---------------------------------------------------------------------------
# This is the documented failure that looks like health from every angle: the
# container reports "Up", ttyd and the dashboard and Honcho all answer, and
# there is no `gateway run` process at all — Telegram and email simply go
# silent. `docker top` lists the container's processes using the HOST's ps, so
# it needs no docker exec and cannot accidentally match this script's own
# command line the way `pgrep -f` would.

top_out="$(docker top hermes-1 -o pid,args 2>/dev/null)"
if [ "${C_STATE[hermes-1]:-absent}" = running ]; then
    if [[ "$top_out" != *"gateway run"* ]]; then
        add_fail "HERMES GATEWAY NOT RUNNING" \
            "hermes-1 is Up but there is no 'hermes gateway run' process. Telegram and email are silent. Recovery: cd /opt/stacks/hermes && docker compose restart"
    else
        add_ok
    fi
fi

# The gateway process can exist while its internal scheduler is stuck, so check
# the scheduler's OUTPUT too — same principle as everything else here. Job
# 66f4fb805e40 runs every 10 minutes, so the newest last_run_at across all
# enabled cron jobs must never be more than ~25 minutes old. That single number
# is a heartbeat for the whole `hermes cron` subsystem.
#
# The same pass reports any ENABLED job whose last run ended in error. Disabled
# jobs are ignored on purpose: a8b382f308fc has been sitting on last_status=error
# since it was switched off on 7 August, and alerting about a job the user
# deliberately turned off is precisely the noise this script must not make.

if [ -s "$CRON_JOBS" ]; then
    cron_report="$(python3 - "$CRON_JOBS" <<'PY' 2>/dev/null
import json, sys
from datetime import datetime, timezone
try:
    data = json.load(open(sys.argv[1], encoding="utf-8-sig"))
except Exception as exc:
    print(f"UNREADABLE\t{exc}"); raise SystemExit(0)
now = datetime.now(timezone.utc)
newest, errors = None, []
for job in data.get("jobs", []):
    if not job.get("enabled", True):
        continue
    if (job.get("schedule") or {}).get("kind") != "cron":
        continue          # one-shot reminders have no cadence to be late against
    ts = job.get("last_run_at")
    if ts:
        try:
            dt = datetime.fromisoformat(ts)
            newest = dt if newest is None or dt > newest else newest
        except ValueError:
            pass
    if job.get("last_status") not in (None, "ok"):
        errors.append("%s (%s) last_status=%s: %s"
                      % (job.get("id"), (job.get("name") or "")[:40],
                         job.get("last_status"), str(job.get("last_error"))[:160]))
age = int((now - newest).total_seconds() / 60) if newest else -1
print("AGE\t%d" % age)
for e in errors:
    print("ERROR\t%s" % e)
PY
)"
    while IFS=$'\t' read -r kind rest; do
        case "$kind" in
            UNREADABLE)
                add_fail "HERMES CRON UNREADABLE" "cron/jobs.json could not be parsed: $rest" ;;
            AGE)
                if [ "$rest" -lt 0 ]; then
                    add_fail "HERMES CRON SCHEDULER STALLED" "no enabled cron job has ever recorded a run"
                elif [ "$rest" -gt 25 ]; then
                    add_fail "HERMES CRON SCHEDULER STALLED" \
                        "newest hermes cron run is ${rest} min old; job 66f4fb805e40 runs every 10 min, so the in-container scheduler has stopped"
                else
                    add_ok
                fi ;;
            ERROR)
                add_fail "HERMES CRON JOB IN ERROR" "$rest" ;;
        esac
    done <<<"$cron_report"
else
    add_warn "HERMES CRON UNREADABLE" "$CRON_JOBS is missing or empty"
fi

# ---------------------------------------------------------------------------
# 5. backups — by calling the scripts that already know
# ---------------------------------------------------------------------------
# Both backup scripts already answer "is the backup real and recent" properly,
# with their own staleness horizons. Re-deriving that here would give two
# sources of truth that can disagree. So: call them, aggregate their exit codes,
# and quote their own one-line summary as the detail.

check_backup() {
    local label="$1" script="$2" out rc
    if [ ! -x "$script" ]; then
        add_warn "BACKUP SCRIPT MISSING" "$label: $script is not executable"
        return 0
    fi
    out="$("$script" --check 2>&1)"
    rc=$?
    case "$rc" in
        0)  add_ok ;;
        2)  add_fail "BACKUP MISSING ($label)" "no backup generations exist at all — ${out//$'\n'/ }" ;;
        3)  add_fail "BACKUP FAILED ($label)"  "the last recorded run did not succeed — ${out//$'\n'/ }" ;;
        4)  add_fail "BACKUP STALE ($label)"   "newest backup is past its staleness horizon — ${out//$'\n'/ }" ;;
        75) add_ok ;;   # a backup is running right now; not a fault
        *)  add_warn "BACKUP CHECK ODD ($label)" "--check exited $rc: ${out//$'\n'/ }" ;;
    esac
}
check_backup "hermes state" "$HERMES_BACKUP"
check_backup "paperless"    "$PAPERLESS_BACKUP"

# ---------------------------------------------------------------------------
# 6. STALE OUTPUT — the reason this script exists
# ---------------------------------------------------------------------------
# Everything above can be green while the box is quietly producing yesterday's
# work. These are the checks that look at the artefact instead of the exit code.
#
# Gating: weekdays only (the jobs are `1-5`), and only after DEADLINE_UTC, so a
# 15-minute cadence does not alarm at 02:00 while the summariser is still
# running. Weekend and pre-deadline runs skip these silently.

DAILY_DUE=0
if [ "$DOW_UTC" -le 5 ] && [[ "$HHMM_UTC" > "$DEADLINE_UTC" ]]; then
    DAILY_DUE=1
fi

if [ "$DAILY_DUE" = 1 ]; then
    SUMMARY="$SUMMARY_DIR/richtlijn-$TODAY_UTC.txt"

    # ---- 6a. today's summary exists and is not a stub ---------------------
    # If this is missing there SHOULD be no episode today, and the summariser is
    # the thing that broke — that distinction is the whole reason the message
    # names the summariser rather than the podcast.
    if [ ! -f "$SUMMARY" ]; then
        add_fail "RICHTLIJN SUMMARY MISSING" \
            "$SUMMARY does not exist. The 04:15 UTC summariser (b28fc7d18fa1) produced nothing today, so any episode published today was built from an older source."
    elif [ "$(stat -c %s "$SUMMARY")" -lt "$MIN_SUMMARY_BYTES" ]; then
        add_fail "RICHTLIJN SUMMARY TRUNCATED" \
            "$SUMMARY is only $(stat -c %s "$SUMMARY") bytes (floor $MIN_SUMMARY_BYTES). The summariser wrote a stub."
    else
        add_ok

        # ---- 6b. the newest episode, and how it relates to that summary ---
        # This is the exact 17 August 2026 defect, caught from host files alone.
        #
        # On that day the podcast job ran at 01:30 UTC, the summariser did not
        # finish writing richtlijn-2026-08-17.txt until 01:55 UTC, and the
        # episode was therefore built from richtlijn-2026-08-14.txt — while both
        # jobs recorded `ok`. The fingerprint is arithmetic and needs no log
        # parsing at all: **the episode is OLDER than the summary it claims to
        # be made from.** An episode can only be derived from a file that
        # already existed when it was rendered.
        #
        # (A summary regenerated by hand later in the day trips this too. That
        # is why both timestamps are printed in the message: a manual rerun is
        # obvious at a glance, and being told about it is not harmful.)
        meta_report="$(python3 - "$TTSP_META" "$SUMMARY" "$TODAY_UTC" "$TTSP_EPISODES" <<'PY' 2>/dev/null
import json, os, sys
from datetime import datetime, timezone
meta_path, summary, today, epdir = sys.argv[1:5]
def out(k, v): print(f"{k}\t{v}")
try:
    entries = json.load(open(meta_path, encoding="utf-8"))
except Exception as exc:
    out("UNREADABLE", f"{meta_path}: {exc}"); raise SystemExit(0)
if not entries:
    out("NOEPISODES", meta_path); raise SystemExit(0)

def pub(e):
    try:
        return datetime.fromisoformat(e["published"])
    except Exception:
        return datetime.fromtimestamp(0, timezone.utc)
newest = max(entries, key=pub)
p = pub(newest)
if p.tzinfo is None:
    p = p.replace(tzinfo=timezone.utc)
out("NEWEST", "%s|%s|%s|%s" % (newest.get("filename"), p.isoformat(),
                               newest.get("size_bytes"), (newest.get("title") or "")[:70]))
out("PUBDATE", p.astimezone(timezone.utc).strftime("%Y-%m-%d"))

st = os.stat(summary)
out("GAPSEC", "%d" % (p.timestamp() - st.st_mtime))
out("SUMMARYMTIME", datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat())

fn = newest.get("filename") or ""
path = os.path.join(epdir, fn)
if not fn:
    out("NOFILENAME", "the newest feed entry has no filename field")
elif not os.path.isfile(path):
    out("AUDIOMISSING", path)
else:
    actual = os.path.getsize(path)
    out("AUDIOSIZE", "%d|%s" % (actual, newest.get("size_bytes")))
PY
)"

        EP_FILENAME=""; EP_PUBDATE=""; EP_PUB=""; EP_TITLE=""
        GAPSEC=""; SUMMARY_MTIME=""
        while IFS=$'\t' read -r kind rest; do
            case "$kind" in
                UNREADABLE)
                    add_fail "PODCAST META UNREADABLE" "$rest" ;;
                NOEPISODES)
                    add_fail "PODCAST FEED EMPTY" "$rest contains no episodes at all" ;;
                NEWEST)
                    IFS='|' read -r EP_FILENAME EP_PUB _epsize EP_TITLE <<<"$rest" ;;
                PUBDATE)
                    EP_PUBDATE="$rest" ;;
                GAPSEC)
                    GAPSEC="$rest" ;;
                SUMMARYMTIME)
                    SUMMARY_MTIME="$rest" ;;
                NOFILENAME)
                    add_fail "PODCAST AUDIO MISSING" "$rest" ;;
                AUDIOMISSING)
                    add_fail "PODCAST AUDIO MISSING" \
                        "the newest feed entry points at $rest, which does not exist — the feed advertises an episode nobody can download" ;;
                AUDIOSIZE)
                    IFS='|' read -r _actual _claimed <<<"$rest"
                    if [ "${_actual:-0}" -lt 1000000 ]; then
                        add_fail "PODCAST AUDIO BROKEN" \
                            "the newest episode's mp3 is only ${_actual} bytes — a real episode here is ~20-25 MB"
                    elif [ "${_actual}" != "${_claimed}" ]; then
                        add_warn "PODCAST AUDIO SIZE MISMATCH" \
                            "mp3 on disk is ${_actual} bytes, meta.json claims ${_claimed}"
                    else
                        add_ok
                    fi ;;
            esac
        done <<<"$meta_report"

        # Is there an episode for today at all?
        if [ -n "$EP_PUBDATE" ] && [ "$EP_PUBDATE" != "$TODAY_UTC" ]; then
            add_fail "PODCAST EPISODE MISSING" \
                "today's summary exists but the newest episode is from $EP_PUBDATE ($EP_TITLE). The 05:15 UTC podcast job (8d3c80e70295) produced nothing today."
        elif [ -n "$EP_PUBDATE" ]; then
            add_ok
        fi

        # THE 17 AUGUST TRIPWIRE.
        if [ -n "$GAPSEC" ] && [ "$GAPSEC" -lt 0 ] && [ "$EP_PUBDATE" = "$TODAY_UTC" ]; then
            add_fail "PODCAST EPISODE PREDATES ITS SOURCE" \
                "today's episode ($EP_FILENAME, published $EP_PUB) was rendered $(( -GAPSEC / 60 )) min BEFORE $SUMMARY was written ($SUMMARY_MTIME). It cannot have been made from today's summary. This is the 2026-08-17 defect: both jobs report ok, the episode is built from an older richtlijn. Do not share the episode."
        elif [ -n "$GAPSEC" ]; then
            add_ok
        fi

        # Was it actually PUBLISHED? Generated-but-not-served is its own class
        # of silent failure, and the public feed body was already fetched above.
        if [ -n "$EP_FILENAME" ] && [ -n "$EDGE_FEED" ]; then
            if [[ "$EDGE_FEED" != *"$EP_FILENAME"* ]]; then
                add_fail "PODCAST EPISODE NOT PUBLISHED" \
                    "$EP_FILENAME is the newest entry in meta.json but does not appear in the public feed at podcast.hermeshosting.cc"
            else
                add_ok
            fi
        fi
    fi

    # ---- 6c. the existing report-based tripwire ---------------------------
    # ttsp-richtlijn-watchdog.py already reads the podcast job's own saved
    # report and checks which richtlijn file it names. That is a DIFFERENT
    # signal from the timestamp arithmetic above — it catches a run that was
    # late enough for the timestamps to look fine but still worked from the
    # wrong file. Reuse it rather than reimplement it.
    #
    # It normally runs inside hermes-1 as job 74d146df2027 at 07:00 UTC, but it
    # honours $HERMES_HOME and reads only bind-mounted files, so the host can
    # run it directly with the host path — which means this check keeps working
    # even when hermes-1 is the thing that is broken.
    #
    # Its contract: always exits 0, and ANY stdout is the alarm.
    if [ -r "$RICHTLIJN_WATCHDOG" ]; then
        wd_out="$(HERMES_HOME="$HERMES_HOME_HOST" python3 "$RICHTLIJN_WATCHDOG" 2>&1)"
        if [ -n "$wd_out" ]; then
            add_fail "PODCAST SOURCE MISMATCH (job report)" "${wd_out//$'\n'/ | }"
        else
            add_ok
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 7. search — READ the existing watchdog, never run a query
# ---------------------------------------------------------------------------
# search-watchdog.py runs hourly at minute 17 inside hermes-1 and deliberately
# issues exactly ONE query per hour, because cumulative probing already got this
# host's egress IP blocked. This section therefore asserts on the watchdog's
# state file and log line and issues no query of its own.
#
# Three separate things, and they get three separate headlines on purpose:
#   * the watchdog itself has stopped running   (nobody is watching search)
#   * the watchdog says search is degraded      (search is broken)
#   * Tavily credits are running out            (the safety net is draining)
# The third one gets its own headline because the existing watchdog already
# does that for the same reason: filing a credit warning under "search
# degraded" teaches you to ignore a message that is not about search failing.

if [ -s "$SEARCH_STATE" ]; then
    sw="$(python3 - "$SEARCH_STATE" <<'PY' 2>/dev/null
import json, sys
from datetime import datetime, timezone
d = json.load(open(sys.argv[1]))
last = d.get("last_check_at")
age = -1
if last:
    try:
        dt = datetime.fromisoformat(last)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = int((datetime.now(timezone.utc) - dt).total_seconds() / 60)
    except ValueError:
        pass
print("%s\t%d\t%s" % (d.get("status", "unknown"), age, d.get("signature", "")))
PY
)"
    IFS=$'\t' read -r sw_status sw_age sw_sig <<<"$sw"
    # Hourly at :17, so 130 minutes allows one missed cycle before complaining —
    # a single skipped run during a gateway restart is not worth a message.
    if [ "${sw_age:--1}" -lt 0 ] || [ "${sw_age}" -gt 130 ]; then
        add_fail "SEARCH WATCHDOG STALE" \
            "search-watchdog last checked ${sw_age} min ago (runs hourly at :17). Nobody is watching web search."
    elif [ "$sw_status" != ok ]; then
        add_fail "WEB SEARCH DEGRADED" \
            "search-watchdog reports status=$sw_status faults=[$sw_sig]. Detail: tail -5 $SEARCH_LOG"
    else
        add_ok
    fi
else
    add_warn "SEARCH WATCHDOG STALE" "$SEARCH_STATE is missing — the search watchdog has never written state"
fi

# Tavily credits, read off the watchdog's own last log line. Its own alert
# threshold is 80% of 1000; the same number is used here so the two never
# disagree about what "low" means.
if [ -s "$SEARCH_LOG" ]; then
    # `tail` on a FILE, not through a pipe: no SIGPIPE, no pipefail trap.
    last_line="$(tail -n 1 "$SEARCH_LOG")"
    credits="${last_line##*credits=}"
    credits="${credits%% *}"
    used="${credits%%/*}"; limit="${credits##*/}"
    if [[ "$used" =~ ^[0-9]+$ ]] && [[ "$limit" =~ ^[0-9]+$ ]] && [ "$limit" -gt 0 ]; then
        if [ $(( used * 100 / limit )) -ge 80 ]; then
            add_warn "TAVILY CREDITS LOW" \
                "$used/$limit credits used ($(( used * 100 / limit ))%). Tavily is the paid safety net behind SearXNG and web-extract draws on the same pool; when it runs out a SearXNG failure becomes a total search failure."
        else
            add_ok
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 7b. the SearXNG engine POOL — read the counters, still never run a query
# ---------------------------------------------------------------------------
# Section 7 asks "is search working right now", from one query an hour. This
# asks a different question that no probe on this box could answer before:
# **how many engines are still capable of answering at all?**
#
# On 29 August 2026 the answer was four of ten, and nothing knew. duckduckgo
# was 90% CAPTCHA, startpage 95%, infospace and wikidata 100% HTTP 403, and
# brave was 429-throttled on three requests in four. /healthz returned 200 the
# whole time and section 7 stayed green, because the surviving engines were
# still enough to answer the watchdog's one query. Repairing the pool moved the
# eval set from MRR 0.3697 / recall@8 0.640 to 0.5547 / 0.840 — five more
# queries answered out of 25, a bigger jump than every ranking change ever
# measured here put together. A pool decays quietly and gives back more than
# tuning does, so it is worth a check of its own.
#
# TWO FREE READS, NO SEARCH. /config lists which engines are supposed to be
# enabled; /stats/errors is SearXNG's own accumulated error counter. Neither
# issues a query, so this section obeys the same rule as the rest of the script
# — see the header. Do not "improve" it by searching.
#
# THE CLASSES ARE NOT EQUIVALENT, AND THIS IS THE WHOLE DESIGN:
#
#   403 / AccessDenied  is an access DECISION. It does not heal, it does not
#                       back off, it will still be 403 next month. -> FAIL.
#   CAPTCHA             is a LOUD failure. SearXNG suspends the engine itself,
#                       so it self-limits to near-zero cost and it recovers on
#                       its own. duckduckgo and startpage are LEFT ENABLED here
#                       for exactly that reason (settings.yml says so). It must
#                       therefore NEVER raise an alarm by itself — it only
#                       counts against the deliverable pool below.
#   429 / TooManyReqs   is throttling: real, but transient. -> WARN, high bar.
#   timeouts            are the known 3.0s engine ceiling, a question Atta has
#                       explicitly left alone. Counted as context, never
#                       alarmed on — this check does not get to re-open a
#                       closed decision.
#
# THE COUNTERS ARE CUMULATIVE SINCE THE CONTAINER STARTED, which cuts both
# ways and both are deliberate. Slow decay accumulates and gets caught; but a
# REPAIRED engine keeps its historical percentage until the counters reset, so
# this check stays red after a genuine fix until searxng is restarted. The
# alert text says so, rather than leaving a reader to wonder why the fix "did
# not take".
#
# THREE HEADLINES, NOT ONE, for the same reason section 7 has three: the
# remedies are different. BLOCKED means edit settings.yml. STARVED means the
# pool is too thin to answer well. THROTTLED means a key or a quota.

if [ "${C_STATE[searxng]:-absent}" = running ]; then
    # Uptime gate: percentages over a handful of requests are noise.
    sx_up_min=-1
    sx_started="$(docker inspect searxng --format '{{.State.StartedAt}}' 2>/dev/null)"
    if [ -n "$sx_started" ]; then
        sx_epoch="$(date -d "$sx_started" +%s 2>/dev/null || true)"
        [ -n "${sx_epoch:-}" ] && sx_up_min=$(( ( $(date +%s) - sx_epoch ) / 60 ))
    fi

    if [ "$sx_up_min" -ge 0 ] && [ "$sx_up_min" -lt "$ENGINE_STATS_MIN_UPTIME_MIN" ]; then
        # Not a fault and not a pass — the same "not due" shape the daily
        # output checks use. Counted ok so the run stays quiet.
        add_ok
    else
        pool="$(python3 - "$SEARXNG_STATS_BASE" "$ENGINE_DEAD_PCT" <<'PY' 2>/dev/null
import json, sys, urllib.request

base, dead_pct = sys.argv[1].rstrip("/"), float(sys.argv[2])


def get(path):
    with urllib.request.urlopen(base + path, timeout=5) as resp:
        return json.load(resp)


cfg = get("/config")
errs = get("/stats/errors")

# Only GENERAL engines can contribute to a web search. An image or science
# engine sitting enabled-and-silent is not a starved pool, and counting it
# would put ~80 engines in the denominator and make the metric meaningless.
general = [e["name"] for e in cfg.get("engines", []) or []
           if e.get("enabled") and "general" in (e.get("categories") or [])]
if not general:
    sys.exit(1)

DENIED = "SearxEngineAccessDeniedException"
CAPTCHA = "SearxEngineCaptchaException"
THROTTLE = "SearxEngineTooManyRequestsException"

denied, captcha, throttled, max_throttle = [], [], [], 0
for name in general:
    agg = {}
    for rec in errs.get(name) or []:
        cls = (rec.get("exception_classname") or "").split(".")[-1]
        agg[cls] = agg.get(cls, 0) + (rec.get("percentage") or 0)
    # An engine can be both blocked and captcha'd. Count it ONCE, under the
    # class that decides the remedy: 403 needs a config change, CAPTCHA does
    # not.
    if agg.get(DENIED, 0) >= dead_pct:
        denied.append(name)
    elif agg.get(CAPTCHA, 0) >= dead_pct:
        captcha.append(name)
    thr = agg.get(THROTTLE, 0)
    if thr > 0:
        throttled.append("%s %d%%" % (name, thr))
        if thr > max_throttle:
            max_throttle = thr

# \x1f, NOT tab: tab is IFS *whitespace*, so bash's `read` collapses a run of
# them and an empty field (no blocked engines — the healthy case!) silently
# shifts every later field left. That bug was live in this section for one
# test run and reported duckduckgo as HTTP 403. The same separator is used
# for the alert-state read further down, for the same reason.
print("\x1f".join([str(len(general)),
                 str(len(general) - len(denied) - len(captcha)),
                 ", ".join(denied), ", ".join(captcha),
                 "; ".join(throttled), str(int(max_throttle))]))
PY
)"
        if [ -z "$pool" ]; then
            add_warn "ENGINE POOL UNREADABLE" \
                "could not read $SEARXNG_STATS_BASE/config and /stats/errors. SearXNG is up and answering /healthz, so this is the stats endpoints, not the service — but while it lasts a starved engine pool would go unnoticed."
        else
            IFS=$'\x1f' read -r eg_total eg_deliver eg_denied eg_captcha eg_thr eg_maxthr <<<"$pool"
            eg_ctx="pool: $eg_deliver of $eg_total enabled general engines can deliver"
            [ -n "$eg_captcha" ] && eg_ctx="$eg_ctx; CAPTCHA (self-healing, not a fault): $eg_captcha"
            eg_reset="Counters are cumulative since searxng started ${sx_up_min} min ago — after repairing an engine, \`cd /opt/stacks/searxng && docker compose restart searxng\` to reset them, or this check stays red on the old history."

            # 1. blocked engines — the class that never recovers by itself
            if [ -n "$eg_denied" ]; then
                add_fail "SEARCH ENGINE BLOCKED" \
                    "$eg_denied returning HTTP 403 on at least ${ENGINE_DEAD_PCT}% of requests. A 403 is an access decision, not a rate limit: it will not recover on its own and the engine is dead weight until it is disabled or replaced. That is how infospace and wikidata were lost on 29 Aug 2026 — named as PRECEDENT, not as the subject of this alert; the engines at fault are named at the start of this line. Edit /opt/stacks/searxng/searxng/settings.yml. $eg_ctx. $eg_reset"
            else
                add_ok
            fi

            # 2. the pool floor — the headline number
            if   [ "${eg_deliver:-0}" -lt "$ENGINE_POOL_FAIL" ]; then
                add_fail "SEARCH ENGINE POOL STARVED" \
                    "only $eg_deliver of $eg_total enabled general engines can still deliver results. Search will return thin or empty result sets and the Tavily paid fallback will absorb the difference silently. Diagnose with $SEARXNG_STATS_BASE/stats/errors and monitor/engine_share.py — do NOT diagnose by running searches, that is what got this host's IP blocked. $eg_ctx."
            elif [ "${eg_deliver:-0}" -lt "$ENGINE_POOL_WARN" ]; then
                add_warn "SEARCH ENGINE POOL STARVED" \
                    "$eg_deliver of $eg_total enabled general engines can still deliver. The healthy baseline measured on this box is six. Not yet breaking search, but the pool is thinning and recall degrades before anything reports a failure. $eg_ctx."
            else
                add_ok
            fi

            # 3. throttling — transient, so a warning and a high bar
            if [ "${eg_maxthr:-0}" -ge "$ENGINE_THROTTLE_WARN_PCT" ]; then
                add_warn "SEARCH ENGINE THROTTLED" \
                    "rate-limited: $eg_thr. Above ${ENGINE_THROTTLE_WARN_PCT}% this is a quota or a key, not weather. Note \`google cse\` runs on a borrowed, revocable shared CX token and \`braveapi\` on BRAVE_SEARCH_API_KEY — check the Brave dashboard for the real monthly cap, which its own quota headers do not report honestly. $eg_ctx."
            else
                add_ok
            fi
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 8. the uplink — via netwatch, with our own thresholds
# ---------------------------------------------------------------------------
# See the NET_* constants above for why netwatch's own exit 4 is not used
# directly. Exit 2 and 3 ARE used directly: they mean the every-minute cron job
# has stopped recording, which is a fact about the scheduler, not about weather.

if [ -x "$NETWATCH" ]; then
    net_out="$("$NETWATCH" --check 24 2>&1)"
    net_rc=$?
    case "$net_rc" in
        2) add_fail "NETWATCH NOT RECORDING" "no samples in 24 h — the every-minute netwatch cron job is not running" ;;
        3) add_fail "NETWATCH NOT RECORDING" "newest sample is over 30 min old — the every-minute netwatch cron job has stopped" ;;
        *)
            # Parse netwatch's own summary rather than re-reading its JSONL, so
            # there is still one implementation of what a sample means.
            lossy="$(sed -n 's/.*minutes with any loss: \([0-9]*\)\/\([0-9]*\).*/\1 \2/p' <<<"$net_out")"
            dnsf="$(sed  -n 's/.*dns *: \([0-9]*\) minutes with BOTH.*/\1/p'            <<<"$net_out")"
            tlsf="$(sed  -n 's/.*imap tls *: \([0-9]*\)\/[0-9]*.*/\1/p'                 <<<"$net_out")"
            read -r n_lossy n_total <<<"${lossy:-0 0}"
            if [ "${n_total:-0}" -gt 0 ] && \
               [ $(( ${n_lossy:-0} * 100 / n_total )) -ge "$NET_LOSS_MINUTES_WARN_PCT" ]; then
                add_warn "UPLINK DEGRADED" \
                    "$n_lossy of $n_total sampled minutes showed packet loss in the last 24 h. Starlink; correlate with: $NETWATCH --correlate 24"
            elif [ "${dnsf:-0}" -ge "$NET_DNS_FAIL_WARN" ]; then
                add_warn "UPLINK DEGRADED" "$dnsf minutes with BOTH pinned resolvers failing in 24 h"
            elif [ "${tlsf:-0}" -ge "$NET_TLS_FAIL_WARN" ]; then
                add_warn "UPLINK DEGRADED" "$tlsf IMAP TLS greeting probes failed in 24 h — expect email timeouts in errors.log"
            else
                add_ok
            fi ;;
    esac
fi

# ---------------------------------------------------------------------------
# 9. Paperless: is the consume pipeline actually consuming?
# ---------------------------------------------------------------------------
# Paperless answers /api/ and reports healthy even when its consumer worker has
# stopped — the files just sit in consume/ forever. Same shape of bug as the
# podcast: the service reports fine while producing nothing. A file older than
# two hours in consume/ means it was never ingested (OCR here is capped at one
# worker precisely to stay out of the 03:00 window, but two hours is far beyond
# even a slow scan).

if [ -d "$PAPERLESS_CONSUME" ]; then
    stuck="$(find "$PAPERLESS_CONSUME" -maxdepth 1 -type f -mmin +120 -printf '%f\n' 2>/dev/null)"
    if [ -n "$stuck" ]; then
        n="$(printf '%s\n' "$stuck" | wc -l)"
        add_fail "PAPERLESS CONSUME BACKLOG" \
            "$n file(s) have sat in $PAPERLESS_CONSUME for over 2 h without being ingested: ${stuck//$'\n'/, }"
    else
        add_ok
    fi
fi

# ---------------------------------------------------------------------------
# 10. headroom
# ---------------------------------------------------------------------------
# Memory is the binding constraint, not disk: 12 GB total against 402 GB free
# on /. `available` is the number that matters — not `free` — because it counts
# reclaimable page cache, and this box legitimately runs with ~6 GB in cache.

read -r _ mem_total mem_used _ _ _ mem_avail < <(free -m | sed -n '2p')
read -r _ swap_total swap_used _          < <(free -m | sed -n '3p')

if [ "${mem_avail:-0}" -lt "$MEM_FAIL_MB" ]; then
    add_fail "MEMORY EXHAUSTED" \
        "only ${mem_avail} MB available of ${mem_total} MB. web-extract peaks ~1 GB parsing PDFs; the next parse will OOM."
elif [ "${mem_avail:-0}" -lt "$MEM_WARN_MB" ]; then
    add_warn "MEMORY LOW" \
        "${mem_avail} MB available of ${mem_total} MB (baseline ~7800 MB). Memory is the binding constraint on this box."
else
    add_ok
fi
# --- swap: measure the FLOW, not the STOCK ---------------------------------
# This check used to be `swap_used > 4000 MB`, reported as "SWAPPING HEAVILY —
# something is over-committing RAM". That is a claim about a RATE derived from
# a measurement of an AMOUNT, and it was wrong.
#
# `swap_used` is a high-water RESIDUE. Linux never pages anything back in
# proactively — a swapped page returns only when something touches it — so the
# number ratchets up during any transient pressure and then stays there for
# weeks. Measured 11 Sep 2026: 4.7 GB sat in swap while si/so were flat ZERO
# across seven samples and memory PSI totalled 18.6 s of stall in 6.5 days of
# uptime (0.003% of wall clock). Nothing was over-committing anything.
#
# 2330 MB of that 4.7 GB belonged to embedding-server, whose RSS was 269 MB: a
# cold ONNX/Python heap untouched for hours between searches, and exactly the
# right thing for the kernel to park on disk. Forcing a rerank faulted 1.4 GB
# back in WITHOUT swap_used dropping — those pages are dead load-time
# allocation the process never reads again. Swap was doing its job. (It is NOT
# the 3 GB cgroup cap either: memory.events showed max 0, peak 2484 MB.)
#
# The old check had fired 124 times running since 9 Sep, which is precisely the
# "watchdog you learn to ignore" this script's header forbids. So ask the three
# questions that have operational answers instead:
#   1. Is the box MOVING pages?          -> pswpin/pswpout delta since last run
#   2. Is reclaim STALLING work?         -> /proc/pressure/memory `some total`
#   3. Is swap about to RUN OUT?         -> stock, but as % and set high
# Both 1 and 2 read cumulative counters, so the delta covers the WHOLE interval
# between runs rather than whatever instant the cron tick happened to land on.

SWAP_COUNTERS="$STATE_DIR/swap-counters"

now_epoch="$(date +%s)"
read -r pswpin_now pswpout_now < <(
    awk '/^pswpin /{i=$2} /^pswpout /{o=$2} END{print i+0, o+0}' /proc/vmstat 2>/dev/null)
# PSI needs CONFIG_PSI; treat its absence as "no stall data", never as a fault.
psi_some_now=0
if [ -r /proc/pressure/memory ]; then
    psi_some_now="$(awk '/^some /{for (n = 1; n <= NF; n++)
        if ($n ~ /^total=/) { sub(/^total=/, "", $n); print $n + 0; exit }}' \
        /proc/pressure/memory 2>/dev/null)"
fi
: "${pswpin_now:=0}" "${pswpout_now:=0}" "${psi_some_now:=0}"

swap_rate_judged=0
if [ -s "$SWAP_COUNTERS" ]; then
    read -r prev_epoch prev_in prev_out prev_psi < "$SWAP_COUNTERS"
    # Insist on four integers before trusting any of them. A half-written or
    # truncated line would otherwise read as prev_out=0 against a valid recent
    # epoch, making the whole since-boot total look like one interval's worth
    # of paging — a 672 MB/min false alarm in the case that was tested here.
    if ! printf '%s\n' "${prev_epoch:-}" "${prev_in:-}" "${prev_out:-}" "${prev_psi:-}" \
         | grep -qvE '^[0-9]+$'; then
        elapsed=$(( now_epoch - prev_epoch ))
        # A reboot resets the counters and a clock step can invert the interval.
        # Either way there is no meaningful rate — re-baseline and stay silent.
        if [ "$elapsed" -gt 0 ] \
           && [ "$pswpout_now"  -ge "$prev_out" ] \
           && [ "$pswpin_now"   -ge "$prev_in" ] \
           && [ "$psi_some_now" -ge "$prev_psi" ]; then
            swap_rate_judged=1
            # 4 KiB pages -> MB is /256. Divide once, at the end, so a small
            # delta does not floor to zero before the interval is applied.
            swap_out_mbmin=$(( (pswpout_now - prev_out) * 60 / (256 * elapsed) ))
            swap_in_mbmin=$(( (pswpin_now  - prev_in)  * 60 / (256 * elapsed) ))
            # PSI total is microseconds of stall; percentage of an interval
            # given in seconds is therefore delta / (elapsed * 10000).
            mem_psi_pct=$(( (psi_some_now - prev_psi) / (elapsed * 10000) ))
        fi
    fi
fi
printf '%s %s %s %s\n' \
    "$now_epoch" "$pswpin_now" "$pswpout_now" "$psi_some_now" >"$SWAP_COUNTERS" 2>/dev/null || true

if [ "$swap_rate_judged" = 1 ]; then
    if [ "${swap_out_mbmin:-0}" -ge "$SWAP_RATE_WARN_MBMIN" ] \
       || [ "${mem_psi_pct:-0}" -ge "$MEM_PSI_WARN_PCT" ]; then
        add_warn "MEMORY THRASHING" \
            "paging out ${swap_out_mbmin} MB/min (in ${swap_in_mbmin} MB/min) and memory reclaim stalled tasks ${mem_psi_pct}% of the last ${elapsed}s. This is real pressure, unlike a large swap_used, which is only residue — ${swap_used} MB is currently parked."
    else
        add_ok
    fi
else
    # First run after install or reboot: baseline written above, nothing to
    # compare against yet. Not a finding.
    add_ok
fi

if [ "${swap_total:-0}" -gt 0 ] \
   && [ $(( swap_used * 100 / swap_total )) -ge "$SWAP_FULL_WARN_PCT" ]; then
    add_warn "SWAP NEARLY FULL" \
        "${swap_used} MB of ${swap_total} MB used ($(( swap_used * 100 / swap_total ))%) — the next memory spike has nowhere to go and will OOM-kill instead."
else
    add_ok
fi

disk_pct="$(df --output=pcent / | tr -dc '0-9\n' | sed -n '2p')"
disk_avail="$(df -h --output=avail / | sed -n '2p' | tr -d ' ')"
if [ "${disk_pct:-0}" -ge "$DISK_FAIL_PCT" ]; then
    add_fail "DISK FULL" "/ is ${disk_pct}% used, ${disk_avail} left"
elif [ "${disk_pct:-0}" -ge "$DISK_WARN_PCT" ]; then
    add_warn "DISK FILLING" "/ is ${disk_pct}% used, ${disk_avail} left. Episodes add ~500 MB/month and have no retention policy yet."
else
    add_ok
fi

# Camofox is the one container with its own hard memory cap (2 GB). Host-level
# `free` cannot see it approaching that cap, because 2 GB out of 12 GB never
# looks alarming from outside; MemPerc from `docker stats` is measured against
# the container's OWN limit, which is the number that matters.
# `docker stats --no-stream` costs ~2 s, which is most of this script's runtime;
# it is worth it because nothing else reveals a per-container cgroup limit.
stats_out="$(timeout 15 docker stats --no-stream --format '{{.Name}} {{.MemPerc}}' 2>/dev/null)"
if [ -n "$stats_out" ]; then
    cam_pct="$(sed -n 's/^camofox *\([0-9.]*\)%.*/\1/p' <<<"$stats_out")"
    cam_int="${cam_pct%%.*}"
    if [[ "$cam_int" =~ ^[0-9]+$ ]] && [ "$cam_int" -ge "$CAMOFOX_CAP_WARN_PCT" ]; then
        add_warn "CAMOFOX NEAR ITS CAP" \
            "camofox is at ${cam_pct}% of its own 2 GB limit — a browser leak, and the container will be OOM-killed inside its cgroup"
    else
        add_ok
    fi
fi

# ---------------------------------------------------------------------------
# verdict
# ---------------------------------------------------------------------------

FAIL_COUNT=0; WARN_COUNT=0
for f in ${FINDINGS+"${FINDINGS[@]}"}; do
    case "$f" in FAIL*) FAIL_COUNT=$((FAIL_COUNT+1)) ;; WARN*) WARN_COUNT=$((WARN_COUNT+1)) ;; esac
done

if   [ "$FAIL_COUNT" -gt 0 ]; then STATE=fail; RC=3
elif [ "$WARN_COUNT" -gt 0 ]; then STATE=warn; RC=2
else                               STATE=ok;   RC=0
fi

# The de-duplication signature is the sorted set of distinct HEADLINES. Details
# (a byte count, a percentage) change between runs on the same underlying fault
# and must not re-trigger; the failure CLASS changing must.
SIGNATURE=""
if [ "${#FINDINGS[@]}" -gt 0 ]; then
    SIGNATURE="$(printf '%s\n' "${FINDINGS[@]}" | cut -f2 | sort -u | paste -sd, -)"
fi

RUN_SECONDS=$(( $(date +%s) - START_EPOCH ))

# ---------------------------------------------------------------------------
# human-readable report
# ---------------------------------------------------------------------------

report() {
    printf 'Afrodite stack health — %s\n' "$NOW_UTC"
    printf 'state=%s  %d checks passed, %d warning(s), %d failure(s)  (%ds)\n' \
        "$STATE" "$OK_COUNT" "$WARN_COUNT" "$FAIL_COUNT" "$RUN_SECONDS"
    if [ "$DAILY_DUE" = 1 ]; then
        printf 'daily-output checks: DUE (weekday, past %s UTC)\n' "$DEADLINE_UTC"
    else
        printf 'daily-output checks: not due (weekend, or before %s UTC)\n' "$DEADLINE_UTC"
    fi
    if [ "${#FINDINGS[@]}" -gt 0 ]; then
        printf '\n'
        printf '%s\n' "${FINDINGS[@]}" | while IFS=$'\t' read -r sev head detail; do
            printf '[%s] %s\n      %s\n' "$sev" "$head" "$detail"
        done
    fi
}

REPORT="$(report)"
[ "$QUIET" = 1 ] || printf '%s\n' "$REPORT"

# ---------------------------------------------------------------------------
# persist: last-run.json, the published copy, and the rotating log
# ---------------------------------------------------------------------------

python3 - "$LAST_RUN" "$STATE" "$SIGNATURE" "$OK_COUNT" "$WARN_COUNT" "$FAIL_COUNT" \
         "$RUN_SECONDS" "$DAILY_DUE" ${FINDINGS+"${FINDINGS[@]}"} <<'PY' || true
import json, sys, time
path, state, signature, ok, warn, fail, secs, due = sys.argv[1:9]
findings = []
for raw in sys.argv[9:]:
    sev, head, detail = (raw.split("\t") + ["", ""])[:3]
    findings.append({"severity": sev, "headline": head, "detail": detail})
json.dump({
    "state": state,
    "signature": signature,
    "finished_epoch": int(time.time()),
    "finished_human": time.strftime("%Y-%m-%d %H:%M:%S"),
    "ok_count": int(ok), "warn_count": int(warn), "fail_count": int(fail),
    "run_seconds": int(secs),
    "daily_output_checks_due": due == "1",
    "findings": findings,
}, open(path, "w"), indent=2)
PY
# The agent inside hermes-1 cannot see /opt/stacks; hand it a copy where it can.
# Never fatal — failing to publish a status must not fail the check.
cp -f "$LAST_RUN" "$PUBLISHED_STATUS" 2>/dev/null || true
chmod 644 "$PUBLISHED_STATUS" 2>/dev/null || true

if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 5242880 ]; then
    mv -f "$LOG" "$LOG.1"
fi
if [ "$STATE" = ok ]; then
    log "ok — $OK_COUNT checks passed, ${RUN_SECONDS}s"
else
    log "$STATE — $OK_COUNT ok / $WARN_COUNT warn / $FAIL_COUNT fail, ${RUN_SECONDS}s — [$SIGNATURE]"
    printf '%s\n' "$REPORT" | sed 's/^/    /' >>"$LOG"
fi

# ---------------------------------------------------------------------------
# notification
# ---------------------------------------------------------------------------
# Telegram is the established channel here, but this is a HOST script: it cannot
# use `hermes cron`'s delivery, and there is no origin chat to inherit. So it
# posts to the Bot API itself with the chat id pinned, using the same
# TELEGRAM_BOT_TOKEN the gateway uses.
#
# THE TOKEN IS NEVER PUT IN argv. Passing it as part of a URL would expose it in
# `ps` output to every user and to every process listing this box records. curl
# -K reads the url line from STDIN instead, so the command line stays clean.
#
# A failure to notify is logged and recorded but never changes the exit code:
# the check's result is a fact about the box, not about whether Telegram was
# reachable — and "Telegram is unreachable" is itself a plausible symptom of the
# outage being reported.

notify() {
    local text="$1" token http
    token="$(awk -F= '$1=="TELEGRAM_BOT_TOKEN"{sub(/^[^=]*=/,""); gsub(/^[ \t]+|[ \t\r]+$/,""); print; exit}' "$ENV_FILE" 2>/dev/null)"
    if [ -z "$token" ]; then
        log "NOTIFY FAILED: no TELEGRAM_BOT_TOKEN in $ENV_FILE"
        return 1
    fi
    # Telegram caps a message at 4096 characters.
    text="${text:0:3900}"
    http="$(printf 'url = "https://api.telegram.org/bot%s/sendMessage"\n' "$token" \
        | curl -s -m 20 -o /dev/null -w '%{http_code}' -K - \
               --data-urlencode "chat_id=$TG_CHAT" \
               --data-urlencode "disable_web_page_preview=true" \
               --data-urlencode "text=$text" 2>/dev/null)"
    if [ "$http" = 200 ]; then
        log "notified Telegram chat $TG_CHAT"
        return 0
    fi
    log "NOTIFY FAILED: Telegram API returned '$http'"
    return 1
}

# Decide whether to speak. Same three rules as search-watchdog.py:
#   new or changed fault  -> speak now
#   same fault, cooldown  -> stay silent until REALERT_HOURS have passed
#   healthy after a fault -> speak once, so silence is never ambiguous

# NOTE THE DELIMITER. These four fields are read with US (0x1f), NOT a tab.
# bash's `read` treats tab as IFS *whitespace*, which means a run of consecutive
# tabs collapses into one delimiter — so the moment `signature` is empty (i.e.
# every healthy run) the remaining fields shift left by one and last_alert_epoch
# gets handed the timestamp. That bug was live in this script for ten minutes
# and produced a Python traceback on every healthy run; 0x1f is not IFS
# whitespace, so empty fields survive.
prev="$(python3 - "$ALERT_STATE" <<'PY' 2>/dev/null
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    d = {}
print("\x1f".join([str(d.get("state") or "ok"), str(d.get("signature") or ""),
                   str(d.get("last_alert_epoch") or 0), str(d.get("since") or "")]))
PY
)"
IFS=$'\x1f' read -r PREV_STATE PREV_SIG PREV_ALERT_EPOCH PREV_SINCE <<<"${prev}"
: "${PREV_STATE:=ok}"; : "${PREV_SIG:=}"; : "${PREV_ALERT_EPOCH:=0}"
# Belt and braces: a corrupted state file must never be able to crash the run
# that is trying to report an outage.
[[ "$PREV_ALERT_EPOCH" =~ ^[0-9]+$ ]] || PREV_ALERT_EPOCH=0

# ---------------------------------------------------------------------------
# A RECOVERY MUST NOT BE AN ARTEFACT OF HAVING STOPPED LOOKING
# ---------------------------------------------------------------------------
# The daily-output checks are only due on weekdays after DEADLINE_UTC. So a run
# that is failing on "PODCAST EPISODE PREDATES ITS SOURCE" at 23:00 UTC turns
# green at 00:00 UTC purely because the calendar rolled over — and would then
# send "RECOVERED — Afrodite stack is healthy again" about a broken episode that
# is still sitting in the public feed. A false all-clear is exactly the class of
# misleading signal this box has been bitten by before.
#
# So: if the ONLY previous faults were daily-output faults and those checks are
# not due this run, the fault is CARRIED FORWARD rather than cleared. No
# recovery message, and the alert state keeps the old signature so the fault is
# re-evaluated honestly the next time the checks are actually due.
DAILY_HEADLINES=(
    "RICHTLIJN SUMMARY MISSING" "RICHTLIJN SUMMARY TRUNCATED"
    "PODCAST EPISODE MISSING" "PODCAST EPISODE PREDATES ITS SOURCE"
    "PODCAST EPISODE NOT PUBLISHED" "PODCAST AUDIO MISSING"
    "PODCAST AUDIO BROKEN" "PODCAST AUDIO SIZE MISMATCH"
    "PODCAST META UNREADABLE" "PODCAST FEED EMPTY"
    "PODCAST SOURCE MISMATCH (job report)"
)
CARRY_FORWARD=0
if [ "$STATE" = ok ] && [ "$DAILY_DUE" = 0 ] && [ "$PREV_STATE" != ok ] && [ -n "$PREV_SIG" ]; then
    all_daily=1
    while IFS=, read -ra _heads; do
        for h in "${_heads[@]}"; do
            found=0
            for d in "${DAILY_HEADLINES[@]}"; do [ "$h" = "$d" ] && found=1 && break; done
            [ "$found" = 1 ] || all_daily=0
        done
    done <<<"$PREV_SIG"
    if [ "$all_daily" = 1 ]; then
        CARRY_FORWARD=1
        log "not declaring recovery: previous fault [$PREV_SIG] is a daily-output fault and those checks are not due this run"
    fi
fi

SPEAK=0; MSG=""
# What gets WRITTEN to alert-state.json. Normally the run's own verdict; under
# carry-forward it is the previous verdict, held until the checks are due again.
STATE_FOR_ALERT="$STATE"; SIG_FOR_ALERT="$SIGNATURE"
if [ "$CARRY_FORWARD" = 1 ]; then
    # Say nothing, change nothing. The next due run decides.
    SINCE="${PREV_SINCE:-$NOW_UTC}"
    STATE_FOR_ALERT="$PREV_STATE"; SIG_FOR_ALERT="$PREV_SIG"
elif [ "$STATE" = ok ]; then
    if [ "$PREV_STATE" != ok ]; then
        SPEAK=1
        MSG="RECOVERED — Afrodite stack is healthy again

All $OK_COUNT checks pass.
Was degraded since: ${PREV_SINCE:-unknown}
Previous faults: ${PREV_SIG:-unknown}
Checked: $NOW_UTC UTC"
    fi
    SINCE="$NOW_UTC"
else
    if [ "$SIGNATURE" != "$PREV_SIG" ] || [ "$PREV_STATE" = ok ]; then
        SPEAK=1
        SINCE="$NOW_UTC"
    else
        SINCE="${PREV_SINCE:-$NOW_UTC}"
        age_h=$(( ( $(date +%s) - ${PREV_ALERT_EPOCH:-0} ) / 3600 ))
        [ "$age_h" -ge "$REALERT_HOURS" ] && SPEAK=1
    fi
    # The headline of the FIRST failure becomes the subject line, so the message
    # opens with the failure class rather than with a generic banner.
    first_head="$(printf '%s\n' "${FINDINGS[@]}" | awk -F'\t' '$1=="FAIL"{print $2; exit}')"
    [ -n "$first_head" ] || first_head="$(printf '%s\n' "${FINDINGS[@]}" | cut -f2 | head -n 1)"
    MSG="$first_head — Afrodite

$REPORT

Since: $SINCE
Full log: /opt/stacks/hermes/health/stack-health.log"
fi

NOTIFIED=0
if [ "$SPEAK" = 1 ] && [ "$NOTIFY" = 1 ]; then
    # A FAILED send deliberately does NOT advance last_alert_epoch, so the next
    # run retries. "Telegram was unreachable" must not be able to swallow the
    # one message that says the box is broken.
    notify "$MSG" && NOTIFIED=1
elif [ "$SPEAK" = 1 ]; then
    # --no-notify counts as a delivered message for state purposes. It is a dry
    # run of the DECISION (speak / stay silent / re-alert), not of the transport,
    # and if it left last_alert_epoch at 0 the cooldown could never engage — so
    # the one thing the flag exists to test would be untestable.
    log "would notify (--no-notify): ${STATE} [${SIGNATURE:-recovered}]"
    NOTIFIED=1
fi

python3 - "$ALERT_STATE" "$STATE_FOR_ALERT" "$SIG_FOR_ALERT" "$SINCE" "$NOTIFIED" "$PREV_ALERT_EPOCH" <<'PY' || true
import json, sys, time
path, state, sig, since, notified, prev_epoch = sys.argv[1:7]
json.dump({
    "state": state,
    "signature": sig,
    "since": since,
    "last_alert_epoch": int(time.time()) if notified == "1" else int(prev_epoch or 0),
    "last_check_epoch": int(time.time()),
}, open(path, "w"), indent=2)
PY

exit "$RC"
