"""Daily refresh of the local data archives.

launchd runs this every morning through the "nbaPredicts Refresh" app
(~/Applications/nbaPredictsRefresh.app: a signed copy of Python under its
own bundle id), the one program allowed to reach the external SSD; other
launchd jobs may not (macOS: "Operation not permitted"). See
scripts/launchd/com.martinez-hub.nbapredicts-refresh.plist.

Each step is independent: a failure is logged and the rest still run.

    live-sweep       finished live-log days, internal disk -> SSD
    sportsdataverse  release files that changed
    kaggle           datasets with a newer version
    polymarket       new games: events, 1-minute prices, trades (all markets)
    espn_winprob     new final games
    nba_lineups      new game days
    injury_pdfs      new report versions

Weekly (when the last success is 6.5 days old or more; a failure retries
the next morning):

    wikidata         players, teams, awards, seasons queried again
    wikipedia        new days of page views; new players' articles
    open_data        Hugging Face datasets that changed; GitHub repos pulled
    crosswalk        the id table rebuilt (new players and games)

It first checks the internal disk (where the live logger writes) and posts
a notification when space is low. It never touches the project's git, runs
one at a time (a lock on the internal disk), skips Polymarket while another
Polymarket archive is running, and exits quietly when the SSD is not
mounted.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Volumes/ExtremeSSD/nbaPredicts")
LOCK = Path.home() / ".local/share/nbapredicts/refresh.lock"
STAMPS = Path.home() / ".local/share/nbapredicts/refresh_weekly.json"
HOURS = 3600
WEEKLY_AFTER = 6.5 * 24 * HOURS

STEPS = [
    (
        "live-sweep",
        ["-c", "from nbapredicts.cli import main; main(['live-sweep'])"],
        HOURS,
    ),
    ("sportsdataverse", ["scripts/sportsdataverse_pin.py"], 2 * HOURS),
    ("kaggle", ["scripts/kaggle_archive.py"], 2 * HOURS),
    ("polymarket events", ["scripts/polymarket_archive.py", "events"], HOURS),
    ("polymarket prices", ["scripts/polymarket_archive.py", "prices"], 4 * HOURS),
    (
        "polymarket trades",
        ["scripts/polymarket_archive.py", "trades", "--kinds", "all"],
        6 * HOURS,
    ),
    ("espn_winprob", ["scripts/espn_winprob_archive.py", "--first", "2026"], HOURS),
    ("nba_lineups", ["scripts/nba_lineups_archive.py"], HOURS),
    ("injury_pdfs", ["scripts/injury_pdf_archive.py"], 2 * HOURS),
]

WEEKLY = [
    ("wikidata", ["scripts/history_archive.py", "wikidata", "--refresh"], HOURS),
    ("wikipedia", ["scripts/wikipedia_archive.py"], 2 * HOURS),
    ("open_data", ["scripts/open_data_archive.py"], 2 * HOURS),
    ("crosswalk", ["scripts/build_crosswalk.py"], HOURS),
]


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def polymarket_busy() -> bool:
    mine = str(os.getpid())
    out = subprocess.run(
        ["pgrep", "-f", "polymarket_archive.py"], capture_output=True, text=True
    ).stdout.split()
    return any(pid != mine for pid in out)


def check_disk() -> None:
    """Log the internal disk's free space; notify when the logger is close
    to stopping (or has stopped)."""
    free = shutil.disk_usage(Path.home()).free
    log(f"internal disk: {free / 1024**3:.1f} GB free")
    try:
        import pandas as pd

        from nbapredicts.live import disk_alert, notify

        alert = disk_alert(free, {}, pd.Timestamp.now(tz="UTC"))
        if alert:
            notify(alert)
    except Exception as err:  # the alert must not stop the refresh
        log(f"disk alert failed: {err}")


def run_step(name: str, args: list[str], timeout: int) -> bool:
    start = time.time()
    try:
        done = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired as err:
        # What the step printed before it was stopped (bytes, even in text mode).
        out = err.stderr or err.stdout or b""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        log(f"{name}: stopped after {timeout}s {out.strip().splitlines()[-1:]}")
        return False
    tail = (done.stderr or done.stdout).strip().splitlines()[-1:]
    status = "ok" if done.returncode == 0 else f"exit {done.returncode}"
    log(f"{name}: {status} in {time.time() - start:.0f}s {tail}")
    return done.returncode == 0


def main() -> None:
    check_disk()
    if not os.path.ismount("/Volumes/ExtremeSSD"):
        log("SSD not mounted; nothing to do")
        return
    try:
        os.listdir(ROOT)
    except PermissionError:
        log("no permission for the SSD: allow nbaPredicts Refresh in System Settings")
        sys.exit(1)
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log("another refresh is running; exiting")
            return
        for name, args, timeout in STEPS:
            if name.startswith("polymarket") and polymarket_busy():
                log(f"{name}: skipped (another Polymarket archive is running)")
                continue
            run_step(name, args, timeout)
        stamps = json.loads(STAMPS.read_text()) if STAMPS.exists() else {}
        for name, args, timeout in WEEKLY:
            if time.time() - stamps.get(name, 0) < WEEKLY_AFTER:
                continue
            if run_step(name, args, timeout):
                stamps[name] = time.time()
                STAMPS.write_text(json.dumps(stamps, indent=1))
    log("refresh done")


if __name__ == "__main__":
    main()
