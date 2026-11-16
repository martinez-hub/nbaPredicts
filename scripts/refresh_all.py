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

It never touches git, runs one at a time (a lock on the internal disk),
skips Polymarket while another Polymarket archive is running, and exits
quietly when the SSD is not mounted.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path("/Volumes/ExtremeSSD/nbaPredicts")
LOCK = Path.home() / ".local/share/nbapredicts/refresh.lock"
HOURS = 3600

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


def log(msg: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True)


def polymarket_busy() -> bool:
    mine = str(os.getpid())
    out = subprocess.run(
        ["pgrep", "-f", "polymarket_archive.py"], capture_output=True, text=True
    ).stdout.split()
    return any(pid != mine for pid in out)


def main() -> None:
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
            start = time.time()
            try:
                done = subprocess.run(
                    [sys.executable, *args],
                    cwd=ROOT,
                    timeout=timeout,
                    capture_output=True,
                    text=True,
                )
                tail = (done.stderr or done.stdout).strip().splitlines()[-1:]
                status = "ok" if done.returncode == 0 else f"exit {done.returncode}"
                log(f"{name}: {status} in {time.time() - start:.0f}s {tail}")
            except subprocess.TimeoutExpired:
                log(f"{name}: stopped after {timeout}s")
    log("refresh done")


if __name__ == "__main__":
    main()
