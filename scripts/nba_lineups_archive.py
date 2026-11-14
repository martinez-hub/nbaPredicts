"""Archive the NBA's daily lineups files, kept local.

stats.nba.com/js/data/leaders/00_daily_lineups_YYYYMMDD.json lists, for each
game that day, each player's lineup status (Confirmed / Expected starter),
roster status (Active / Inactive) and the time that status was set. The
server keeps only each day's final state, and nothing says how long old days
stay up (files exist from 2023-24; earlier dates redirect), so every game
day is copied once. NBA data: it stays in data_local/.

    uv run python scripts/nba_lineups_archive.py [--since 2023-10-01]

Days already on disk are skipped; today's (still changing) file is not
stored here: the live logger snapshots it through the day.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from nbapredicts.collect import Store

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "nba_lineups"
URL = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{day}.json"
USER_AGENT = "research-client/1.0"
PAUSE = 1.0

log = logging.getLogger("nba_lineups_archive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--since", default="2023-10-01")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    games = Store(ROOT / "data").load("games")
    today = pd.Timestamp.now(tz="America/New_York").date()
    days = sorted(
        d for d in set(games["date"]) if date.fromisoformat(args.since) <= d < today
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    OUT.mkdir(parents=True, exist_ok=True)
    missing_path = OUT / "missing_days.json"
    missing = (
        set(json.loads(missing_path.read_text())) if missing_path.exists() else set()
    )
    saved = 0
    for day in days:
        stamp = day.strftime("%Y%m%d")
        path = OUT / f"{day}.json.gz"
        if path.exists() or str(day) in missing:
            continue
        time.sleep(PAUSE)
        resp = session.get(URL.format(day=stamp), timeout=60, allow_redirects=False)
        if resp.status_code in (301, 302, 404):
            missing.add(str(day))
            continue
        if resp.status_code in (401, 403, 429):
            log.error("stopped: %d on %s", resp.status_code, day)
            sys.exit(2)
        resp.raise_for_status()
        with gzip.open(path, "wt") as f:
            f.write(resp.text)
        saved += 1
        if saved % 50 == 0:
            log.info("%d days saved (at %s)", saved, day)
    missing_path.write_text(json.dumps(sorted(missing)))
    have = sorted(p.name[:10] for p in OUT.glob("*.json.gz"))
    log.info(
        "saved %d; %d days on disk (%s to %s); %d game days without a file",
        saved,
        len(have),
        have[0] if have else "-",
        have[-1] if have else "-",
        len(missing),
    )


if __name__ == "__main__":
    main()
