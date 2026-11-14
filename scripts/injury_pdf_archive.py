"""Copy every version of the official injury report PDF, kept local.

The league republished its injury report three times a day, then hourly from
2021-10, then every 15 minutes from 2025-12-21, and the CDN still serves the
old files under their old names. sportsdataverse's parsed copy lists every
version's time (``report_datetime``), so only those names are requested:
``Injury-Report_2024-01-15_05PM.pdf`` before the 15-minute switch (versions
stamped hh:30 are named for the hour) and ``..._05_30PM.pdf`` after it. NBA
data: it stays in data_local/.

    uv run python scripts/injury_pdf_archive.py

The CDN answers a missing file with 403, the same code as a refusal, so the
run stops if its first 25 requests all fail; later misses are recorded.
"""

from __future__ import annotations

import glob
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data_local" / "sportsdataverse" / "nba_stats_injury_report"
OUT = ROOT / "data_local" / "injury_pdfs"
CACHED = ROOT / "cache.nosync" / "injury_reports"  # the archive's daily copies
URL = "https://ak-static.cms.nba.com/referee/injury/{name}"
SWITCH = pd.Timestamp("2025-12-21")  # 15-minute names from here on
USER_AGENT = "research-client/1.0"
PAUSE = 1.0

log = logging.getLogger("injury_pdf_archive")


def file_name(ts: pd.Timestamp) -> str | None:
    if ts >= SWITCH:
        if ts.minute not in (0, 15, 30, 45):
            return None
        return f"Injury-Report_{ts:%Y-%m-%d_%I_%M%p}.pdf"
    if ts.minute != 30:
        return None
    return f"Injury-Report_{ts:%Y-%m-%d_%I%p}.pdf"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    files = sorted(glob.glob(str(RELEASE / "injury_report_*.parquet")))
    stamps = pd.concat(pd.read_parquet(f, columns=["report_datetime"]) for f in files)
    names = sorted(
        {n for n in map(file_name, stamps["report_datetime"].drop_duplicates()) if n}
    )
    OUT.mkdir(parents=True, exist_ok=True)
    missing_path = OUT / "missing.json"
    missing = (
        set(json.loads(missing_path.read_text())) if missing_path.exists() else set()
    )
    todo = [
        n
        for n in names
        if n not in missing and not (OUT / n).exists() and not (CACHED / n).exists()
    ]
    log.info("%d versions listed; %d to fetch", len(names), len(todo))
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    got = 0
    for i, name in enumerate(todo):
        time.sleep(PAUSE)
        resp = session.get(URL.format(name=name), timeout=60)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            (OUT / name).write_bytes(resp.content)
            got += 1
        else:
            missing.add(name)
            if got == 0 and i + 1 >= 25:
                missing_path.write_text(json.dumps(sorted(missing)))
                log.error(
                    "first %d requests all failed (%d): stopping",
                    i + 1,
                    resp.status_code,
                )
                sys.exit(2)
        if (i + 1) % 500 == 0:
            missing_path.write_text(json.dumps(sorted(missing)))
            log.info(
                "%d of %d: %d saved, %d missing", i + 1, len(todo), got, len(missing)
            )
    missing_path.write_text(json.dumps(sorted(missing)))
    log.info("saved %d; %d listed versions missing on the CDN", got, len(missing))


if __name__ == "__main__":
    main()
