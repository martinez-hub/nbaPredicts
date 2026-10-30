"""Build the history of official NBA injury reports from the Wayback Machine.

The league's report PDFs (ak-static.cms.nba.com/referee/injury/...) stay on
its CDN for roughly a season; the Wayback Machine has ~14,500 of them from
December 2018 on, with the publication time (Eastern) in the file name. For
each game day this keeps the latest report published by ``CUTOFF`` (before
the first evening tip-offs), downloads it once into ``cache.nosync/``, parses
it, and writes ``data/injury_reports/<season>.csv``: one row per listed
player with team, status, reason and the report's publication time.

    uv run python scripts/injury_archive.py [--limit N] [--seasons 2019 2020 ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from nbapredicts.collect import Store
from nbapredicts.injuries import _HEADERS, parse_report

ROOT = Path(__file__).resolve().parents[1]
CDX = (
    "https://web.archive.org/cdx/search/cdx?url=ak-static.cms.nba.com/referee/injury/"
    "&matchType=prefix&fl=timestamp,original&filter=statuscode:200&collapse=urlkey"
)
NAME = re.compile(r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_?(\d{2})?(AM|PM)\.pdf")
CUTOFF = (18, 30)  # latest report at or before 6:30 PM ET on each game day

log = logging.getLogger("injury_archive")


def report_index() -> pd.DataFrame:
    # The CDX server is often slow or answers 503/504; a saved copy of its
    # listing (cache.nosync/cdx_injury.txt) is used when present.
    cached = ROOT / "cache.nosync" / "cdx_injury.txt"
    text = cached.read_text() if cached.exists() else ""
    for attempt in range(0 if text else 4):
        try:
            resp = requests.get(CDX, timeout=300)
            if resp.status_code == 200 and "Injury-Report_" in resp.text:
                text = resp.text
                break
        except requests.RequestException:
            pass
        time.sleep(30 * (attempt + 1))
    if not text:
        raise RuntimeError("the Wayback CDX index did not answer")
    rows = []
    for line in text.splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2:
            continue
        stamp, url = parts
        m = NAME.search(url)
        if not m:
            continue
        day, hour, minute, half = m.groups()
        published = datetime.strptime(
            f"{day} {hour}:{minute or '00'} {half}", "%Y-%m-%d %I:%M %p"
        )
        rows.append({"wayback": stamp, "url": url, "day": day, "published": published})
    return pd.DataFrame(rows)


def pick_daily(index: pd.DataFrame) -> pd.DataFrame:
    cut = index["published"].dt.hour * 60 + index["published"].dt.minute
    early = index[cut <= CUTOFF[0] * 60 + CUTOFF[1]]
    return early.sort_values("published").groupby("day").tail(1).reset_index(drop=True)


def season_of(day: str) -> int:
    d = datetime.strptime(day, "%Y-%m-%d")
    return d.year + 1 if d.month >= 8 else d.year


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=None,
        help="rebuild only these seasons' files (the current one is also written"
        " by the daily runs)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    teams = json.loads((ROOT / "data" / "teams.json").read_text())
    names = [t["name"] for t in teams.values()]
    cache = ROOT / "cache.nosync" / "injury_reports"
    cache.mkdir(parents=True, exist_ok=True)

    daily = pick_daily(report_index())
    if args.seasons:
        daily = daily[daily["day"].map(season_of).isin(args.seasons)]
    if args.limit:
        daily = daily.tail(args.limit)
    log.info("%d game days with a report by 6:30 PM ET", len(daily))

    session = requests.Session()
    session.headers.update(_HEADERS)
    frames = []
    for i, r in enumerate(daily.itertuples(), 1):
        path = cache / r.url.rsplit("/", 1)[-1]
        if not path.exists():
            # The league's CDN keeps about a season of reports; older ones come
            # from the Wayback Machine.
            wayback = f"https://web.archive.org/web/{r.wayback}id_/{r.url}"
            for attempt, source in enumerate([r.url] + [wayback] * 4):
                try:
                    resp = session.get(source, timeout=60)
                    if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                        path.write_bytes(resp.content)
                        break
                except requests.RequestException:
                    pass
                if attempt:
                    time.sleep(2 * attempt)
            time.sleep(1.0)  # be polite to the Wayback Machine
        if not path.exists():
            log.warning("could not fetch %s", r.url)
            continue
        try:
            rows = parse_report(path.read_bytes(), names)
        except Exception as err:  # a malformed or older-layout PDF
            log.warning("could not parse %s: %s", path.name, err)
            continue
        rows["day"] = r.day
        rows["published"] = r.published.strftime("%Y-%m-%d %H:%M")
        rows["season"] = season_of(r.day)
        frames.append(rows)
        if i % 100 == 0:
            log.info("%d/%d reports", i, len(daily))

    table = pd.concat(frames, ignore_index=True)
    Store(ROOT / "data").save("injury_reports", table)
    log.info(
        "saved %d rows from %d reports; by season: %s",
        len(table),
        table["published"].nunique(),
        table.groupby("season").size().to_dict(),
    )


if __name__ == "__main__":
    main()
