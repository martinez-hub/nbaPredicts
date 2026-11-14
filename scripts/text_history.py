"""NBA injury notes and player news since the mid-2000s, from the Internet
Archive's copies of the pages that published them (experiment data).

The daily job logs ESPN's injury notes and news from 2026-10-28 on
(``text_log.py``). For earlier seasons the only free record is what the
Wayback Machine saved:

- ``espn_injuries``: ESPN's injury page, every team's injured players with a
  dated note (espn.go.com from 2010, espn.com from 2016; one or two copies a
  year before 2010).
- ``cbs_injuries``: CBS Sports' injury page: updated date, injury, expected
  return (from 2012).
- ``rotoworld_news``: Rotoworld's NBA player news, a dated blurb and analysis
  per item (2012-2018, after which the page moved).

One copy a day at most. The text belongs to its publishers, so the corpus
stays on this disk (``data_local/``, git-ignored) and out of the public repo.

    uv run python scripts/text_history.py list
    uv run python scripts/text_history.py fetch [--workers 2]
    uv run python scripts/text_history.py parse
"""

from __future__ import annotations

import argparse
import gzip
import html
import json
import logging
import re
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "cache.nosync/text_history"
OUT = ROOT / "data_local/text_history"
SOURCES = {
    "espn_injuries": ["espn.go.com/nba/injuries", "www.espn.com/nba/injuries"],
    "cbs_injuries": ["www.cbssports.com/nba/injuries"],
    "rotoworld_news": ["www.rotoworld.com/playernews/nba/basketball-player-news"],
}
CDX = "https://web.archive.org/cdx/search/cdx"
# The Wayback Machine answers python-requests' default User-Agent with 429.
USER_AGENT = (
    "nbaPredicts research crawler (+https://github.com/martinez-hub/nbaPredicts)"
)
COLS = [
    "source", "snapshot", "team", "player", "position", "status", "note_date",
    "injury", "text", "impact", "expected_return",
]  # fmt: skip

log = logging.getLogger("text_history")


# --------------------------------------------------------------------------
# list and fetch
# --------------------------------------------------------------------------


def new_session() -> requests.Session:
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    return session


def get(session: requests.Session, endpoint: str, **params) -> requests.Response:
    for attempt in range(6):
        try:
            r = session.get(endpoint, params=params, timeout=60)
            if r.status_code == 200:
                return r
            if r.status_code not in (429, 500, 502, 503, 504):
                r.raise_for_status()
            log.warning("HTTP %d, retrying %s", r.status_code, endpoint)
        except (requests.ConnectionError, requests.Timeout) as err:
            log.warning("%s, retrying %s", type(err).__name__, endpoint)
        time.sleep(min(120, 5 * 2**attempt))
    raise RuntimeError(f"giving up on {endpoint} {params}")


def list_snapshots() -> None:
    """One saved copy a day per source, oldest first."""
    session = new_session()
    out = {}
    for source, urls in SOURCES.items():
        seen: dict[str, tuple[str, str]] = {}
        for url in urls:
            rows = get(
                session, CDX, url=url, output="json", fl="timestamp,original",
                filter="statuscode:200", collapse="timestamp:8",
            ).json()[1:]  # fmt: skip
            for ts, original in rows:
                seen.setdefault(ts[:8], (ts, original))
            time.sleep(1.0)
        out[source] = sorted(seen.values())
        years: dict[str, int] = {}
        for ts, _ in out[source]:
            years[ts[:4]] = years.get(ts[:4], 0) + 1
        log.info("%s: %d days %s", source, len(out[source]), years)
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "snapshots.json").write_text(json.dumps(out))


def raw_path(source: str, ts: str) -> Path:
    return RAW / "raw" / source / f"{ts}.html.gz"


def fetch(workers: int) -> None:
    """Every listed copy, stored as fetched (decompressed); resumable."""
    snaps = json.loads((RAW / "snapshots.json").read_text())
    todo = [
        (s, ts, url)
        for s, rows in snaps.items()
        for ts, url in rows
        if not raw_path(s, ts).exists()
    ]
    log.info("%d copies to fetch", len(todo))
    local = threading.local()

    def one(source: str, ts: str, url: str) -> None:
        if not hasattr(local, "session"):
            local.session = new_session()
        r = get(local.session, f"https://web.archive.org/web/{ts}id_/{url}")
        body = r.content
        if body[:2] == b"\x1f\x8b":  # stored gzipped by the original server
            body = gzip.decompress(body)
        path = raw_path(source, ts)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(body))
        tmp.replace(path)
        time.sleep(1.5)  # be polite to the Wayback Machine

    failed = 0
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, *t): t for t in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as err:  # one bad copy should not stop the pull
                failed += 1
                log.warning("%s %s failed: %s", *futures[fut][:2], err)
            if i % 100 == 0:
                log.info("%d/%d fetched, %d failed", i, len(todo), failed)
    log.info("done, %d failed", failed)


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------


def page_lines(raw: bytes) -> list[str]:
    text = raw.decode("utf-8", "ignore")
    text = re.sub(r"<(script|style)\b.*?</\1>", " ", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "\n", text)
    return [
        html.unescape(x).replace("\xa0", " ").strip()
        for x in text.split("\n")
        if x.strip()
    ]


MONTHS = {
    m: i
    for i, m in enumerate("Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)
}
MON_DAY = re.compile(r"^(?:[A-Z][a-z]{2}, )?([A-Z][a-z]{2})\.? (\d{1,2})$")


def dated(month: str, day: str, snapshot: datetime) -> str | None:
    """'Dec 31' seen in a copy saved on 2017-01-01 is 2016-12-31: the latest
    such date no later than a day after the copy."""
    if month not in MONTHS:
        return None
    for year in (snapshot.year, snapshot.year - 1):
        try:
            d = date(year, MONTHS[month], int(day))
        except ValueError:
            continue
        if d <= snapshot.date() + timedelta(days=1):
            return d.isoformat()
    return None


def espn(lines: list[str], snap: datetime) -> Iterator[dict]:
    """Old layout: player, status, date, 'Comment:', note. New layout:
    player, position, return date, status, 'Mon D: note'."""
    team = None
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i + 1] == "NAME":
            team = lines[i]
            i += 2
            continue
        if team and i + 4 < len(lines) and lines[i + 3] == "Comment:":
            m = MON_DAY.match(lines[i + 2])
            if m:
                yield {
                    "team": team,
                    "player": lines[i],
                    "status": lines[i + 1],
                    "note_date": dated(*m.groups(), snap),
                    "text": lines[i + 4],
                }
                i += 5
                continue
        if team and i + 4 < len(lines):
            m = re.match(r"^([A-Z][a-z]{2}) (\d{1,2}): (.+)$", lines[i + 4])
            if m and len(lines[i + 1]) <= 3:
                yield {
                    "team": team,
                    "player": lines[i],
                    "position": lines[i + 1],
                    "expected_return": lines[i + 2],
                    "status": lines[i + 3],
                    "note_date": dated(m[1], m[2], snap),
                    "text": m[3],
                }
                i += 5
                continue
        i += 1


def cbs(lines: list[str], snap: datetime) -> Iterator[dict]:
    """Old layout: updated mm/dd/yy, player, position, injury, expected
    return. New layout: short name, player, position, 'Fri, Dec 30',
    injury, status."""
    team = None
    i = 0
    while i < len(lines):
        if i + 1 < len(lines) and lines[i + 1] in ("Updated", "Player"):
            team = lines[i]
            i += 6
            continue
        if team and i + 4 < len(lines) and re.match(r"^\d{2}/\d{2}/\d{2}$", lines[i]):
            d = datetime.strptime(lines[i], "%m/%d/%y").date().isoformat()
            yield {
                "team": team,
                "player": lines[i + 1],
                "position": lines[i + 2],
                "note_date": d,
                "injury": lines[i + 3],
                "text": lines[i + 4],
            }
            i += 5
            continue
        if team and i + 5 < len(lines):
            m = MON_DAY.match(lines[i + 3])
            if m and len(lines[i + 2]) <= 3 and lines[i][1:3] == ". ":
                yield {
                    "team": team,
                    "player": lines[i + 1],
                    "position": lines[i + 2],
                    "note_date": dated(*m.groups(), snap),
                    "injury": lines[i + 4],
                    "text": lines[i + 5],
                }
                i += 6
                continue
        i += 1


ROTO_STAMP = re.compile(r"^([A-Z][a-z]{2}) (\d{1,2}) - \d{1,2}:\d{2} [AP]M$")


def rotoworld(lines: list[str], snap: datetime) -> Iterator[dict]:
    """Player, '- G -', team, news, analysis, 'Jan 1 - 11:19 PM'."""
    for i in range(len(lines) - 5):
        if not re.match(r"^- [A-Z/]+ -$", lines[i + 1]):
            continue
        for j in range(i + 3, min(i + 7, len(lines))):
            m = ROTO_STAMP.match(lines[j])
            if m:
                body = lines[i + 3 : j]
                yield {
                    "team": lines[i + 2],
                    "player": lines[i],
                    "position": lines[i + 1].strip("- "),
                    "note_date": dated(m[1], m[2], snap),
                    "text": body[0],
                    "impact": " ".join(body[1:]) or None,
                }
                break


PARSERS = {"espn_injuries": espn, "cbs_injuries": cbs, "rotoworld_news": rotoworld}


def parse_copy(source: str, path: Path) -> list[dict]:
    snap = datetime.strptime(path.name[:14], "%Y%m%d%H%M%S")
    lines = page_lines(gzip.decompress(path.read_bytes()))
    return [
        {"source": source, "snapshot": snap.isoformat(), **row}
        for row in PARSERS[source](lines, snap)
    ]


def parse() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for source in SOURCES:
        rows = []
        copies = sorted((RAW / "raw" / source).glob("*.html.gz"))
        for path in copies:
            rows += parse_copy(source, path)
        df = pd.DataFrame(rows, columns=COLS)
        # A note stays on the page for days: keep it once, from the first copy.
        df = df.sort_values("snapshot").drop_duplicates(
            ["player", "note_date", "text"], keep="first"
        )
        df.to_csv(OUT / f"{source}.csv.gz", index=False)
        years = df["note_date"].str[:4].value_counts().sort_index().to_dict()
        summary[source] = {"copies": len(copies), "notes": len(df), "by_year": years}
        log.info("%s: %s", source, json.dumps(summary[source]))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["list", "fetch", "parse"])
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    {"list": list_snapshots, "fetch": lambda: fetch(args.workers), "parse": parse}[
        args.step
    ]()


if __name__ == "__main__":
    main()
