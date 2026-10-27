"""The NBA's official injury report, merged with ESPN's injury list.

The league publishes the report as a PDF every 15 minutes on game days, with
five statuses (Out, Doubtful, Questionable, Probable, Available). ESPN only
says Out or Day-To-Day. Each status becomes a chance of playing, which the
availability feature and the player projections use. When the report is
missing or does not parse, ESPN's list is used alone.
"""

from __future__ import annotations

import io
import logging
import re
import unicodedata
from datetime import datetime
from typing import Any

import pandas as pd
import requests

log = logging.getLogger(__name__)

INDEX = "https://official.nba.com/nba-injury-report-{a}-{b}-season/"
_PDF = re.compile(
    r"https://ak-static\.cms\.nba\.com/referee/injury/"
    r"Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{2})_?(\d{2})?(AM|PM)\.pdf"
)
STATUSES = ("Out", "Doubtful", "Questionable", "Probable", "Available")
P_PLAY = {
    "out": 0.0,
    "doubtful": 0.25,
    "questionable": 0.5,
    "day-to-day": 0.6,  # ESPN's only non-Out status
    "probable": 0.9,
    "available": 1.0,
}
_ROW = re.compile(
    r"([^\s,][^,]*?,\s+[^\s].*?)\s{2,}(" + "|".join(STATUSES) + r")(?:\s{2,}|\s*$)"
)
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# official.nba.com refuses non-browser user agents.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}


def normalize_name(name: str) -> str:
    """'Butler III, Jimmy' and 'Jimmy Butler III' both become 'jimmy butler'."""
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first} {last}"
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = re.sub(r"[^a-z ]", " ", name.lower()).split()
    return " ".join(t for t in tokens if t not in _SUFFIXES)


def latest_report_url(season: int, now: pd.Timestamp) -> str | None:
    """The newest report published at or before ``now`` (Eastern time)."""
    url = INDEX.format(a=season - 1, b=str(season)[-2:])
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    now_et = now.tz_convert("America/New_York").tz_localize(None)
    best: tuple[datetime, str] | None = None
    for m in _PDF.finditer(resp.text):
        day, hour, minute, half = m.groups()
        stamp = datetime.strptime(
            f"{day} {hour}:{minute or '00'} {half}", "%Y-%m-%d %I:%M %p"
        )
        if stamp <= now_et and (best is None or stamp > best[0]):
            best = (stamp, m.group(0))
    return best[1] if best else None


def parse_report(pdf: bytes, team_names: list[str]) -> pd.DataFrame:
    """Rows of (team, player, status) from one report PDF."""
    from pypdf import PdfReader

    names = sorted(team_names, key=len, reverse=True)
    rows: list[dict[str, Any]] = []
    team = None
    for page in PdfReader(io.BytesIO(pdf)).pages:
        lines = page.extract_text(extraction_mode="layout").splitlines()
        # Layout extraction lists a page's rows bottom-up.
        for line in reversed(lines):
            flat = " ".join(line.split())
            for candidate in names:
                if candidate in flat:
                    team = candidate
                    break
            match = _ROW.search(line)
            if team is None or match is None:
                continue
            player = " ".join(match.group(1).split()[-4:])  # drop leading columns
            if "," not in player:
                continue
            player = player[player.rfind(" ", 0, player.index(",")) + 1 :]
            rows.append({"team": team, "player": player, "status": match.group(2)})
    return pd.DataFrame(rows, columns=["team", "player", "status"])


def official_injuries(
    season: int,
    now: pd.Timestamp,
    teams: dict[str, dict[str, Any]],
    rosters: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None]:
    """The latest official report, matched to ESPN athlete ids."""
    url = latest_report_url(season, now)
    if url is None:
        return pd.DataFrame(), None
    resp = requests.get(url, headers=_HEADERS, timeout=30)
    resp.raise_for_status()
    by_name = {t["name"]: tid for tid, t in teams.items()}
    report = parse_report(resp.content, list(by_name))
    report["team_id"] = report["team"].map(by_name)
    report["key"] = report["player"].map(normalize_name)
    known = rosters.assign(key=rosters["name"].map(normalize_name))
    merged = report.merge(
        known[["team_id", "key", "athlete_id", "name"]],
        on=["team_id", "key"],
        how="left",
    )
    matched = merged.dropna(subset=["athlete_id"])
    log.info(
        "official injury report %s: %d rows, %d matched to rosters",
        url.rsplit("/", 1)[-1],
        len(report),
        len(matched),
    )
    return matched[["team_id", "athlete_id", "name", "status"]], url


def combine(espn: pd.DataFrame, official: pd.DataFrame) -> pd.DataFrame:
    """One status per player, the official report winning; adds ``p_play``."""
    frames = [f for f in (official, espn) if not f.empty]
    if not frames:
        return pd.DataFrame(
            columns=["team_id", "athlete_id", "name", "status", "p_play"]
        )
    both = pd.concat(frames, ignore_index=True).drop_duplicates("athlete_id")
    both["p_play"] = both["status"].str.lower().map(P_PLAY).fillna(1.0)
    return both.reset_index(drop=True)
