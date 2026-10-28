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
    # 1.4% of matched Doubtful players played (2018-19..2026-27 reports
    # joined to box scores); Questionable 51%, Probable 82%.
    "doubtful": 0.03,
    "questionable": 0.5,
    "day-to-day": 0.6,  # ESPN's only non-Out status
    "probable": 0.9,
    "available": 1.0,
}
# "Last, First" or, often in the PDFs, "Nance Jr.,Larry" with no space.
_ROW = re.compile(
    r"([^\s,][^,]*?,\s*[^\s].*?)\s{2,}(" + "|".join(STATUSES) + r")(?:\s{2,}|\s*$)"
)
_GAME_DATE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}
# official.nba.com refuses non-browser user agents.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    " (KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}


def normalize_name(name: str) -> str:
    """'Butler III, Jimmy' and 'Jimmy Butler III' both become 'jimmy butler'."""
    if not isinstance(name, str):
        return ""
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


def _team_keys(team_names: list[str]) -> dict[str, str]:
    """Strings that identify a team in the report, mapped to its full name.

    The full name, the name without its last word ("Portland Trail") and the
    nickname ("Blazers"), so a name the PDF wraps across two lines is still
    recognized from either half. Strings two teams share are dropped.
    """
    seen: dict[str, set[str]] = {}
    for name in team_names:
        words = name.split()
        for key in {name, " ".join(words[:-1]), words[-1]}:
            if key:
                seen.setdefault(key, set()).add(name)
    # Keys under four letters ("LA") occur inside other words.
    return {k: next(iter(v)) for k, v in seen.items() if len(v) == 1 and len(k) >= 4}


def _squash(text: str) -> str:
    return "".join(text.split()).lower()


def _last_team(head: str, keys: dict[str, str]) -> str | None:
    """The team whose key ends last in ``head``; the longer key wins a tie
    ("hornets" over "nets")."""
    best: tuple[int, int, str] | None = None
    for key, team in keys.items():
        at = head.rfind(key)
        if at >= 0:
            cand = (at + len(key), len(key), team)
            if best is None or cand > best:
                best = cand
    return best[2] if best else None


def parse_report(pdf: bytes, team_names: list[str]) -> pd.DataFrame:
    """Rows of (team, player, status) from one report PDF."""
    from pypdf import PdfReader

    # Compared with spaces removed: the PDF prints "Portland  TrailBlazers".
    keys = {_squash(k): v for k, v in _team_keys(team_names).items()}
    rows: list[dict[str, Any]] = []
    team = game_date = None
    for page in PdfReader(io.BytesIO(pdf)).pages:
        lines = page.extract_text(extraction_mode="layout").splitlines()
        # Layout extraction lists a page's rows bottom-up.
        for line in reversed(lines):
            match = _ROW.search(line)
            # The team column ends at the last run of 2+ spaces before the
            # player's "Last, First" comma; looking only there keeps a
            # reason's text from switching the team.
            if match:
                comma = line.index(",", match.start(1))
                boundary = line.rfind("  ", 0, comma)
                head = line[: max(boundary, 0)]
            else:
                head = line
            # A report covers today's games and often tomorrow's; the Game
            # Date column marks the first row of each date.
            dated = _GAME_DATE.search(head)
            if dated:
                game_date = datetime.strptime(dated.group(0), "%m/%d/%Y").date()
            found = _last_team(_squash(head), keys)
            if found:
                team = found
            if team is None or match is None:
                continue
            # The whole player column, so "Porter Jr., Michael" and "Jones
            # Garcia, David" keep every word of the surname.
            start = boundary + 2 if boundary >= 0 else match.start(1)
            # The name column ends at the first 2+ space gap after the comma
            # (older reports put the reason column before the status).
            gap = re.search(r"\S\s{2,}", line[comma + 1 :])
            end = comma + 1 + gap.start() + 1 if gap else match.start(2)
            player = " ".join(line[start : min(end, match.start(2))].split())
            if "," not in player:
                continue
            reason = " ".join(line[match.end(2) :].split())  # first line of it
            rows.append(
                {
                    "team": team,
                    "player": player,
                    "status": match.group(2),
                    "reason": reason,
                    "game_date": game_date.isoformat() if game_date else None,
                }
            )
    return pd.DataFrame(
        rows, columns=["team", "player", "status", "reason", "game_date"]
    )


def official_injuries(
    season: int,
    now: pd.Timestamp,
    teams: dict[str, dict[str, Any]],
    rosters: pd.DataFrame,
) -> tuple[pd.DataFrame, str | None, pd.DataFrame]:
    """The latest official report, matched to ESPN athlete ids, and its raw rows."""
    url = latest_report_url(season, now)
    if url is None:
        return pd.DataFrame(), None, pd.DataFrame()
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
    # Fall back to a name that is unique league-wide (a late trade, or a team
    # column the PDF mangled).
    unique = known.drop_duplicates("key", keep=False).set_index("key")
    gap = merged["athlete_id"].isna() & merged["key"].isin(unique.index)
    for col in ("athlete_id", "name", "team_id"):
        merged.loc[gap, col] = merged.loc[gap, "key"].map(unique[col])
    matched = merged.dropna(subset=["athlete_id"])
    log.info(
        "official injury report %s: %d rows, %d matched to rosters",
        url.rsplit("/", 1)[-1],
        len(report),
        len(matched),
    )
    return matched[["team_id", "athlete_id", "name", "status"]], url, report


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


REPORT_CUTOFF = (18, 30)  # the history keeps each day's last report by 6:30 PM ET


def report_time(url: str) -> datetime | None:
    m = _PDF.search(url)
    if not m:
        return None
    day, hour, minute, half = m.groups()
    return datetime.strptime(
        f"{day} {hour}:{minute or '00'} {half}", "%Y-%m-%d %I:%M %p"
    )


def record_report(store: Any, raw: pd.DataFrame, url: str | None, season: int) -> bool:
    """Add today's report to ``data/injury_reports`` (same rule as the archive).

    A day keeps its latest report published by ``REPORT_CUTOFF``; later
    reports are not recorded, so the history matches what the evening run
    (and the backfilled seasons) saw.
    """
    published = report_time(url) if url else None
    if raw.empty or published is None:
        return False
    if (published.hour, published.minute) > REPORT_CUTOFF:
        return False
    day, stamp = published.strftime("%Y-%m-%d"), published.strftime("%Y-%m-%d %H:%M")
    cols = ["team", "player", "status", "reason", "game_date"]
    rows = raw.reindex(columns=cols).assign(day=day, published=stamp, season=season)
    old = store.load("injury_reports")
    if not old.empty:
        same_day = old["day"] == day
        if same_day.any() and old.loc[same_day, "published"].max() >= stamp:
            return False
        rows = pd.concat([old[~same_day], rows], ignore_index=True)
    store.save("injury_reports", rows)
    return True
