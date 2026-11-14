"""The official injury report's history, rebuilt at each run.

Past games' availability comes from each day's last official report
published by 6:30 PM ET (``injuries.REPORT_CUTOFF``). That history used to
be committed to the public repo; it is the league's data, so it is now
rebuilt at each run instead, from sportsdataverse's free parsed copy of
every version of the report (github.com/sportsdataverse/sportsdataverse-
data, release ``nba_stats_injury_report``). The copy runs a day or two
behind, so recent game days it lacks come straight from the league's PDFs.

In a 2023-24..2025-26 backtest the rebuilt history did no worse than the
committed one (it has every version, so each day's pick sits closer to the
cutoff): the features-only model gained 0.0003 Brier pooled and the
published stack was unchanged.

A history that covers too few game days stops the run, rather than letting
it train as if every player had been available.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from .injuries import REPORT_CUTOFF

log = logging.getLogger(__name__)

RELEASE_FILE = (
    "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/"
    "nba_stats_injury_report/injury_report_{season}.parquet"
)
FIRST_SEASON = 2019  # the copy starts 2018-12-17
FIRST_DAY = "2018-12-17"
# Before this day the league named each hourly file for the hour while the
# copy stamps it at half past; from it on, files come every 15 minutes and
# names match the times.
SWITCH = pd.Timestamp("2025-12-21")
COLUMNS = (
    "team",
    "player",
    "status",
    "reason",
    "game_date",
    "day",
    "published",
    "season",
)
MIN_COVERAGE = 0.9  # share of past game days that must have a report
RECENT_DAYS = 4  # game days this recent come from the league's own files

USER_AGENT = "research-client/1.0"


def daily_reports(release: pd.DataFrame) -> pd.DataFrame:
    """Each day's last version published by the cutoff, one row per listing."""
    rel = release[release["report_player_name"].notna()].copy()
    if rel.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    t = pd.to_datetime(rel["report_datetime"])
    published = t.where(t >= SWITCH, t - pd.Timedelta(minutes=30))
    rel["published_ts"] = published
    rel["day"] = published.dt.strftime("%Y-%m-%d")
    minutes = published.dt.hour * 60 + published.dt.minute
    rel = rel[minutes <= REPORT_CUTOFF[0] * 60 + REPORT_CUTOFF[1]]
    last = rel.groupby("day")["published_ts"].transform("max")
    rel = rel[rel["published_ts"] == last]
    return pd.DataFrame(
        {
            "team": rel["team"],
            "player": rel["report_player_name"],
            "status": rel["current_status"],
            "reason": rel["reason"],
            "game_date": pd.to_datetime(rel["game_date"]).dt.strftime("%Y-%m-%d"),
            "day": rel["day"],
            "published": rel["published_ts"].dt.strftime("%Y-%m-%d %H:%M"),
            "season": rel["season"].astype(int),
        }
    ).reset_index(drop=True)


def injury_history(
    games: pd.DataFrame,
    now: pd.Timestamp,
    release: pd.DataFrame,
    fetch_day: Callable[[str], pd.DataFrame],
    first_day: str = FIRST_DAY,
) -> pd.DataFrame:
    """The copy's daily reports, plus recent game days it lacks (fetched with
    ``fetch_day``; today's is the latest report so far, as the run used to
    record it); raises if too few game days have a report."""
    history = daily_reports(release)
    today = now.tz_convert("America/New_York").date()
    game_days = sorted(
        str(d)
        for d in games.loc[games["status"] == "final", "date"].unique()
        if date.fromisoformat(first_day) <= d <= today
    )
    # The copy stops at its last refresh, so for the last few game days the
    # league's own cutoff report replaces it; the copy stands in when the
    # league has nothing for a day.
    fetched = {}
    for day in game_days[-RECENT_DAYS:]:
        rows = fetch_day(day)
        if not rows.empty:
            fetched[day] = rows
    if fetched:
        history = pd.concat(
            [history[~history["day"].isin(fetched)], *fetched.values()],
            ignore_index=True,
        )
        log.info("injury history: %d recent days from the league", len(fetched))
    covered = set(history["day"])
    share = sum(d in covered for d in game_days) / max(len(game_days), 1)
    if share < MIN_COVERAGE:
        raise RuntimeError(
            f"injury history covers {share:.0%} of {len(game_days)} game days "
            f"since {first_day}; refusing to train without it"
        )
    log.info(
        "injury history: %d rows, %d days, %.1f%% of game days",
        len(history),
        len(covered),
        100 * share,
    )
    return history[list(COLUMNS)]


def load_release(
    cache_dir: Path, last_season: int, max_age_hours: float = 6.0
) -> pd.DataFrame:
    """Every season file of the copy, downloaded when older than
    ``max_age_hours`` (past seasons are kept once downloaded)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    frames = []
    for season in range(FIRST_SEASON, last_season + 1):
        path = cache_dir / f"injury_report_{season}.parquet"
        stale = not path.exists() or (
            season >= last_season - 1
            and time.time() - path.stat().st_mtime > max_age_hours * 3600
        )
        if stale:
            resp = session.get(RELEASE_FILE.format(season=season), timeout=120)
            if resp.status_code == 404 and season == last_season:
                continue  # the new season's file is not out yet
            resp.raise_for_status()
            tmp = path.with_suffix(".tmp")
            tmp.write_bytes(resp.content)
            tmp.rename(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def league_fetcher(teams: dict) -> Callable[[str], pd.DataFrame]:
    """A day's cutoff report straight from the league's PDFs."""
    from .backfill import report_by_cutoff
    from .injuries import parse_report, report_time

    names = [t["name"] for t in teams.values()]

    def fetch(day: str) -> pd.DataFrame:
        found = report_by_cutoff(day)
        if found is None:
            log.warning("injury history: no league report for %s", day)
            return pd.DataFrame(columns=list(COLUMNS))
        url, pdf = found
        raw = parse_report(pdf, names)
        published = report_time(url)
        season = int(day[:4]) + (int(day[5:7]) >= 8)
        return raw.reindex(
            columns=["team", "player", "status", "reason", "game_date"]
        ).assign(day=day, published=published.strftime("%Y-%m-%d %H:%M"), season=season)

    return fetch


def load_history(
    root: Path, games: pd.DataFrame, teams: dict, now: pd.Timestamp
) -> pd.DataFrame:
    """What the daily run, the backfill and the backtest use."""
    last_season = int(games["season"].max())
    release = load_release(root / "cache.nosync" / "injury_release", last_season)
    return injury_history(games, now, release, league_fetcher(teams))
