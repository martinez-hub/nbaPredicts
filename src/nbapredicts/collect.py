"""Build and update the game, team box score and player box score tables.

Tables live as parquet under ``data/``:

- ``games``: one row per game (final or scheduled), from team schedules.
- ``team_box``: two rows per final game, one per team.
- ``player_box``: one row per player per final game, including DNPs.

The backfill is idempotent: rerunning it only fetches what is missing, and the
summary cache makes a crashed run cheap to resume.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .espn import SEASON_TYPES, ESPNClient, ESPNError

log = logging.getLogger(__name__)

EASTERN = "America/New_York"
FIRST_SEASON = 2013  # 2012-13

GAME_COLS = [
    "game_id",
    "start",
    "date",
    "season",
    "season_type",
    "status",
    "neutral",
    "home_id",
    "away_id",
    "home_abbr",
    "away_abbr",
    "home_pts",
    "away_pts",
]
TEAM_STAT_COLS = [
    "pts",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "tov",
    "pf",
    "paint_pts",
    "fastbreak_pts",
]
PLAYER_STAT_COLS = [
    "min",
    "pts",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "reb",
    "ast",
    "tov",
    "stl",
    "blk",
    "oreb",
    "dreb",
    "pf",
    "plus_minus",
]


def current_season(today: datetime | None = None) -> int:
    """ESPN season label for ``today``: seasons tip off in October."""
    today = today or datetime.now(UTC)
    return today.year + 1 if today.month >= 8 else today.year


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _score(competitor: dict[str, Any]) -> float | None:
    score = competitor.get("score")
    if isinstance(score, dict):
        score = score.get("value")
    try:
        return float(score) if score not in (None, "") else None
    except (TypeError, ValueError):
        return None


def parse_schedule_event(event: dict[str, Any]) -> dict[str, Any] | None:
    comp = event["competitions"][0]
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    if set(sides) != {"home", "away"}:
        return None
    status = comp["status"]["type"]
    if status.get("completed") and status.get("name") == "STATUS_FINAL":
        state = "final"
    elif status.get("state") == "pre":
        state = "scheduled"
    else:
        # Postponed, canceled, suspended or in progress.
        state = status.get("name", "unknown").removeprefix("STATUS_").lower()
    start = pd.Timestamp(event["date"])
    season_type = event.get("seasonType", {}).get("type") or event["season"].get("type")
    return {
        "game_id": str(event["id"]),
        "start": start,
        "date": start.tz_convert(EASTERN).date(),
        "season": int(event["season"]["year"]),
        "season_type": int(season_type),
        "status": state,
        "neutral": bool(comp.get("neutralSite", False)),
        "home_id": str(sides["home"]["team"]["id"]),
        "away_id": str(sides["away"]["team"]["id"]),
        "home_abbr": sides["home"]["team"].get("abbreviation"),
        "away_abbr": sides["away"]["team"].get("abbreviation"),
        "home_pts": _score(sides["home"]) if state == "final" else None,
        "away_pts": _score(sides["away"]) if state == "final" else None,
    }


def _made_attempted(value: str) -> tuple[float, float]:
    try:
        made, attempted = value.split("-")
        return float(made), float(attempted)
    except (AttributeError, ValueError):
        return float("nan"), float("nan")


def _num(value: Any) -> float:
    try:
        return float(str(value).replace("+", ""))
    except (TypeError, ValueError):
        return float("nan")


def parse_summary(
    game_id: str, summary: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (team rows, player rows) for one final game."""
    header = summary["header"]["competitions"][0]
    points = {str(c["team"]["id"]): _score(c) for c in header["competitors"]}
    homeaway = {str(c["team"]["id"]): c["homeAway"] for c in header["competitors"]}
    ids = list(points)

    team_rows = []
    for team in summary["boxscore"].get("teams", []):
        tid = str(team["team"]["id"])
        stats = {s["name"]: s.get("displayValue") for s in team.get("statistics", [])}
        fgm, fga = _made_attempted(stats.get("fieldGoalsMade-fieldGoalsAttempted"))
        fg3m, fg3a = _made_attempted(
            stats.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
        )
        ftm, fta = _made_attempted(stats.get("freeThrowsMade-freeThrowsAttempted"))
        team_rows.append(
            {
                "game_id": game_id,
                "team_id": tid,
                "opp_id": next((i for i in ids if i != tid), None),
                "is_home": homeaway.get(tid) == "home",
                "pts": points.get(tid),
                "fgm": fgm,
                "fga": fga,
                "fg3m": fg3m,
                "fg3a": fg3a,
                "ftm": ftm,
                "fta": fta,
                "oreb": _num(stats.get("offensiveRebounds")),
                "dreb": _num(stats.get("defensiveRebounds")),
                "reb": _num(stats.get("totalRebounds")),
                "ast": _num(stats.get("assists")),
                "stl": _num(stats.get("steals")),
                "blk": _num(stats.get("blocks")),
                "tov": _num(stats.get("totalTurnovers", stats.get("turnovers"))),
                "pf": _num(stats.get("fouls")),
                "paint_pts": _num(stats.get("pointsInPaint")),
                "fastbreak_pts": _num(stats.get("fastBreakPoints")),
            }
        )

    player_rows = []
    for team in summary["boxscore"].get("players", []):
        tid = str(team["team"]["id"])
        for block in team.get("statistics", []):
            keys = block.get("keys") or []
            for ath in block.get("athletes", []):
                info = ath.get("athlete", {})
                row: dict[str, Any] = {
                    "game_id": game_id,
                    "team_id": tid,
                    "athlete_id": str(info.get("id")),
                    "name": info.get("displayName"),
                    "position": (info.get("position") or {}).get("abbreviation"),
                    "starter": bool(ath.get("starter", False)),
                    "dnp": bool(ath.get("didNotPlay", False)) or not ath.get("stats"),
                }
                values = dict(zip(keys, ath.get("stats") or [], strict=False))
                fgm, fga = _made_attempted(
                    values.get("fieldGoalsMade-fieldGoalsAttempted")
                )
                fg3m, fg3a = _made_attempted(
                    values.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted")
                )
                ftm, fta = _made_attempted(
                    values.get("freeThrowsMade-freeThrowsAttempted")
                )
                row.update(
                    {
                        "min": _num(values.get("minutes")),
                        "pts": _num(values.get("points")),
                        "fgm": fgm,
                        "fga": fga,
                        "fg3m": fg3m,
                        "fg3a": fg3a,
                        "ftm": ftm,
                        "fta": fta,
                        "reb": _num(values.get("rebounds")),
                        "ast": _num(values.get("assists")),
                        "tov": _num(values.get("turnovers")),
                        "stl": _num(values.get("steals")),
                        "blk": _num(values.get("blocks")),
                        "oreb": _num(values.get("offensiveRebounds")),
                        "dreb": _num(values.get("defensiveRebounds")),
                        "pf": _num(values.get("fouls")),
                        "plus_minus": _num(values.get("plusMinus")),
                    }
                )
                player_rows.append(row)
    return team_rows, player_rows


def parse_injuries(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for team in summary.get("injuries", []) or []:
        tid = str(team.get("team", {}).get("id"))
        for inj in team.get("injuries", []) or []:
            ath = inj.get("athlete", {})
            rows.append(
                {
                    "team_id": tid,
                    "athlete_id": str(ath.get("id")),
                    "name": ath.get("displayName"),
                    "status": inj.get("status"),
                }
            )
    return rows


# --------------------------------------------------------------------------
# Store
# --------------------------------------------------------------------------


# Stint lineups (stints.py): the home team's five players, then the away's.
LINEUP_COLS = tuple(f"{side}{k}" for side in "ha" for k in range(5))
ID_COLS = (
    "game_id",
    "team_id",
    "opp_id",
    "athlete_id",
    "home_id",
    "away_id",
    *LINEUP_COLS,
)
SORT_KEYS = {
    "games": ["start", "game_id"],
    "team_box": ["game_id", "team_id"],
    "player_box": ["game_id", "team_id", "athlete_id"],
    "odds": ["game_id"],
    "stints": ["game_id", "t0"],
    "stint_checks": ["game_id"],
}


class Store:
    """Tables as one CSV per season: ``data/<table>/<season>.csv``.

    CSV rather than parquet so the daily commit is a small text diff: past
    seasons never change, and the current season's file only gains rows.
    A file is rewritten only when its content changed.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        data_dir.mkdir(parents=True, exist_ok=True)

    def load(self, name: str) -> pd.DataFrame:
        files = sorted((self.data_dir / name).glob("*.csv"))
        if not files:
            return pd.DataFrame()
        df = pd.concat(
            [pd.read_csv(f, dtype={c: str for c in ID_COLS}) for f in files],
            ignore_index=True,
        )
        if "start" in df:
            df["start"] = pd.to_datetime(df["start"], utc=True)
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def save(self, name: str, df: pd.DataFrame) -> None:
        directory = self.data_dir / name
        directory.mkdir(parents=True, exist_ok=True)
        keys = SORT_KEYS.get(name, [])
        for season, part in df.groupby("season"):
            part = part.sort_values(keys) if keys else part
            text = part.to_csv(index=False, float_format="%.10g", lineterminator="\n")
            path = directory / f"{int(season)}.csv"
            if path.exists() and path.read_text() == text:
                continue
            tmp = path.with_suffix(".tmp")
            tmp.write_text(text)
            tmp.replace(path)


def _upsert(old: pd.DataFrame, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if old.empty:
        return new.reset_index(drop=True)
    if new.empty:
        return old
    merged = pd.concat([old, new], ignore_index=True)
    return merged.drop_duplicates(keys, keep="last").reset_index(drop=True)


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------


def fetch_schedules(
    client: ESPNClient, seasons: Iterable[int], workers: int = 6
) -> pd.DataFrame:
    team_ids = [str(t["id"]) for t in client.teams()]
    jobs = [(t, s, st) for s in seasons for st in SEASON_TYPES for t in team_ids]
    rows: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(client.team_schedule, *job): job for job in jobs}
        for fut in as_completed(futures):
            for event in fut.result():
                row = parse_schedule_event(event)
                if row is not None:
                    rows[row["game_id"]] = row
    games = pd.DataFrame(list(rows.values()), columns=GAME_COLS)
    log.info("schedules: %d games across %d requests", len(games), len(jobs))
    return games


def fetch_box_scores(
    client: ESPNClient, game_ids: list[str], workers: int = 6
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    team_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    failed: list[str] = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(client.summary, gid): gid for gid in game_ids}
        for i, fut in enumerate(as_completed(futures), 1):
            gid = futures[fut]
            try:
                teams, players = parse_summary(gid, fut.result())
            except (ESPNError, KeyError, IndexError, TypeError) as err:
                log.warning("box score %s failed: %s", gid, err)
                failed.append(gid)
                continue
            if len(teams) != 2:
                failed.append(gid)
                continue
            team_rows += teams
            player_rows += players
            if i % 500 == 0:
                log.info("box scores: %d/%d", i, len(game_ids))
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows), failed


def update(
    store: Store,
    client: ESPNClient,
    seasons: Iterable[int],
    workers: int = 6,
) -> dict[str, int]:
    """Refresh schedules for ``seasons`` and fetch every missing final box score."""
    seasons = list(seasons)
    games = _upsert(
        store.load("games"), fetch_schedules(client, seasons, workers), ["game_id"]
    )
    games = games.sort_values(["start", "game_id"]).reset_index(drop=True)
    store.save("games", games)

    team_box = store.load("team_box")
    have = set(team_box["game_id"]) if not team_box.empty else set()
    finals = games.loc[games["status"] == "final", "game_id"]
    missing = [g for g in finals if g not in have]
    log.info("fetching %d missing box scores", len(missing))

    new_team, new_player, failed = fetch_box_scores(client, missing, workers)
    if not new_team.empty:
        season_of = games.set_index("game_id")["season"]
        new_team["season"] = new_team["game_id"].map(season_of)
        new_player["season"] = new_player["game_id"].map(season_of)
        store.save("team_box", _upsert(team_box, new_team, ["game_id", "team_id"]))
        store.save(
            "player_box",
            _upsert(store.load("player_box"), new_player, ["game_id", "athlete_id"]),
        )
    return {
        "games": len(games),
        "finals": len(finals),
        "new_box_scores": new_team["game_id"].nunique() if not new_team.empty else 0,
        "failed": len(failed),
    }


def fetch_injuries(client: ESPNClient, game_ids: list[str]) -> pd.DataFrame:
    """Injury reports attached to upcoming games' summaries."""
    rows = []
    for gid in game_ids:
        try:
            rows += parse_injuries(client.summary(gid, use_cache=False))
        except ESPNError as err:
            log.warning("injuries for %s failed: %s", gid, err)
    df = pd.DataFrame(rows, columns=["team_id", "athlete_id", "name", "status"])
    return df.drop_duplicates("athlete_id")
