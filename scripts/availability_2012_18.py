"""Pre-game availability for 2012-13..2018-19 (before the official report's
history starts, 2018-12-17), from archived ESPN and CBS injury pages.

Training seasons before the official report count every player as
available, though the results reflect who sat out. The Wayback Machine kept
ESPN's and CBS's injury pages most days (data_local/text_history,
scripts/text_history.py). Each game day takes the latest copy saved before
tip-off and no more than 36 hours old: ESPN's statuses as they are (Out,
Day-To-Day: injuries.P_PLAY has both), CBS's wording translated (Game Time
Decision and Questionable -> Questionable, "Expected to be out until at
least <date>" -> Out until then, ...) for days ESPN lacks.

    uv run python scripts/availability_2012_18.py

Runs the winner backtest twice, with and without these rows, and prints
the Brier scores.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from nbapredicts.backtest import backtest_winner
from nbapredicts.collect import Store
from nbapredicts.features import (
    apply_pregame_availability,
    build_features,
    player_values,
    pregame_availability,
    team_state,
)
from nbapredicts.impact import apply_impact, monthly_ratings, pregame_impact
from nbapredicts.injury_history import COLUMNS, load_history
from nbapredicts.pipeline import MODEL_FEATURES

ROOT = Path(__file__).resolve().parents[1]
TEXT = ROOT / "data_local" / "text_history"
FIRST, LAST = "2012-10-01", "2018-12-16"
MAX_AGE = pd.Timedelta(hours=36)
ALIASES = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "New Jersey Nets": "Brooklyn Nets",
    "Los Angeles Clippers": "LA Clippers",
    "Seattle SuperSonics": "Oklahoma City Thunder",
}


def team_names(teams: dict) -> dict[str, str]:
    """Page team names -> our names: full names, past names, and unique
    city-only names (CBS writes "Atlanta"; "Los Angeles" is ambiguous)."""
    names = {t["name"]: t["name"] for t in teams.values()}
    names.update({k: v for k, v in ALIASES.items() if v in names})
    cities: dict[str, list[str]] = {}
    for n in names.values():
        cities.setdefault(n.rsplit(" ", 1)[0], []).append(n)
    cities.setdefault("Portland", ["Portland Trail Blazers"])
    for city, full in cities.items():
        if len(set(full)) == 1:
            names.setdefault(city, full[0])
    return names


def cbs_status(text: str, game_day: str) -> str | None:
    t = str(text)
    low = t.lower()
    m = re.search(r"out until at least ([A-Z][a-z]{2})\.? (\d{1,2})", t)
    if m:
        year = int(game_day[:4])
        try:
            until = pd.Timestamp(f"{m.group(1)} {m.group(2)} {year}")
        except ValueError:
            return "Out"
        if until < pd.Timestamp(game_day) - pd.Timedelta(days=180):
            until = until.replace(year=year + 1)
        return "Out" if pd.Timestamp(game_day) < until else None
    for key, status in (
        ("game time decision", "Questionable"),
        ("questionable", "Questionable"),
        ("doubtful", "Doubtful"),
        ("likely out", "Doubtful"),
        ("probable", "Probable"),
        ("out", "Out"),
    ):
        if low.startswith(key):
            return status
    return None


def archived_reports(games: pd.DataFrame, teams: dict) -> pd.DataFrame:
    names = team_names(teams)
    g = games[(games["status"] == "final") & (games["start"] >= FIRST)]
    g = g[g["date"].astype(str) <= LAST]
    tips = g.groupby(g["date"].astype(str))["start"].min()  # first tip-off that day
    rows = []
    for source in ("espn_injuries", "cbs_injuries"):
        d = pd.read_csv(TEXT / f"{source}.csv.gz")
        d["snap"] = pd.to_datetime(d["snapshot"]).dt.tz_localize("UTC")
        d = d[(d["snap"] >= pd.Timestamp(FIRST, tz="UTC")) & d["player"].notna()]
        d["team"] = d["team"].map(names)
        d = d[d["team"].notna()]
        snaps = pd.DatetimeIndex(d["snap"].drop_duplicates().sort_values())
        by_snap = dict(tuple(d.groupby("snap")))
        for day, tip in tips.items():
            i = snaps.searchsorted(tip, side="left") - 1
            if i < 0 or tip - snaps[i] > MAX_AGE:
                continue
            snap = snaps[i]
            page = by_snap[snap]
            for _, r in page.iterrows():
                if source == "espn_injuries":
                    status = (
                        str(r["status"]).title().replace("Day-To-Day", "Day-To-Day")
                    )
                    status = {"Suspension": "Out"}.get(status, status)
                else:
                    status = cbs_status(r["text"], day)
                if not status:
                    continue
                rows.append(
                    {
                        "team": r["team"],
                        "player": r["player"],
                        "status": status,
                        "reason": None,
                        "game_date": day,
                        "day": day,
                        "published": snap.tz_convert("America/New_York").strftime(
                            "%Y-%m-%d %H:%M"
                        ),
                        "season": int(day[:4]) + (int(day[5:7]) >= 8),
                        "source": source,
                    }
                )
    rep = pd.DataFrame(rows)
    # ESPN where it has the day, CBS otherwise.
    espn_days = set(rep.loc[rep["source"] == "espn_injuries", "day"])
    rep = rep[(rep["source"] == "espn_injuries") | ~rep["day"].isin(espn_days)]
    rep = rep.drop_duplicates(["day", "team", "player"])
    print(
        f"archived reports: {len(rep)} rows on {rep['day'].nunique()} of "
        f"{len(tips)} game days; {rep.groupby('source')['day'].nunique().to_dict()}"
    )
    return rep[list(COLUMNS)]


def backtest(reports: pd.DataFrame) -> dict:
    store = Store(ROOT / "data")
    games = store.load("games")
    team_box, player_box = store.load("team_box"), store.load("player_box")
    finished = sorted(games.loc[games["status"] == "final", "season"].unique())
    seasons = [int(s) for s in finished[-4:-1]]
    pv = player_values(player_box, games)
    feats = build_features(games, team_state(games, team_box, pv, []), [])
    teams = json.loads((ROOT / "data" / "teams.json").read_text())
    team_ids = {t["name"]: tid for tid, t in teams.items()}
    feats = apply_pregame_availability(
        feats, pregame_availability(pv, games, reports, team_ids)
    )
    ratings = monthly_ratings(player_box, games, store.load("stints"))
    feats = apply_impact(
        feats, pregame_impact(pv, player_box, games, reports, team_ids, ratings)
    )
    return backtest_winner(
        feats, seasons, odds=store.load("odds"), features=MODEL_FEATURES
    )


def main() -> None:
    store = Store(ROOT / "data")
    games = store.load("games")
    teams = json.loads((ROOT / "data" / "teams.json").read_text())
    official = load_history(ROOT, games, teams, pd.Timestamp.now(tz="UTC"))
    archived = archived_reports(games, teams)
    results = {
        "official only": backtest(official),
        "with 2012-18 archive": backtest(
            pd.concat([archived, official], ignore_index=True)
        ),
    }
    print(f"{'model':14s} {'season':>6s} " + " ".join(f"{k:>21s}" for k in results))
    for m in ("equine_market", "equine_gp", "logistic"):
        for i, season in enumerate(results["official only"]["seasons"]):
            vals = [r["seasons"][i][m]["brier"] for r in results.values()]
            print(
                f"{m:14s} {season['season']:6d} " + " ".join(f"{v:21.5f}" for v in vals)
            )


if __name__ == "__main__":
    main()
