from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nbapredicts.collect import PLAYER_STAT_COLS


def make_league(
    n_teams: int = 6,
    n_days: int = 60,
    seasons: tuple[int, ...] = (2024, 2025),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """A synthetic league: games, team box scores and player box scores.

    Teams have a fixed strength, so the home team's chance of winning depends
    on the strength gap, which the winner model should pick up.
    """
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 6, n_teams)
    games, team_rows, player_rows = [], [], []
    gid = 1000
    for season in seasons:
        start = pd.Timestamp(f"{season - 1}-10-20T23:30Z")
        for day in range(n_days):
            order = rng.permutation(n_teams)
            for k in range(0, n_teams - 1, 2):
                h, a = int(order[k]), int(order[k + 1])
                gid += 1
                tip = start + pd.Timedelta(days=day)
                margin = strength[h] - strength[a] + 3 + rng.normal(0, 12)
                hp = float(round(110 + margin / 2))
                ap = float(round(110 - margin / 2))
                games.append(
                    {
                        "game_id": str(gid),
                        "start": tip,
                        "date": tip.tz_convert("America/New_York").date(),
                        "season": season,
                        "season_type": 2,
                        "status": "final",
                        "neutral": False,
                        "home_id": str(h),
                        "away_id": str(a),
                        "home_abbr": f"T{h}",
                        "away_abbr": f"T{a}",
                        "home_pts": hp,
                        "away_pts": ap,
                    }
                )
                for tid, oid, pts, home in ((h, a, hp, True), (a, h, ap, False)):
                    fga = float(rng.integers(80, 95))
                    team_rows.append(
                        {
                            "game_id": str(gid),
                            "team_id": str(tid),
                            "opp_id": str(oid),
                            "is_home": home,
                            "pts": pts,
                            "fgm": float(round(pts / 2.6)),
                            "fga": fga,
                            "fg3m": 12.0,
                            "fg3a": 35.0,
                            "ftm": 15.0,
                            "fta": 20.0,
                            "oreb": 10.0,
                            "dreb": 33.0,
                            "reb": 43.0,
                            "ast": 25.0,
                            "stl": 7.0,
                            "blk": 5.0,
                            "tov": 13.0,
                            "pf": 19.0,
                            "paint_pts": 48.0,
                            "fastbreak_pts": 12.0,
                            "season": season,
                        }
                    )
                    for j in range(9):
                        row = {
                            "game_id": str(gid),
                            "team_id": str(tid),
                            "athlete_id": f"{tid}-{j}",
                            "name": f"P{tid}-{j}",
                            "position": "G",
                            "starter": j < 5,
                            "dnp": False,
                            "season": season,
                        }
                        stats = dict.fromkeys(PLAYER_STAT_COLS, 1.0)
                        stats["min"] = 36.0 - 3 * j
                        stats["pts"] = float(max(0, round(pts / 9 + rng.normal(0, 4))))
                        row.update(stats)
                        player_rows.append(row)
    return pd.DataFrame(games), pd.DataFrame(team_rows), pd.DataFrame(player_rows)


@pytest.fixture
def league() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return make_league()
