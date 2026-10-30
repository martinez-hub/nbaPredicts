from __future__ import annotations

import numpy as np
import pandas as pd

from nbapredicts.impact import TEAM_MINUTES, expected_strength, monthly_ratings


def test_expected_strength_drops_without_a_good_player():
    rating = np.array([6.0, 1.0, 0.0, 0.0, -1.0, -2.0, -2.0])
    minutes = np.array([36.0, 34, 32, 30, 28, 20, 10])
    full = expected_strength(rating, minutes, np.ones(7))
    out = expected_strength(rating, minutes, np.array([0.0, 1, 1, 1, 1, 1, 1]))
    half = expected_strength(rating, minutes, np.array([0.5, 1, 1, 1, 1, 1, 1]))
    assert out < half < full


def test_minutes_fill_the_game():
    # One rating of 1 for everyone: strength = total minutes / 48 = 5.
    rating = np.ones(9)
    minutes = np.linspace(10, 30, 9)
    assert np.isclose(expected_strength(rating, minutes, np.ones(9)), TEAM_MINUTES / 48)


def test_ratings_never_see_their_own_month(league):
    games, _, player_box = league
    games = games.copy()
    start = pd.to_datetime(games["start"], utc=True)
    month = sorted(start.dt.tz_localize(None).dt.to_period("M").unique())[-1]
    cut = month.to_timestamp().tz_localize("UTC")
    base = monthly_ratings(player_box, games, ridge=5.0, min_games=50)
    later = start >= cut
    games.loc[later, ["home_pts", "away_pts"]] = [60.0, 150.0]
    changed = monthly_ratings(player_box, games, ridge=5.0, min_games=50)
    a = base[base["month"] == cut].set_index("athlete_id")["rating"]
    b = changed[changed["month"] == cut].set_index("athlete_id")["rating"]
    assert len(a) and np.allclose(a, b.reindex(a.index))


def _stints(games, player_box, bench_share: float) -> pd.DataFrame:
    """Two stints a game: the starters for the first half, the bench for
    the second, splitting the game's margin ``1 - bench_share`` to
    ``bench_share``."""
    from nbapredicts.collect import LINEUP_COLS

    on = player_box.assign(j=player_box["athlete_id"].str.split("-").str[1].astype(int))
    rows = []
    for g in games.itertuples(index=False):
        margin = g.home_pts - g.away_pts
        for t0, picks, share in (
            (0, range(5), 1 - bench_share),
            (1440, range(4, 9), bench_share),
        ):
            five = {
                tid: on[
                    (on["game_id"] == g.game_id)
                    & (on["team_id"] == tid)
                    & on["j"].isin(picks)
                ]["athlete_id"].tolist()
                for tid in (g.home_id, g.away_id)
            }
            rows.append(
                {
                    "game_id": g.game_id,
                    "season": g.season,
                    "t0": t0,
                    "t1": t0 + 1440,
                    "margin": margin * share,
                    **dict(
                        zip(LINEUP_COLS, five[g.home_id] + five[g.away_id], strict=True)
                    ),
                }
            )
    return pd.DataFrame(rows)


def test_stint_ratings_credit_the_players_on_the_floor(league):
    games, _, player_box = league
    # The best team at home: its stints carry winning margins.
    margin = games["home_pts"] - games["away_pts"]
    team = margin.groupby(games["home_id"]).mean().idxmax()

    def latest(stints=None) -> pd.Series:
        r = monthly_ratings(player_box, games, stints, min_games=50)
        return r[r["month"] == r["month"].max()].set_index("athlete_id")["rating"]

    starters_won = latest(_stints(games, player_box, bench_share=0.0))
    bench_won = latest(_stints(games, player_box, bench_share=1.0))
    # The same games and box scores; only who was on the floor for the
    # margin differs, and the ratings follow it.
    star, bench = f"{team}-0", f"{team}-8"
    assert starters_won[star] - bench_won[star] > 0.1
    assert bench_won[bench] - starters_won[bench] > 0.1
    assert not np.allclose(latest(), starters_won)


def test_stint_ratings_never_see_their_own_month(league):
    games, _, player_box = league
    stints = _stints(games, player_box, bench_share=0.5)
    start = pd.to_datetime(games["start"], utc=True)
    month = sorted(start.dt.tz_localize(None).dt.to_period("M").unique())[-1]
    cut = month.to_timestamp().tz_localize("UTC")
    base = monthly_ratings(player_box, games, stints, min_games=50)
    later = set(games.loc[start >= cut, "game_id"])
    g2 = games.copy()
    g2.loc[g2["game_id"].isin(later), ["home_pts", "away_pts"]] = [60.0, 150.0]
    s2 = stints.copy()
    s2.loc[s2["game_id"].isin(later), "margin"] = -45.0
    changed = monthly_ratings(player_box, g2, s2, min_games=50)
    a = base[base["month"] == cut].set_index("athlete_id")["rating"]
    b = changed[changed["month"] == cut].set_index("athlete_id")["rating"]
    assert len(a) and np.allclose(a, b.reindex(a.index))
    # And the rewrite does reach the month after, so the check has teeth.
    nxt = base["month"].max()
    assert nxt > cut and not np.allclose(
        base[base["month"] == nxt]["rating"], changed[changed["month"] == nxt]["rating"]
    )
