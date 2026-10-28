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
