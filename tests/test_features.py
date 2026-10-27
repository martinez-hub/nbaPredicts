from __future__ import annotations

import numpy as np
import pandas as pd

from nbapredicts.features import (
    FEATURES,
    build_features,
    current_rosters,
    elo_ratings,
    player_values,
    team_state,
)


def features_for(games, team_box, player_box, upcoming=()):
    pv = player_values(player_box, games)
    state = team_state(games, team_box, pv, list(upcoming))
    return build_features(games, state, list(upcoming)).set_index("game_id")


def test_features_do_not_see_the_game_or_later_games(league):
    games, team_box, player_box = league
    base = features_for(games, team_box, player_box)

    # Rewrite the result and box score of every game from the cut onward.
    cut = games["start"].sort_values().iloc[len(games) // 2]
    later = set(games.loc[games["start"] >= cut, "game_id"])
    g2, t2, p2 = games.copy(), team_box.copy(), player_box.copy()
    g2.loc[g2["game_id"].isin(later), ["home_pts", "away_pts"]] = [80.0, 140.0]
    t2.loc[t2["game_id"].isin(later), ["pts", "fga", "tov"]] = [99.0, 120.0, 30.0]
    p2.loc[p2["game_id"].isin(later), "pts"] = 50.0
    changed = features_for(g2, t2, p2)

    # Everything up to and including the first rewritten game is unchanged.
    upto = games.loc[games["start"] <= cut, "game_id"]
    pd.testing.assert_frame_equal(
        base.loc[upto, FEATURES], changed.loc[upto, FEATURES], check_exact=False
    )
    # And the rewrite does reach later games, so the check above has teeth.
    after = games.loc[games["start"] > cut + pd.Timedelta(days=3), "game_id"]
    assert not np.allclose(
        base.loc[after, "net_rtg_l_diff"], changed.loc[after, "net_rtg_l_diff"]
    )


def test_upcoming_game_features_match_a_final_game_built_the_same_way(league):
    games, team_box, player_box = league
    # Pretend the last game has not been played yet.
    last = games.sort_values("start").iloc[-1]
    g2 = games.copy()
    g2.loc[g2["game_id"] == last["game_id"], ["status", "home_pts", "away_pts"]] = [
        "scheduled",
        np.nan,
        np.nan,
    ]
    t2 = team_box[team_box["game_id"] != last["game_id"]]
    p2 = player_box[player_box["game_id"] != last["game_id"]]
    as_final = features_for(games, team_box, player_box).loc[last["game_id"]]
    as_upcoming = features_for(g2, t2, p2, upcoming=[last["game_id"]]).loc[
        last["game_id"]
    ]
    # Availability differs by construction (box score vs. roster); the rest is
    # pre-game state and must be identical.
    same = [f for f in FEATURES if not f.startswith("avail")]
    np.testing.assert_allclose(
        as_final[same].astype(float), as_upcoming[same].astype(float), rtol=1e-9
    )
    assert np.isnan(as_upcoming["home_win"])


def test_elo_is_zero_sum_and_read_before_the_game(league):
    games, _, _ = league
    elo, final = elo_ratings(games)
    first = elo.iloc[0]
    assert first["home_elo"] == first["away_elo"] == 1505.0
    # Season carryover regresses toward the mean, so the total drifts only by
    # that pull; within one season ratings are zero-sum.
    one_season = games[games["season"] == games["season"].min()]
    _, ratings = elo_ratings(one_season)
    assert abs(sum(ratings.values()) - 1505.0 * len(ratings)) < 1e-6
    assert len(final) == games["home_id"].nunique()


def test_season_record_resets_and_counts_only_earlier_games(league):
    games, team_box, player_box = league
    pv = player_values(player_box, games)
    state = team_state(games, team_box, pv, [])
    openers = state.groupby(["team_id", "season"]).head(1)
    assert (openers["season_games"] == 0).all()
    assert (openers["season_win"] == 0.5).all()
    second = state.groupby(["team_id", "season"]).nth(1)
    assert (second["season_games"] == 1).all()


def test_rosters_follow_each_team_not_the_calendar(league):
    games, _, player_box = league
    pv = player_values(player_box, games)
    rosters = current_rosters(pv)
    assert rosters.groupby("team_id")["athlete_id"].nunique().min() == 9
