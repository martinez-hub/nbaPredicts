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


def test_upcoming_availability_weights_players_by_chance_of_playing(league):
    from nbapredicts.features import availability, current_rosters

    games, _, player_box = league
    pv = player_values(player_box, games)
    upcoming = games.tail(1)
    team = upcoming["home_id"].iloc[0]
    star = (
        current_rosters(pv).query("team_id == @team").sort_values("post_value").iloc[-1]
    )

    def strength(p_play: float) -> float:
        inj = pd.DataFrame(
            {
                "team_id": [team],
                "athlete_id": [star["athlete_id"]],
                "name": ["x"],
                "status": ["Questionable"],
                "p_play": [p_play],
            }
        )
        a = availability(pv, upcoming, inj)
        return a[
            (a["game_id"] == upcoming["game_id"].iloc[0]) & (a["team_id"] == team)
        ]["avail"].iloc[-1]

    assert strength(0.0) < strength(0.5) < strength(1.0)


def _pregame(games, player_box, reports):
    from nbapredicts.features import pregame_availability

    pv = player_values(player_box, games)
    teams = {f"Team {t}": t for t in games["home_id"].unique()}
    return pregame_availability(pv, games, reports, teams).set_index(
        ["game_id", "team_id"]
    )["avail_pre"]


def test_pregame_availability_uses_the_report_and_not_the_game(league):
    games, _, player_box = league
    # Letter-only names: name matching ignores digits, as real names have none.
    player_box = player_box.assign(
        name=player_box["athlete_id"].map(
            lambda a: (
                "Player " + "".join(chr(97 + int(c)) if c.isdigit() else " " for c in a)
            )
        )
    )
    last = games.sort_values("start").iloc[-1]
    team, day = last["home_id"], str(last["date"])
    empty = pd.DataFrame(columns=["team", "player", "status", "day"])
    base = _pregame(games, player_box, empty).loc[(last["game_id"], team)]

    # The team's best-rated player going into the game.
    pv = player_values(player_box, games)
    before = pv[(pv["team_id"] == team) & (pv["start"] < last["start"])]
    star = before.sort_values("start").groupby("name")["post_value"].last().idxmax()
    first, last_name = star.split(" ", 1)
    out = pd.DataFrame(
        {
            "team": [f"Team {team}"],
            "player": [f"{last_name}, {first}"],
            "status": ["Out"],
            "day": [day],
        }
    )
    with_out = _pregame(games, player_box, out).loc[(last["game_id"], team)]
    assert with_out < base
    # Only that player is affected: a Questionable tag costs less than Out.
    q = out.assign(status="Questionable")
    assert with_out < _pregame(games, player_box, q).loc[(last["game_id"], team)] < base

    # A report published after tip-off (the 6:30 PM report on an afternoon
    # game day) is not known before the game.
    def published(hours: float) -> str:
        t = pd.Timestamp(last["start"]) + pd.Timedelta(hours=hours)
        return t.tz_convert("America/New_York").strftime("%Y-%m-%d %H:%M")

    early = out.assign(published=published(-2))
    late = out.assign(published=published(1))
    assert _pregame(games, player_box, early).loc[(last["game_id"], team)] == with_out
    assert _pregame(games, player_box, late).loc[(last["game_id"], team)] == base

    # Rewriting the game's own box score does not move its pre-game value.
    p2 = player_box.copy()
    p2.loc[p2["game_id"] == last["game_id"], "pts"] = 99.0
    assert _pregame(games, p2, empty).loc[(last["game_id"], team)] == base

    # The report's own columns can come along onto the roster row.
    from nbapredicts.features import pregame_rosters

    teams = {f"Team {t}": t for t in games["home_id"].unique()}
    reason = q.assign(reason="Injury/Illness - Left Ankle; Sprain")
    roster = pregame_rosters(
        pv, games, reason, teams, report_columns=("status", "reason")
    )
    rows = roster[(roster["game_id"] == last["game_id"]) & (roster["team_id"] == team)]
    listed = rows[rows["name"] == star]
    assert listed[["status", "reason", "p_play"]].values.tolist() == [
        ["Questionable", "Injury/Illness - Left Ankle; Sprain", 0.5]
    ]
    assert rows.loc[rows["name"] != star, "status"].isna().all()


def _letters(athlete_id: str) -> str:
    """'3-0' -> 'Player d a': name matching ignores digits, as real names have none."""
    return "Player " + "".join(
        chr(97 + int(c)) if c.isdigit() else " " for c in athlete_id
    )


def _report_name(name: str) -> str:
    first, last = name.split(" ", 1)
    return f"{last}, {first}"


def test_apply_pregame_availability_replaces_the_three_features(league):
    from nbapredicts.features import apply_pregame_availability, pregame_availability

    games, team_box, player_box = league
    player_box = player_box.assign(name=player_box["athlete_id"].map(_letters))
    feats = features_for(games, team_box, player_box).reset_index()
    teams = {f"Team {t}": t for t in games["home_id"].unique()}
    pv = player_values(player_box, games)
    empty = pd.DataFrame(columns=["team", "player", "status", "day"])
    out = apply_pregame_availability(
        feats, pregame_availability(pv, games, empty, teams)
    )
    assert len(out) == len(feats)
    assert (
        out["avail_delta_home"].notna().all() and out["avail_delta_away"].notna().all()
    )
    # Everyone plays every synthetic game, so with no injury report the
    # pre-game strength equals the box-score strength.
    late = out["season_games_min"] > 0
    np.testing.assert_allclose(
        out.loc[late, "avail_diff"], feats.loc[late, "avail_diff"], atol=1e-6
    )

    # A report ruling out the home team's best player moves that game only.
    last = feats.sort_values("start").iloc[-1]
    before = pv[(pv["team_id"] == last["home_id"]) & (pv["start"] < last["start"])]
    star = before.sort_values("start").groupby("name")["post_value"].last().idxmax()
    report = pd.DataFrame(
        {
            "team": [f"Team {last['home_id']}"],
            "player": [_report_name(star)],
            "status": ["Out"],
            "day": [str(last["date"])],
        }
    )
    moved = apply_pregame_availability(
        feats, pregame_availability(pv, games, report, teams)
    ).set_index("game_id")["avail_diff"]
    base = out.set_index("game_id")["avail_diff"]
    assert moved[last["game_id"]] < base[last["game_id"]]
    assert np.allclose(moved.drop(last["game_id"]), base.drop(last["game_id"]))


def test_apply_pregame_availability_keeps_upcoming_games_live(league):
    from nbapredicts.features import apply_pregame_availability, pregame_availability

    games, team_box, player_box = league
    last = games.sort_values("start").iloc[-1]
    g2 = games.copy()
    g2.loc[g2["game_id"] == last["game_id"], ["status", "home_pts", "away_pts"]] = [
        "scheduled",
        np.nan,
        np.nan,
    ]
    t2 = team_box[team_box["game_id"] != last["game_id"]]
    p2 = player_box[player_box["game_id"] != last["game_id"]]
    feats = features_for(g2, t2, p2, upcoming=[last["game_id"]]).reset_index()
    teams = {f"Team {t}": t for t in games["home_id"].unique()}
    empty = pd.DataFrame(columns=["team", "player", "status", "day"])
    pre = pregame_availability(player_values(p2, g2), g2, empty, teams)
    assert last["game_id"] not in set(pre["game_id"])  # computed live, not here
    out = apply_pregame_availability(feats, pre).set_index("game_id")
    before = feats.set_index("game_id")
    assert (
        out.loc[last["game_id"], "avail_diff"]
        == before.loc[last["game_id"], "avail_diff"]
    )
    assert np.isfinite(out.loc[last["game_id"], "avail_delta_home"])
