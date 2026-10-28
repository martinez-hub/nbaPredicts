"""Pre-game features for every game, final or upcoming.

Every value attached to a game is computed from games that started strictly
before it: rolling statistics are shifted by one game, Elo is read before the
game updates it, and player values come from earlier appearances only. The one
exception is availability (who suits up), which for past games is read from the
box score and for upcoming games from ESPN's injury report; both are known at
tip-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .espn import POSTSEASON

SHORT, LONG = 8, 30  # EWM spans, in games
ELO_START, ELO_K, ELO_HFA, ELO_CARRY = 1505.0, 20.0, 100.0, 0.75
ROTATION = 8  # players counted in a team's availability strength
REPLACEMENT_GMSC = 3.0  # value assumed for a player with no history

RATE_COLS = [
    "off_rtg",
    "def_rtg",
    "net_rtg",
    "pace",
    "efg",
    "opp_efg",
    "tov_pct",
    "opp_tov_pct",
    "orb_pct",
    "ftr",
    "fg3a_rate",
    "pts",
    "opp_pts",
]

FEATURES = [
    "elo_diff",
    "neutral",
    "playoff",
    "net_rtg_s_diff",
    "net_rtg_l_diff",
    "off_rtg_l_diff",
    "def_rtg_l_diff",
    "efg_l_diff",
    "opp_efg_l_diff",
    "tov_pct_l_diff",
    "opp_tov_pct_l_diff",
    "orb_pct_l_diff",
    "ftr_l_diff",
    "fg3a_rate_l_diff",
    "pace_l_mean",
    "win10_diff",
    "season_win_diff",
    "rest_home",
    "rest_away",
    "b2b_home",
    "b2b_away",
    "avail_diff",
    "avail_delta_home",
    "avail_delta_away",
    "season_games_min",
]


def team_games(games: pd.DataFrame, team_box: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) for final games, with per-game advanced stats."""
    finals = games[games["status"] == "final"][
        ["game_id", "start", "date", "season", "season_type", "neutral"]
    ]
    team_box = team_box.drop(columns=["season"], errors="ignore")
    tb = team_box.merge(finals, on="game_id", how="inner")
    stats = [
        c
        for c in team_box.columns
        if c not in ("game_id", "team_id", "opp_id", "is_home")
    ]
    opp = team_box[["game_id", "team_id", *stats]].rename(
        columns={"team_id": "opp_id", **{c: f"opp_{c}" for c in stats}}
    )
    tg = tb.merge(opp, on=["game_id", "opp_id"], how="inner")

    poss = tg["fga"] - tg["oreb"] + tg["tov"] + 0.44 * tg["fta"]
    opp_poss = tg["opp_fga"] - tg["opp_oreb"] + tg["opp_tov"] + 0.44 * tg["opp_fta"]
    # ESPN has the score but no box score for ~5% of 2012-16 games; those get
    # the season's median pace so the ratings still use the real score.
    avg_poss = (poss + opp_poss) / 2
    avg_poss = avg_poss.fillna(avg_poss.groupby(tg["season"]).transform("median"))
    tg = tg.assign(
        pace=avg_poss,
        off_rtg=100 * tg["pts"] / avg_poss,
        def_rtg=100 * tg["opp_pts"] / avg_poss,
        efg=(tg["fgm"] + 0.5 * tg["fg3m"]) / tg["fga"],
        opp_efg=(tg["opp_fgm"] + 0.5 * tg["opp_fg3m"]) / tg["opp_fga"],
        tov_pct=tg["tov"] / poss,
        opp_tov_pct=tg["opp_tov"] / opp_poss,
        orb_pct=tg["oreb"] / (tg["oreb"] + tg["opp_dreb"]),
        ftr=tg["ftm"] / tg["fga"],
        fg3a_rate=tg["fg3a"] / tg["fga"],
        win=(tg["pts"] > tg["opp_pts"]).astype(float),
    )
    tg["net_rtg"] = tg["off_rtg"] - tg["def_rtg"]
    keep = [
        "game_id",
        "start",
        "date",
        "season",
        "season_type",
        "neutral",
        "team_id",
        "opp_id",
        "is_home",
        "win",
        *RATE_COLS,
    ]
    return tg[keep].sort_values(["start", "game_id"]).reset_index(drop=True)


def _group_ewm(values: pd.Series, keys: pd.Series, span: int) -> pd.Series:
    """EWM mean of ``values`` within each ``keys`` group, aligned to ``values``."""
    return (
        values.groupby(keys, sort=False)
        .ewm(span=span, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(values.index)
    )


def _rolling_team_state(slots: pd.DataFrame) -> pd.DataFrame:
    """Shifted EWM statistics per team over a frame of past and upcoming slots.

    ``slots`` holds one row per (game, team), final rows carrying stats and
    upcoming rows NaN, sorted by start. The shift means a row only sees the
    rows before it.
    """
    slots = slots.sort_values(["start", "game_id"]).reset_index(drop=True)
    team = slots["team_id"]
    g = slots.groupby("team_id", sort=False)
    out = pd.DataFrame(index=slots.index)
    for col in RATE_COLS:
        prev = g[col].shift(1)
        out[f"{col}_s"] = _group_ewm(prev, team, SHORT)
        out[f"{col}_l"] = _group_ewm(prev, team, LONG)
    prev_win = g["win"].shift(1)
    out["win10"] = (
        prev_win.groupby(team, sort=False)
        .rolling(10, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    # Within-season record from earlier final games only: the in-season shift
    # leaves the opener NaN, and upcoming rows (no result) add nothing.
    season_key = [team, slots["season"]]
    prev_in_season = slots.groupby(["team_id", "season"], sort=False)["win"].shift(1)
    season_n = prev_in_season.notna().groupby(season_key, sort=False).cumsum()
    season_wins = prev_in_season.fillna(0.0).groupby(season_key, sort=False).cumsum()
    first_of_season = slots.groupby(["team_id", "season"], sort=False).cumcount() == 0
    out["season_games"] = season_n
    out["season_win"] = (season_wins / season_n.replace(0, np.nan)).fillna(0.5)
    prev_date = g["date"].shift(1)
    rest = (pd.to_datetime(slots["date"]) - pd.to_datetime(prev_date)).dt.days
    out["rest"] = rest.clip(upper=10).fillna(10).where(~first_of_season, 10)
    out["b2b"] = (out["rest"] == 1).astype(float)
    return pd.concat([slots, out], axis=1)


def elo_ratings(games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Pre-game Elo for every game in ``games`` (final ones update ratings).

    Returns per-game ``home_elo``/``away_elo`` and the ratings after the last
    final game.
    """
    ratings: dict[str, float] = {}
    season_of: dict[str, int] = {}
    rows = []
    ordered = games.sort_values(["start", "game_id"])
    for g in ordered.itertuples(index=False):
        for tid in (g.home_id, g.away_id):
            if tid not in ratings:
                ratings[tid] = ELO_START
            elif season_of.get(tid) != g.season:
                ratings[tid] = ELO_CARRY * ratings[tid] + (1 - ELO_CARRY) * ELO_START
            season_of[tid] = g.season
        h, a = ratings[g.home_id], ratings[g.away_id]
        rows.append((g.game_id, h, a))
        if g.status != "final":
            continue
        hfa = 0.0 if g.neutral else ELO_HFA
        expected = 1 / (1 + 10 ** (-(h + hfa - a) / 400))
        margin = g.home_pts - g.away_pts
        home_won = 1.0 if margin > 0 else 0.0
        winner_diff = (h + hfa - a) if margin > 0 else (a - h - hfa)
        mult = ((abs(margin) + 3) ** 0.8) / (7.5 + 0.006 * winner_diff)
        delta = ELO_K * mult * (home_won - expected)
        ratings[g.home_id] = h + delta
        ratings[g.away_id] = a - delta
    elo = pd.DataFrame(rows, columns=["game_id", "home_elo", "away_elo"])
    return elo, ratings


def player_values(player_box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each appearance with the player's value going in (prior) and after (post).

    Value is an EWM of Hollinger's Game Score over the player's earlier games,
    across teams and seasons.
    """
    pb = player_box[(~player_box["dnp"]) & (player_box["min"] > 0)].merge(
        games[["game_id", "start"]], on="game_id"
    )
    pb = pb.sort_values(["start", "game_id"]).reset_index(drop=True)
    gmsc = (
        pb["pts"]
        + 0.4 * pb["fgm"]
        - 0.7 * pb["fga"]
        - 0.4 * (pb["fta"] - pb["ftm"])
        + 0.7 * pb["oreb"]
        + 0.3 * pb["dreb"]
        + pb["stl"]
        + 0.7 * pb["ast"]
        + 0.7 * pb["blk"]
        - 0.4 * pb["pf"]
        - pb["tov"]
    )
    pb["gmsc"] = gmsc
    pb["post_value"] = _group_ewm(pb["gmsc"], pb["athlete_id"], 20)
    pb["prior_value"] = pb.groupby("athlete_id", sort=False)["post_value"].shift(1)
    return pb


def _top_sum(values: pd.Series, keys: list[pd.Series]) -> pd.Series:
    frame = pd.DataFrame({"v": values.fillna(REPLACEMENT_GMSC)})
    for i, k in enumerate(keys):
        frame[f"k{i}"] = k.values
    cols = [f"k{i}" for i in range(len(keys))]
    frame = frame.sort_values("v", ascending=False)
    return frame.groupby(cols, sort=False).head(ROTATION).groupby(cols)["v"].sum()


def current_rosters(
    pv: pd.DataFrame, recent_games: int = 10, espn: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Each team's current players with their latest value.

    With ``espn`` (ESPN's current team rosters), membership comes from it, so
    offseason moves count before the player has played for his new team; a
    player with no history gets the replacement value. Without it, a player
    belongs to the team of his latest appearance if that was one of the
    team's last ``recent_games`` games.
    """
    latest = pv.sort_values("start").groupby("athlete_id").tail(1)
    if espn is not None and not espn.empty:
        value = latest.set_index("athlete_id")["post_value"]
        roster = espn[["athlete_id", "name", "team_id"]].copy()
        roster["post_value"] = roster["athlete_id"].map(value).fillna(REPLACEMENT_GMSC)
        return roster
    team_games_ = pv[["team_id", "game_id", "start"]].drop_duplicates(
        ["team_id", "game_id"]
    )
    recent = team_games_.sort_values("start").groupby("team_id").tail(recent_games)
    roster = latest.merge(recent[["team_id", "game_id"]], on=["team_id", "game_id"])
    return roster[["athlete_id", "name", "team_id", "post_value"]]


def expected_top_sum(
    values: np.ndarray, p_play: np.ndarray, samples: int = 512, seed: int = 0
) -> float:
    """Expected sum of the ``ROTATION`` best values among players who suit up.

    Each player plays independently with probability ``p_play``. Weighting
    values by ``p_play`` before choosing the top players would let a 50/50
    star drop below a healthy reserve and count for nothing; sampling who
    plays gives the real expectation.
    """
    if len(values) == 0:
        return float("nan")
    values = np.nan_to_num(values, nan=REPLACEMENT_GMSC)
    rng = np.random.default_rng(seed)
    plays = rng.random((samples, len(values))) < p_play
    played = np.where(plays, values, -np.inf)
    top = -np.sort(-played, axis=1)[:, :ROTATION]
    return float(np.where(np.isfinite(top), top, 0.0).sum(axis=1).mean())


def availability(
    pv: pd.DataFrame,
    upcoming: pd.DataFrame,
    injuries: pd.DataFrame | None,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Top-``ROTATION`` summed player value for each (game, team).

    Past games: the players who actually played, valued before the game.
    Upcoming games: the current roster (see ``current_rosters``), each player's
    value weighted by his chance of playing (``p_play`` from the injury
    report; without it, players listed Out are dropped).
    """
    past = _top_sum(pv["prior_value"], [pv["game_id"], pv["team_id"]])
    past = (
        past.rename("avail")
        .reset_index()
        .rename(columns={"k0": "game_id", "k1": "team_id"})
    )
    if upcoming.empty:
        return past

    roster = rosters.copy() if rosters is not None else current_rosters(pv)
    roster["p_play"] = 1.0
    if injuries is not None and not injuries.empty:
        if "p_play" in injuries:
            p_play = injuries.set_index("athlete_id")["p_play"]
            roster["p_play"] = roster["athlete_id"].map(p_play).fillna(1.0)
        else:
            out = injuries["status"].str.lower() == "out"
            roster.loc[
                roster["athlete_id"].isin(injuries.loc[out, "athlete_id"]), "p_play"
            ] = 0.0
    strength = {
        tid: expected_top_sum(
            rows["post_value"].to_numpy(float), rows["p_play"].to_numpy(float)
        )
        for tid, rows in roster.groupby("team_id")
    }
    rows = []
    for g in upcoming.itertuples(index=False):
        for tid in (g.home_id, g.away_id):
            rows.append((g.game_id, tid, strength.get(tid, np.nan)))
    future = pd.DataFrame(rows, columns=["game_id", "team_id", "avail"])
    return pd.concat([past, future], ignore_index=True)


def team_state(
    games: pd.DataFrame,
    team_box: pd.DataFrame,
    pv: pd.DataFrame,
    upcoming_ids: list[str],
    injuries: pd.DataFrame | None = None,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (game, team), final and upcoming, with pre-game team state.

    Final rows also carry that game's own statistics (``pts``, ``off_rtg``...);
    the pre-game columns end in ``_s``/``_l`` or are ``win10``, ``season_win``,
    ``season_games``, ``rest``, ``b2b``, ``avail`` and ``avail_delta``.
    """
    upcoming = games[games["game_id"].isin(upcoming_ids)]
    tg = team_games(games, team_box)

    future_slots = []
    for g in upcoming.itertuples(index=False):
        for tid, oid, home in (
            (g.home_id, g.away_id, True),
            (g.away_id, g.home_id, False),
        ):
            future_slots.append(
                {
                    "game_id": g.game_id,
                    "start": g.start,
                    "date": g.date,
                    "season": g.season,
                    "season_type": g.season_type,
                    "neutral": g.neutral,
                    "team_id": tid,
                    "opp_id": oid,
                    "is_home": home,
                }
            )
    slots = pd.concat([tg, pd.DataFrame(future_slots)], ignore_index=True)
    slots["date"] = pd.to_datetime(slots["date"])
    state = _rolling_team_state(slots)

    avail = availability(pv, upcoming, injuries, rosters)
    state = state.merge(avail, on=["game_id", "team_id"], how="left")
    state = state.sort_values(["start", "game_id"]).reset_index(drop=True)
    state["avail_typical"] = state.groupby("team_id")["avail"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    state["avail_delta"] = state["avail"] - state["avail_typical"]
    state["is_home"] = state["is_home"].astype(bool)
    return state


def build_features(
    games: pd.DataFrame,
    state: pd.DataFrame,
    upcoming_ids: list[str] | None = None,
) -> pd.DataFrame:
    """One row per game (final, plus the requested upcoming ones) with FEATURES.

    ``state`` is ``team_state``'s output for the same games. Final games also
    carry ``home_win``, ``home_pts`` and ``away_pts``.
    """
    upcoming_ids = upcoming_ids or []
    finals = games[games["status"] == "final"]
    upcoming = games[games["game_id"].isin(upcoming_ids)]

    cols = [c for c in state.columns if c.endswith(("_s", "_l"))] + [
        "win10",
        "season_win",
        "season_games",
        "rest",
        "b2b",
        "avail",
        "avail_delta",
    ]
    is_home = state["is_home"]
    home = state[is_home].set_index("game_id")[cols].add_prefix("h_")
    away = state[~is_home].set_index("game_id")[cols].add_prefix("a_")

    base = pd.concat([finals, upcoming], ignore_index=True)
    elo, _ = elo_ratings(base)
    df = base.merge(elo, on="game_id").set_index("game_id").join(home).join(away)
    df = df.reset_index()

    out = pd.DataFrame({"game_id": df["game_id"]})
    out["elo_diff"] = df["home_elo"] - df["away_elo"]
    out["neutral"] = df["neutral"].astype(float)
    out["playoff"] = (df["season_type"] == POSTSEASON).astype(float)
    for col in (
        "net_rtg_s",
        "net_rtg_l",
        "off_rtg_l",
        "def_rtg_l",
        "efg_l",
        "opp_efg_l",
        "tov_pct_l",
        "opp_tov_pct_l",
        "orb_pct_l",
        "ftr_l",
        "fg3a_rate_l",
    ):
        out[f"{col}_diff"] = df[f"h_{col}"] - df[f"a_{col}"]
    out["pace_l_mean"] = (df["h_pace_l"] + df["a_pace_l"]) / 2
    out["win10_diff"] = df["h_win10"] - df["a_win10"]
    out["season_win_diff"] = df["h_season_win"] - df["a_season_win"]
    out["rest_home"], out["rest_away"] = df["h_rest"], df["a_rest"]
    out["b2b_home"], out["b2b_away"] = df["h_b2b"], df["a_b2b"]
    out["avail_diff"] = df["h_avail"] - df["a_avail"]
    out["avail_delta_home"] = df["h_avail_delta"].fillna(0.0)
    out["avail_delta_away"] = df["a_avail_delta"].fillna(0.0)
    out["season_games_min"] = np.minimum(df["h_season_games"], df["a_season_games"])

    meta = df[
        [
            "game_id",
            "start",
            "date",
            "season",
            "season_type",
            "status",
            "home_id",
            "away_id",
            "home_abbr",
            "away_abbr",
            "home_pts",
            "away_pts",
            "home_elo",
            "away_elo",
            "h_avail",
            "a_avail",
        ]
    ]
    out = meta.merge(out, on="game_id")
    out["home_win"] = np.where(
        out["status"] == "final",
        (out["home_pts"] > out["away_pts"]).astype(float),
        np.nan,
    )
    return out.sort_values(["start", "game_id"]).reset_index(drop=True)


def pregame_availability(
    pv: pd.DataFrame,
    games: pd.DataFrame,
    reports: pd.DataFrame,
    team_ids: dict[str, str],
    window: int = 10,
    samples: int = 256,
) -> pd.DataFrame:
    """Expected top-``ROTATION`` strength as it could be known before each game.

    Roster: the players who appeared for the team in its previous ``window``
    games, each valued as of his latest appearance before this game. Chance
    of playing: that day's official injury report (Out 0, Doubtful 0.25,
    Questionable 0.5, Probable 0.9, Available 1); players it does not list,
    and every player on days without a report, count as available. Unlike
    the box-score availability, nothing here is learned from the game itself.
    ``team_ids`` maps the report's full team names to ESPN team ids.
    """
    from .injuries import P_PLAY, normalize_name

    finals = games[games["status"] == "final"][["game_id", "start", "date"]]
    app = pv[["game_id", "team_id", "athlete_id", "name", "post_value"]].merge(
        finals, on="game_id"
    )
    team_games_ = (
        app[["team_id", "game_id", "start", "date"]]
        .drop_duplicates(["team_id", "game_id"])
        .sort_values("start")
    )
    team_games_["k"] = team_games_.groupby("team_id").cumcount()
    app = app.merge(team_games_[["team_id", "game_id", "k"]], on=["team_id", "game_id"])

    # Each appearance counts toward the team's next ``window`` games.
    spread = app.loc[app.index.repeat(window)].copy()
    spread["target_k"] = spread["k"] + np.tile(np.arange(1, window + 1), len(app))
    latest = (
        spread.sort_values("k").groupby(["team_id", "target_k", "athlete_id"]).tail(1)
    )
    target = team_games_.rename(columns={"k": "target_k"})[
        ["team_id", "target_k", "game_id", "date"]
    ]
    roster = latest[["team_id", "target_k", "athlete_id", "name", "post_value"]].merge(
        target, on=["team_id", "target_k"]
    )

    rep = reports.copy()
    rep["team_id"] = rep["team"].map(team_ids)
    rep["key"] = rep["player"].map(normalize_name)
    # A report lists today's and tomorrow's games: join on each row's own
    # game date, and when two reports cover a game keep the later one.
    game_day = rep["game_date"] if "game_date" in rep else rep["day"]
    rep["date"] = pd.to_datetime(game_day.fillna(rep["day"])).dt.date
    rep["p_play"] = rep["status"].str.lower().map(P_PLAY).fillna(1.0)
    order = rep["published"] if "published" in rep else rep["day"]
    rep = rep.assign(_order=order).sort_values("_order")
    rep = rep.drop_duplicates(["team_id", "date", "key"], keep="last")
    roster["key"] = roster["name"].map(normalize_name)
    roster["date"] = pd.to_datetime(roster["date"]).dt.date
    roster = roster.merge(
        rep[["team_id", "date", "key", "p_play"]],
        on=["team_id", "date", "key"],
        how="left",
    )
    roster["p_play"] = roster["p_play"].fillna(1.0)

    rows = [
        (
            gid,
            tid,
            expected_top_sum(
                g["post_value"].to_numpy(float), g["p_play"].to_numpy(float), samples
            ),
        )
        for (gid, tid), g in roster.groupby(["game_id", "team_id"], sort=False)
    ]
    return pd.DataFrame(rows, columns=["game_id", "team_id", "avail_pre"])


def apply_pregame_availability(feats: pd.DataFrame, pre: pd.DataFrame) -> pd.DataFrame:
    """Replace the availability features with ``pregame_availability``'s values.

    For finished games, ``avail_diff`` becomes home minus away pre-game
    strength and ``avail_delta_*`` each team's pre-game strength minus its
    average over its previous ten games. Upcoming games keep their live
    strength (roster and today's report, already the pre-game definition);
    their delta is taken against the same pre-game average.
    """
    pre = pre.merge(feats[["game_id", "start"]], on="game_id").sort_values("start")
    typical = pre.groupby("team_id")["avail_pre"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    recent = pre.groupby("team_id")["avail_pre"].apply(lambda s: s.tail(10).mean())
    pre = pre.assign(delta=(pre["avail_pre"] - typical).fillna(0.0)).set_index(
        ["game_id", "team_id"]
    )
    out = feats.copy()
    for side in ("home", "away"):
        keys = list(zip(out["game_id"], out[f"{side}_id"], strict=True))
        value = pre["avail_pre"].reindex(keys).to_numpy()
        delta = pre["delta"].reindex(keys).to_numpy()
        live = out[f"{side[0]}_avail"].to_numpy(dtype=float)
        live_delta = live - out[f"{side}_id"].map(recent).to_numpy(dtype=float)
        has = ~np.isnan(value)
        out[f"_{side}_pre"] = np.where(has, value, live)
        out[f"avail_delta_{side}"] = np.nan_to_num(np.where(has, delta, live_delta))
    both = out["_home_pre"].notna() & out["_away_pre"].notna()
    out["avail_diff"] = np.where(
        both, out["_home_pre"] - out["_away_pre"], out["avail_diff"]
    )
    return out.drop(columns=["_home_pre", "_away_pre"])
