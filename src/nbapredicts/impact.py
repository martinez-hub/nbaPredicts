"""Player impact ratings and minutes-weighted team strength.

Ratings are adjusted plus-minus, the average of two ridge fits with older
games down-weighted:

- game level: each game's home margin is home advantage plus the sum over
  players of (minutes / 48) x rating, home players positive and away
  players negative;
- stint level: the same for each stint (``stints.py``), the stretches with
  the same ten players on the floor, weighted by their length; a game whose
  stints did not pass the box-score check enters as its whole-game row.

Both are in points of margin per 48 minutes on the floor and are refit at
the start of every month from games before it, so a game's ratings never
saw it. The average predicted EQUINE's results better than either fit
alone (docs/results/README.md, "Stint-level RAPM").

Team strength for a game is the expected sum of rating x minutes share over
the players who suit up: each player's minutes are projected from his recent
games, availability is sampled from the injury report, and the available
players' minutes are rescaled to fill the game's 240. Strength is in points
of margin, the unit a betting line moves in.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import solve

from .collect import LINEUP_COLS

MINUTES_SPAN = 10  # games in the projected-minutes EWM
TEAM_MINUTES = 240.0
GAME_RIDGE = 5.0
STINT_RIDGE = 45.0  # stint rows are many and noisy (chosen on 2016-17..2022-23)


def _utc(start: pd.Series) -> np.ndarray:
    """Tip-off times as naive UTC datetime64, for sorting and searching."""
    return start.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")


def _design(app: pd.DataFrame, games: pd.DataFrame, players: pd.Index):
    """Sparse (game x player) matrix of signed minutes/48, the margins and
    tip-off times; the last column is home advantage."""
    g = games.reset_index(drop=True)
    row = pd.Series(np.arange(len(g)), index=g["game_id"])
    a = app[app["game_id"].isin(g["game_id"])]
    home = (
        a["team_id"].to_numpy()
        == a["game_id"].map(g.set_index("game_id")["home_id"]).to_numpy()
    )
    sign = np.where(home, 1.0, -1.0)
    x = sparse.csr_matrix(
        (
            sign * a["min"].to_numpy(float) / 48.0,
            (row[a["game_id"]].to_numpy(), players.get_indexer(a["athlete_id"])),
        ),
        shape=(len(g), len(players)),
    )
    hca = np.where(g["neutral"].astype(bool), 0.0, 1.0)
    y = (g["home_pts"] - g["away_pts"]).to_numpy(float)
    x = sparse.hstack([x, sparse.csr_matrix(hca[:, None])]).tocsr()
    return x, y, _utc(g["start"])


def _stint_design(stints: pd.DataFrame, games: pd.DataFrame, players: pd.Index):
    """``_design`` for stints. A stint's margin varies with its length, so
    each row is weighted by it: x = +-sqrt(minutes/48) per player on the
    floor and y = margin / sqrt(minutes/48), which keeps the game rows'
    units."""
    g = games.set_index("game_id")
    s = stints[stints["game_id"].isin(g.index)]
    root = np.sqrt((s["t1"] - s["t0"]).to_numpy(float) / 60 / 48)
    m = len(s)
    idx = np.column_stack([players.get_indexer(s[c]) for c in LINEUP_COLS])
    val = np.column_stack([root] * 5 + [-root] * 5)
    keep = idx >= 0
    rows = np.repeat(np.arange(m), len(LINEUP_COLS)).reshape(idx.shape)
    neutral = g.loc[s["game_id"], "neutral"].astype(bool).to_numpy()
    x = sparse.csr_matrix(
        (
            np.concatenate([val[keep], np.where(neutral, 0.0, root)]),
            (
                np.concatenate([rows[keep], np.arange(m)]),
                np.concatenate([idx[keep], np.full(m, len(players))]),
            ),
        ),
        shape=(m, len(players) + 1),
    )
    y = s["margin"].to_numpy(float) / root
    return x, y, _utc(g.loc[s["game_id"], "start"])


def _monthly_fit(
    x: sparse.csr_matrix,
    y: np.ndarray,
    t: np.ndarray,
    months: pd.DatetimeIndex,
    ridge: float,
    half_life_days: float,
) -> list[np.ndarray]:
    """Ridge solution as of each month from the rows before it, each row
    weighted 0.5 ** (age / half-life). The weighted normal equations as of
    a month are the previous month's decayed by the gap plus the rows in
    between, so each month costs one solve."""
    order = np.argsort(t, kind="stable")
    x, y, t = x[order], y[order], t[order]
    n = x.shape[1]
    gram, xty = np.zeros((n, n)), np.zeros(n)
    out, prev, begin = [], None, 0
    for month in months:
        m64 = month.tz_convert("UTC").tz_localize(None).to_datetime64()
        stop = int(np.searchsorted(t, m64))
        if prev is not None:
            decay = 0.5 ** ((month - prev).days / half_life_days)
            gram *= decay
            xty *= decay
        if stop > begin:
            age = (m64 - t[begin:stop]) / np.timedelta64(1, "D")
            w = 0.5 ** (age / half_life_days)
            xs = x[begin:stop]
            gram += (xs.T @ xs.multiply(w[:, None])).toarray()
            xty += xs.T @ (w * y[begin:stop])
        prev, begin = month, stop
        out.append(solve(gram + ridge * np.eye(n), xty, assume_a="pos"))
    return out


def monthly_ratings(
    player_box: pd.DataFrame,
    games: pd.DataFrame,
    stints: pd.DataFrame | None = None,
    ridge: float = GAME_RIDGE,
    stint_ridge: float = STINT_RIDGE,
    half_life_days: float = 730.0,
    min_games: int = 500,
) -> pd.DataFrame:
    """Ratings as of the first of each month: (month, athlete_id, rating).

    With ``stints`` (the ``stints`` table), the average of the game-level
    and stint-level fits; without, the game-level fit alone.
    """
    finals = games[games["status"] == "final"].copy()
    finals["start"] = pd.to_datetime(finals["start"], utc=True)
    app = player_box[(~player_box["dnp"]) & (player_box["min"] > 0)][
        ["game_id", "team_id", "athlete_id", "min"]
    ]
    players = pd.Index(app["athlete_id"].unique())
    months = pd.date_range(
        finals["start"].min().normalize().replace(day=1) + pd.offsets.MonthBegin(1),
        finals["start"].max() + pd.offsets.MonthBegin(1),
        freq="MS",
        tz="UTC",
    )
    starts = np.sort(_utc(finals["start"]))
    months = months[
        [
            np.searchsorted(starts, m.tz_localize(None).to_datetime64()) >= min_games
            for m in months
        ]
    ]
    if months.empty:
        return pd.DataFrame(columns=["month", "athlete_id", "rating"])
    fits = [_monthly_fit(*_design(app, finals, players), months, ridge, half_life_days)]
    if stints is not None and not stints.empty:
        rated = finals["game_id"].isin(set(stints["game_id"]))
        parts = [
            _stint_design(stints, finals, players),
            _design(app, finals[~rated], players),
        ]
        x, y, t = (
            sparse.vstack([p[0] for p in parts]).tocsr(),
            np.concatenate([p[1] for p in parts]),
            np.concatenate([p[2] for p in parts]),
        )
        fits.append(_monthly_fit(x, y, t, months, stint_ridge, half_life_days))
    first = (
        app.merge(finals[["game_id", "start"]], on="game_id")
        .groupby("athlete_id")["start"]
        .min()
        .reindex(players)
    )
    out = []
    for k, month in enumerate(months):
        seen = (first < month).to_numpy()
        rating = np.mean([f[k][:-1] for f in fits], axis=0)
        out.append(
            pd.DataFrame(
                {"month": month, "athlete_id": players[seen], "rating": rating[seen]}
            )
        )
    return pd.concat(out, ignore_index=True)


def projected_minutes(player_box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Each appearance with the player's projected minutes going into the next game."""
    pb = player_box[(~player_box["dnp"]) & (player_box["min"] > 0)].merge(
        games[["game_id", "start"]], on="game_id"
    )
    pb = pb.sort_values(["start", "game_id"]).reset_index(drop=True)
    pb["proj_min"] = (
        pb.groupby("athlete_id", sort=False)["min"]
        .ewm(span=MINUTES_SPAN, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
        .reindex(pb.index)
    )
    return pb[["game_id", "team_id", "athlete_id", "start", "proj_min"]]


def expected_strength(
    rating: np.ndarray,
    minutes: np.ndarray,
    p_play: np.ndarray,
    samples: int = 256,
    seed: int = 0,
) -> float:
    """E[sum of rating x minutes/48] with minutes rescaled to fill 240."""
    if len(rating) == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    plays = rng.random((samples, len(rating))) < p_play
    mins = np.where(plays, minutes, 0.0)
    total = mins.sum(axis=1, keepdims=True)
    share = np.divide(
        mins * TEAM_MINUTES, total, out=np.zeros_like(mins), where=total > 0
    )
    share = np.minimum(share, 48.0)  # nobody plays more than the whole game
    return float((share / 48.0 * rating).sum(axis=1).mean())


def pregame_impact(
    pv: pd.DataFrame,
    player_box: pd.DataFrame,
    games: pd.DataFrame,
    reports: pd.DataFrame,
    team_ids: dict[str, str],
    ratings: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """Expected minutes-weighted strength of each team going into each game.

    Rating: the player's as of the first of the game's month (fit on earlier
    games only); a player without one counts as average (0). Minutes: his
    projected minutes after his latest appearance. Availability: as in
    ``features.pregame_rosters``.
    """
    from .features import pregame_rosters

    mins = projected_minutes(player_box, games)
    app = pv.merge(
        mins[["game_id", "athlete_id", "proj_min"]], on=["game_id", "athlete_id"]
    )
    roster = pregame_rosters(
        app, games, reports, team_ids, window, columns=("proj_min",)
    )
    roster["month"] = (
        pd.to_datetime(roster["start"], utc=True)
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp()
        .dt.tz_localize("UTC")
    )
    roster = roster.merge(ratings, on=["month", "athlete_id"], how="left")
    roster["rating"] = roster["rating"].fillna(0.0)
    rows = [
        (
            gid,
            tid,
            expected_strength(
                g["rating"].to_numpy(float),
                g["proj_min"].to_numpy(float),
                g["p_play"].to_numpy(float),
            ),
        )
        for (gid, tid), g in roster.groupby(["game_id", "team_id"], sort=False)
    ]
    return pd.DataFrame(rows, columns=["game_id", "team_id", "impact_pre"])


IMPACT_FEATURES = ["impact_diff", "impact_delta_home", "impact_delta_away"]


def live_impact(
    rosters: pd.DataFrame,
    player_box: pd.DataFrame,
    games: pd.DataFrame,
    ratings: pd.DataFrame,
    injuries: pd.DataFrame | None,
    upcoming: pd.DataFrame,
) -> pd.DataFrame:
    """Expected strength for upcoming games from today's rosters and report.

    Ratings from the latest monthly fit, each player's projected minutes after
    his latest appearance (none yet: no minutes), chance of playing from the
    combined injury report.
    """
    mins = projected_minutes(player_box, games)
    latest_min = mins.sort_values("start").groupby("athlete_id")["proj_min"].last()
    rating = ratings[ratings["month"] == ratings["month"].max()].set_index(
        "athlete_id"
    )["rating"]
    r = rosters.copy()
    r["proj_min"] = r["athlete_id"].map(latest_min).fillna(0.0)
    r["rating"] = r["athlete_id"].map(rating).fillna(0.0)
    p_play = (
        injuries.set_index("athlete_id")["p_play"]
        if injuries is not None and not injuries.empty and "p_play" in injuries
        else pd.Series(dtype=float)
    )
    r["p_play"] = r["athlete_id"].map(p_play).fillna(1.0)
    strength = {
        tid: expected_strength(
            g["rating"].to_numpy(float),
            g["proj_min"].to_numpy(float),
            g["p_play"].to_numpy(float),
        )
        for tid, g in r.groupby("team_id")
    }
    rows = [
        (g.game_id, tid, strength.get(tid, np.nan))
        for g in upcoming.itertuples(index=False)
        for tid in (g.home_id, g.away_id)
    ]
    return pd.DataFrame(rows, columns=["game_id", "team_id", "impact_pre"])


def apply_impact(feats: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
    """Add impact_diff and each team's change from its last ten games."""
    imp = imp.merge(feats[["game_id", "start"]], on="game_id").sort_values("start")
    typical = imp.groupby("team_id")["impact_pre"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=1).mean()
    )
    imp = imp.assign(delta=(imp["impact_pre"] - typical).fillna(0.0)).set_index(
        ["game_id", "team_id"]
    )
    out = feats.copy()
    for side in ("home", "away"):
        keys = list(zip(out["game_id"], out[f"{side}_id"], strict=True))
        out[f"_{side}"] = imp["impact_pre"].reindex(keys).to_numpy()
        out[f"impact_delta_{side}"] = imp["delta"].reindex(keys).fillna(0.0).to_numpy()
    out["impact_diff"] = (out["_home"] - out["_away"]).fillna(0.0)
    return out.drop(columns=["_home", "_away"])
