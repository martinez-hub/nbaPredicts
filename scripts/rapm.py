"""Stint-level RAPM from ESPN play-by-play (experiment).

The impact features rate players by game-level adjusted plus-minus: one row
per game, each player's minutes against the final margin
(``impact.monthly_ratings``). Play-by-play splits each game into stints, the
stretches between substitutions with the same ten players on the floor,
about 30 a game, so a player's rating is fit on who was on the court for
each scoring run instead of the whole game's margin.

Steps:

    uv run python scripts/rapm.py fetch [--smoke]   # play-by-play, cached
    uv run python scripts/rapm.py stints            # stints + minute check
    uv run python scripts/rapm.py tune              # ridge on pre-test months
    uv run python scripts/rapm.py rate              # next-month rating quality
    uv run python scripts/rapm.py equine [--ratings average]  # EQUINE input

``fetch`` asks ESPN's summary endpoint once per finished game (the same
volume as the box-score history) and keeps only what stints need.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "cache.nosync/rapm"
PBP = ROOT / "cache.nosync/pbp"
SUB = "584"  # ESPN's play type id for a substitution
RNG = np.random.default_rng(0)

log = logging.getLogger("rapm")


def clock_seconds(display: str) -> float:
    """'11:36' -> 696.0; '45.2' (the last minute) -> 45.2."""
    if ":" in display:
        m, s = display.split(":", 1)
        return int(m) * 60 + float(s)
    return float(display or 0)


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    """What stints need from a summary: starters and a slim play list."""
    comp = summary["header"]["competitions"][0]
    home = next(c["team"]["id"] for c in comp["competitors"] if c["homeAway"] == "home")
    starters = {}
    for team in summary["boxscore"].get("players", []):
        athletes = team["statistics"][0]["athletes"] if team["statistics"] else []
        starters[team["team"]["id"]] = [
            a["athlete"]["id"] for a in athletes if a.get("starter")
        ]
    plays = [
        [
            int(p.get("sequenceNumber") or 0),
            int(p["period"]["number"]),
            clock_seconds(p.get("clock", {}).get("displayValue", "0")),
            p.get("type", {}).get("id", ""),
            (p.get("team") or {}).get("id"),
            [a["athlete"]["id"] for a in p.get("participants", []) if "athlete" in a],
            int(p.get("homeScore") or 0),
            int(p.get("awayScore") or 0),
        ]
        for p in summary.get("plays", [])
    ]
    return {"home_id": home, "starters": starters, "plays": plays}


def pbp_path(game_id: str) -> Path:
    return PBP / game_id[-2:] / f"{game_id}.json.gz"


def load_pbp(game_id: str) -> dict[str, Any] | None:
    path = pbp_path(game_id)
    if not path.exists():
        return None
    with gzip.open(path, "rt") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# fetch
# --------------------------------------------------------------------------


def fetch(smoke: bool, workers: int) -> None:
    from nbapredicts.collect import Store
    from nbapredicts.espn import ESPNClient

    games = Store(ROOT / "data").load("games")
    finals = games[games["status"] == "final"].sort_values("start")
    ids = finals["game_id"].tolist()
    if smoke:
        picks = []
        for season in (2013, 2016, 2019, 2022, 2025):
            part = finals[finals["season"] == season]
            picks += part.sample(10, random_state=0)["game_id"].tolist()
        ids = picks
    todo = [g for g in ids if not pbp_path(g).exists()]
    log.info("%d games, %d to fetch", len(ids), len(todo))
    client = ESPNClient(cache_dir=None)
    raw_checks = []

    def one(gid: str) -> str:
        data = client.get("summary", event=gid)
        if smoke:
            # Substitution participants: entering first, leaving second.
            for p in data.get("plays", []):
                if p.get("type", {}).get("id") == SUB and len(p["participants"]) == 2:
                    raw_checks.append((gid, p["text"]))
                    break
        out = compact(data)
        path = pbp_path(gid)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as f:
            json.dump(out, f, separators=(",", ":"))
        tmp.replace(path)
        return gid

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(one, g): g for g in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as err:  # one bad game should not stop the pull
                failed.append(futures[fut])
                log.warning("%s failed: %s", futures[fut], err)
            if i % 500 == 0:
                log.info("%d/%d", i, len(todo))
    log.info("done, %d failed", len(failed))
    if smoke:
        smoke_report(ids, raw_checks)


def smoke_report(ids: list[str], raw_checks: list[tuple[str, str]]) -> None:
    from nbapredicts.collect import Store

    games = Store(ROOT / "data").load("games").set_index("game_id")
    rows = []
    for gid in ids:
        d = load_pbp(gid)
        if d is None:
            continue
        plays = d["plays"]
        subs = [p for p in plays if p[3] == SUB]
        last = plays[-1] if plays else None
        g = games.loc[gid]
        rows.append(
            {
                "season": int(g["season"]),
                "plays": len(plays),
                "subs": len(subs),
                "subs_with_two": sum(len(p[5]) == 2 for p in subs),
                "periods": max((p[1] for p in plays), default=0),
                "starters": [len(v) for v in d["starters"].values()],
                "final_score_matches": bool(
                    last is not None
                    and last[6] == g["home_pts"]
                    and last[7] == g["away_pts"]
                ),
            }
        )
    df = pd.DataFrame(rows)
    print(
        df.groupby("season")
        .agg(
            games=("plays", "size"),
            plays=("plays", "mean"),
            subs=("subs", "mean"),
            subs_with_two=("subs_with_two", "mean"),
            final_score=("final_score_matches", "mean"),
        )
        .round(1)
        .to_string()
    )
    print("starters per team:", df["starters"].map(tuple).value_counts().to_dict())
    for gid, text in raw_checks[:6]:
        print(gid, text)


# --------------------------------------------------------------------------
# stints
# --------------------------------------------------------------------------


def period_length(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def period_start(period: int) -> float:
    return sum(period_length(p) for p in range(1, period))


def build_stints(
    game: dict[str, Any], team_of: dict[str, str]
) -> tuple[list[tuple], dict[str, Any]]:
    """Stints of one game, and what went wrong if anything.

    A stint: (start s, end s, home lineup, away lineup, home pts, away pts).
    Period 1 opens with the box score's starters. Later periods open with
    the players who act in the period before entering it (older feeds do
    not log substitutions between periods), filled to five from those on
    the floor at the end of the period before who are not seen entering.
    Substitutions at the same moment change the lineup at once; points
    scored at that moment after the change count for the new lineup.
    """
    home = game["home_id"]
    away = next(t for t in game["starters"] if t != home)
    plays = [p for p in game["plays"] if p[1] >= 1]
    issues: dict[str, Any] = {}
    by_period: dict[int, list] = {}
    for p in plays:
        by_period.setdefault(p[1], []).append(p)
    on = {t: list(game["starters"].get(t, [])) for t in (home, away)}
    stints: list[tuple] = []
    score = (0, 0)
    for period in sorted(by_period):
        rows = by_period[period]
        if period > 1:
            first: dict[str, list[str]] = {home: [], away: []}
            entered: set[str] = set()
            for p in rows:
                if p[3] == SUB and len(p[5]) == 2:
                    came, left = p[5]
                    t = team_of.get(left)
                    if left not in entered and t in first and left not in first[t]:
                        first[t].append(left)
                    entered.add(came)
                    continue
                for a in p[5]:
                    t = team_of.get(a)
                    if a not in entered and t in first and a not in first[t]:
                        first[t].append(a)
            for t in (home, away):
                for a in on[t]:
                    if len(first[t]) >= 5:
                        break
                    if a not in first[t] and a not in entered:
                        first[t].append(a)
                if len(first[t]) != 5:
                    issues.setdefault("period_start", []).append((period, t))
                on[t] = first[t][:5]
        base, length = period_start(period), period_length(period)
        t0 = base
        s0 = score
        for i, p in enumerate(rows):
            now = base + length - p[2]
            score = (p[6], p[7])
            if p[3] != SUB or len(p[5]) not in (1, 2):
                continue
            if len(p[5]) == 2:
                came, left = p[5]
            else:
                # "enters the game for X" with the entering player missing:
                # the first teammate to act off the floor came in here.
                left = p[5][0]
                came = next_off_floor(rows[i + 1 :], team_of, left, on)
                if came is None:
                    issues["unknown_entry"] = issues.get("unknown_entry", 0) + 1
                    continue
            t = team_of.get(left, p[4])
            if t not in on:
                continue
            if now > t0:
                stints.append(
                    (
                        t0,
                        now,
                        tuple(on[home]),
                        tuple(on[away]),
                        score[0] - s0[0],
                        score[1] - s0[1],
                    )
                )
                t0, s0 = now, score
            if left in on[t]:
                on[t] = [came if a == left else a for a in on[t]]
            else:
                issues["sub_not_on_floor"] = issues.get("sub_not_on_floor", 0) + 1
                if came not in on[t]:
                    on[t] = [*on[t], came][-5:]
        end = base + length
        if end > t0:
            stints.append(
                (
                    t0,
                    end,
                    tuple(on[home]),
                    tuple(on[away]),
                    score[0] - s0[0],
                    score[1] - s0[1],
                )
            )
    return stints, issues


def next_off_floor(
    rest: list, team_of: dict[str, str], left: str, on: dict[str, list[str]]
) -> str | None:
    """The first player of ``left``'s team to act while off the floor."""
    team = team_of.get(left)
    if team not in on:
        return None
    came_in: set[str] = set()
    for q in rest:
        if q[3] == SUB and len(q[5]) == 2:
            q_came, q_left = q[5]
            if team_of.get(q_left) == team and q_left not in on[team]:
                if q_left != left and q_left not in came_in:
                    return q_left
            came_in.add(q_came)
            continue
        for a in q[5]:
            if team_of.get(a) == team and a not in on[team] and a != left:
                if a not in came_in:
                    return a
    return None


def reconcile(
    stints: list[tuple], box: pd.DataFrame, final: tuple[int, int]
) -> dict[str, Any]:
    """Rebuilt minutes and points against the box score."""
    mins: dict[str, float] = {}
    for t0, t1, h, a, _, _ in stints:
        for p in (*h, *a):
            mins[p] = mins.get(p, 0.0) + (t1 - t0) / 60
    # Some 2012-13 box scores have no minutes: then only the roster checks.
    has_minutes = bool(box["min"].notna().any())
    played = box[(~box["dnp"]) & (box["min"] > 0)] if has_minutes else box
    err = np.array(
        [
            abs(mins.get(a, 0.0) - m)
            for a, m in zip(played["athlete_id"], played["min"], strict=True)
        ]
    )
    # On the floor for under a minute with 0 in the box (rounded down).
    extra = {a for a in set(mins) - set(played["athlete_id"]) if mins[a] > 1.0}
    if not has_minutes:
        err = np.array([])
    pts = (sum(s[4] for s in stints), sum(s[5] for s in stints))
    sizes_ok = all(len(s[2]) == 5 and len(s[3]) == 5 for s in stints)
    return {
        "mean_err": float(err.mean()) if len(err) else np.nan,
        "max_err": float(err.max()) if len(err) else np.nan,
        "extra_players": len(extra),
        "points_ok": pts == final,
        "sizes_ok": sizes_ok,
        "has_minutes": has_minutes,
    }


def stints_step() -> None:
    from nbapredicts.collect import Store

    store = Store(ROOT / "data")
    games = store.load("games")
    pb = store.load("player_box")
    finals = games[games["status"] == "final"].set_index("game_id")
    box_by_game = {g: part for g, part in pb.groupby("game_id")}
    rows, checks = [], []
    for i, gid in enumerate(finals.index, 1):
        game = load_pbp(gid)
        box = box_by_game.get(gid)
        if game is None or box is None or len(game["starters"]) != 2:
            checks.append({"game_id": gid, "status": "missing"})
            continue
        team_of = dict(zip(box["athlete_id"], box["team_id"], strict=True))
        g = finals.loc[gid]
        try:
            stints, issues = build_stints(game, team_of)
        except (StopIteration, KeyError, IndexError, ValueError) as err:
            checks.append({"game_id": gid, "status": f"error {err}"})
            continue
        rec = reconcile(stints, box, (int(g["home_pts"]), int(g["away_pts"])))
        minutes_ok = (
            rec["mean_err"] <= 0.6 and rec["max_err"] <= 3.0
            if rec["has_minutes"]
            else not issues
        )
        ok = (
            rec["sizes_ok"]
            and rec["points_ok"]
            and rec["extra_players"] == 0
            and minutes_ok
        )
        checks.append(
            {
                "game_id": gid,
                "status": "ok" if ok else "failed",
                **rec,
                "issues": json.dumps(issues),
            }
        )
        if ok:
            for t0, t1, h, a, hp, ap in stints:
                rows.append((gid, (t1 - t0) / 60, hp - ap, *h, *a))
        if i % 2000 == 0:
            log.info("%d/%d games", i, len(finals))
    cols = [
        "game_id",
        "dur",
        "margin",
        *[f"h{k}" for k in range(5)],
        *[f"a{k}" for k in range(5)],
    ]
    WORK.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=cols).to_pickle(WORK / "stints.pkl")
    chk = pd.DataFrame(checks).merge(
        finals[["season"]], left_on="game_id", right_index=True
    )
    chk.to_csv(WORK / "checks.csv", index=False)
    summary = (
        chk.groupby("season")
        .agg(
            games=("status", "size"),
            ok=("status", lambda s: round(float((s == "ok").mean()), 3)),
            mean_err=("mean_err", lambda s: round(float(s.median()), 3)),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))
    print(
        json.dumps(
            {
                "stints": len(rows),
                "per_ok_game": round(
                    len(rows) / max(int((chk["status"] == "ok").sum()), 1), 1
                ),
            }
        )
    )
    print(chk[chk["status"] != "ok"]["status"].value_counts().head(5).to_dict())


# --------------------------------------------------------------------------
# ratings
# --------------------------------------------------------------------------

HALF_LIFE = 730.0  # days, as impact.monthly_ratings
MIN_GAMES = 500
TUNE = ("2016-10-01", "2023-07-01")  # months the ridge is chosen on
TEST = ("2023-10-01", "2026-07-01")  # 2023-24..2025-26


class Data:
    """Games, box minutes, players and stints, loaded once."""

    def __init__(self) -> None:
        from nbapredicts.collect import Store

        store = Store(ROOT / "data")
        games = store.load("games")
        self.games = games[games["status"] == "final"].copy()
        self.games["start"] = pd.to_datetime(self.games["start"], utc=True)
        pb = store.load("player_box")
        self.app = pb[(~pb["dnp"]) & (pb["min"] > 0)][
            ["game_id", "team_id", "athlete_id", "min"]
        ]
        self.players = pd.Index(self.app["athlete_id"].unique())
        self.stints = pd.read_pickle(WORK / "stints.pkl")
        checks = pd.read_csv(WORK / "checks.csv", dtype={"game_id": str})
        self.ok = set(checks.loc[checks["status"] == "ok", "game_id"])


def utc(start: pd.Series) -> np.ndarray:
    """Tip-off times as naive UTC datetime64, for sorting and searching."""
    return start.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy("datetime64[ns]")


def game_rows(d: Data, game_ids: pd.Index):
    """impact._design's rows: signed minutes/48 per player, the margin."""
    from scipy import sparse

    g = d.games.set_index("game_id").loc[game_ids]
    row = pd.Series(np.arange(len(g)), index=g.index)
    a = d.app[d.app["game_id"].isin(g.index)]
    sign = np.where(a["team_id"].to_numpy() == g.loc[a["game_id"], "home_id"], 1, -1)
    n = len(d.players)
    x = sparse.coo_matrix(
        (
            sign * a["min"].to_numpy(float) / 48.0,
            (row[a["game_id"]].to_numpy(), d.players.get_indexer(a["athlete_id"])),
        ),
        shape=(len(g), n + 1),
    ).tocsr()
    hca = np.where(g["neutral"].astype(bool), 0.0, 1.0)
    x = x + sparse.csr_matrix((hca, (np.arange(len(g)), np.full(len(g), n))), x.shape)
    y = (g["home_pts"] - g["away_pts"]).to_numpy(float)
    return x, y, utc(g["start"])


def stint_rows(d: Data, game_ids: pd.Index):
    """One row per stint, weighted by its length: a stint's margin varies
    with its length, so x = +-sqrt(dur/48) per player on the floor and
    y = margin/sqrt(dur/48), which keeps the units of game_rows (a player's
    rating is margin per 48 minutes on the floor)."""
    from scipy import sparse

    s = d.stints[d.stints["game_id"].isin(game_ids)]
    g = d.games.set_index("game_id")
    root = np.sqrt(s["dur"].to_numpy(float) / 48.0)
    n, m = len(d.players), len(s)
    cols = [f"h{k}" for k in range(5)] + [f"a{k}" for k in range(5)]
    idx = np.column_stack([d.players.get_indexer(s[c]) for c in cols])
    val = np.column_stack([root] * 5 + [-root] * 5)
    keep = idx >= 0
    rows = np.repeat(np.arange(m), 10).reshape(m, 10)
    x = sparse.coo_matrix(
        (val[keep], (rows[keep], idx[keep])), shape=(m, n + 1)
    ).tocsr()
    neutral = g.loc[s["game_id"], "neutral"].astype(bool).to_numpy()
    hca = np.where(neutral, 0.0, root)
    x = x + sparse.csr_matrix((hca, (np.arange(m), np.full(m, n))), x.shape)
    y = s["margin"].to_numpy(float) / root
    return x, y, utc(g.loc[s["game_id"], "start"])


def design(d: Data, method: str):
    """All rows, oldest first: stints for games that passed the minute
    check under ``stint``, a whole-game row otherwise and under ``game``."""
    from scipy import sparse

    ids = pd.Index(d.games["game_id"])
    if method == "game":
        parts = [game_rows(d, ids)]
    else:
        ok = ids[ids.isin(d.ok)]
        parts = [stint_rows(d, ok), game_rows(d, ids[~ids.isin(d.ok)])]
    x = sparse.vstack([p[0] for p in parts]).tocsr()
    y = np.concatenate([p[1] for p in parts])
    t = np.concatenate([p[2] for p in parts])
    order = np.argsort(t, kind="stable")
    return x[order], y[order], t[order]


def monthly(
    d: Data,
    method: str,
    lams: list[float],
    months: pd.DatetimeIndex,
    prior: dict | None = None,
):
    """Ratings as of each month for each ridge: {lam: {month: vector}}.

    Exponential decay factors: the weighted normal equations as of the next
    month are this month's times 0.5**(gap/half-life) plus the new games'.
    ``prior`` ({month: vector}) shrinks toward those ratings instead of 0.
    """
    from scipy.linalg import solve

    x, y, t = design(d, method)
    n = x.shape[1]
    s_mat, b = np.zeros((n, n)), np.zeros(n)
    out: dict[float, dict] = {lam: {} for lam in lams}
    prev, start = None, 0
    game_start = np.sort(utc(d.games["start"]))
    for month in months:
        m64 = month.tz_convert("UTC").tz_localize(None).to_datetime64()
        stop = int(np.searchsorted(t, m64))
        if prev is not None:
            gap = (month - prev).days
            s_mat *= 0.5 ** (gap / HALF_LIFE)
            b *= 0.5 ** (gap / HALF_LIFE)
        if stop > start:
            age = (m64 - t[start:stop]) / np.timedelta64(1, "D")
            w = 0.5 ** (age / HALF_LIFE)
            xs = x[start:stop]
            s_mat += (xs.T @ xs.multiply(w[:, None])).toarray()
            b += xs.T @ (w * y[start:stop])
        prev, start = month, stop
        if int(np.searchsorted(game_start, m64)) < MIN_GAMES:
            continue
        toward = prior[month] if prior is not None else 0.0
        for lam in lams:
            out[lam][month] = solve(
                s_mat + lam * np.eye(n), b + lam * toward, assume_a="pos"
            )
    return out


def month_range(d: Data) -> pd.DatetimeIndex:
    first = d.games["start"].min().normalize().replace(day=1)
    return pd.date_range(
        first + pd.offsets.MonthBegin(1),
        d.games["start"].max() + pd.offsets.MonthBegin(1),
        freq="MS",
        tz="UTC",
    )


def predict(d: Data, ratings: dict, window: tuple[str, str]) -> pd.DataFrame:
    """Each game's margin from the ratings as of its month and the minutes
    actually played (how good the ratings are, not the minutes forecast)."""
    g = d.games[
        (d.games["start"] >= pd.Timestamp(window[0], tz="UTC"))
        & (d.games["start"] < pd.Timestamp(window[1], tz="UTC"))
    ].copy()
    g["month"] = g["start"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    g["month"] = g["month"].dt.tz_localize("UTC")
    g = g[g["month"].isin(list(ratings))]
    x, y, _ = game_rows(d, pd.Index(g["game_id"]))
    pred = np.empty(len(g))
    for month, idx in g.reset_index(drop=True).groupby("month").groups.items():
        idx = np.asarray(idx)
        pred[idx] = x[idx] @ ratings[month]
    return pd.DataFrame(
        {
            "game_id": g["game_id"].to_numpy(),
            "season": g["season"].to_numpy(),
            "pred": pred,
            "margin": y,
        }
    )


def quality(p: pd.DataFrame) -> dict[str, float]:
    err = p["pred"] - p["margin"]
    return {
        "games": int(len(p)),
        "mae": round(float(err.abs().mean()), 4),
        "corr": round(float(np.corrcoef(p["pred"], p["margin"])[0, 1]), 4),
        "winners": round(float((np.sign(p["pred"]) == np.sign(p["margin"])).mean()), 4),
    }


def tune(lams: list[float]) -> None:
    d = Data()
    months = month_range(d)
    out = {}
    for method in ("game", "stint"):
        fits = monthly(d, method, lams, months)
        out[method] = {str(lam): quality(predict(d, fits[lam], TUNE)) for lam in lams}
        print(json.dumps({method: out[method]}), flush=True)
    (WORK / "tune.json").write_text(json.dumps(out, indent=1) + "\n")


# Chosen by mean absolute error on TUNE (tune.json); 5 is production's.
CONFIGS = (("game", 5.0), ("game", 15.0), ("stint", 45.0))
PRIOR_LAM = 300.0  # stint rows shrunk toward game_5, best on TUNE of 10..1000


def ratings_frame(d: Data, fits: dict) -> pd.DataFrame:
    """(month, athlete_id, rating) for players seen before each month, the
    shape ``impact.pregame_impact`` takes."""
    first = (
        d.app.merge(d.games[["game_id", "start"]], on="game_id")
        .groupby("athlete_id")["start"]
        .min()
        .reindex(d.players)
    )
    out = []
    for month, vec in fits.items():
        seen = (first < month).to_numpy()
        out.append(
            pd.DataFrame(
                {
                    "month": month,
                    "athlete_id": d.players[seen],
                    "rating": vec[:-1][seen],
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def rate() -> None:
    d = Data()
    months = month_range(d)
    preds, out = {}, {"configs": [], "seasons": {}}
    fits = {}
    for method, lam in CONFIGS:
        fits[f"{method}_{lam:g}"] = monthly(d, method, [lam], months)[lam]
    # Stint rows shrunk toward the game-level ratings, and the two averaged.
    fits["stint_prior"] = monthly(
        d, "stint", [PRIOR_LAM], months, prior=fits["game_5"]
    )[PRIOR_LAM]
    fits["average"] = {
        m: (fits["game_5"][m] + fits["stint_45"][m]) / 2 for m in fits["game_5"]
    }
    for name, f in fits.items():
        ratings_frame(d, f).to_pickle(WORK / f"ratings_{name}.pkl")
        preds[name] = predict(d, f, TEST)
        out["configs"].append(name)
        out.setdefault("tune", {})[name] = quality(predict(d, f, TUNE))
        print(json.dumps({name: quality(preds[name])}), flush=True)
    base = "game_5"
    ref = preds[base].set_index("game_id")
    for name, p in preds.items():
        p = p.set_index("game_id").loc[ref.index]
        rows = {}
        for season, part in [*p.groupby("season"), ("pooled", p)]:
            r = ref.loc[part.index]
            d_abs = (part["pred"] - part["margin"]).abs() - (
                r["pred"] - r["margin"]
            ).abs()
            boots = d_abs.to_numpy()[
                RNG.integers(0, len(d_abs), (2000, len(d_abs)))
            ].mean(axis=1)
            rows[str(season)] = {
                **quality(part.reset_index()),
                "mae_minus_production": [
                    round(float(d_abs.mean()), 4),
                    *np.round(np.percentile(boots, [2.5, 97.5]), 4).tolist(),
                ],
            }
        out["seasons"][name] = rows
        print(json.dumps({name: rows}), flush=True)
    (ROOT / "docs/results/rapm_rating_quality.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


def equine(names: list[str]) -> None:
    """The ratings as EQUINE's impact input: production's game-level ratings
    against each of ``names``, everything else as published."""
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from injury_text import equine_preds, paired
    from timesfm_margin import features

    from nbapredicts.backtest import classification_metrics as cm
    from nbapredicts.collect import FIRST_SEASON
    from nbapredicts.stack import market_logit

    preds = {}
    for name in ["game_5", *names]:
        ratings = pd.read_pickle(WORK / f"ratings_{name}.pkl")
        feats = features(ratings=ratings)
        feats = feats[feats["season"] > FIRST_SEASON].copy()
        feats["mkt"] = np.where(
            feats["p_close"].notna(),
            market_logit(feats["p_close"].fillna(0.5)),
            np.nan,
        )
        preds[name] = equine_preds(feats).set_index("game_id")
        print(json.dumps({"built": name}), flush=True)
    base = preds["game_5"]
    models = ("logistic", "equine_no_line", "equine_stack")
    out = {}
    for name in names:
        p = preds[name].loc[base.index]
        rows = []
        for season, idx in [
            *base.groupby("season").groups.items(),
            ("pooled", base.index),
        ]:
            b, c = base.loc[idx], p.loc[idx]
            keep = b[list(models)].notna().all(axis=1) & c[list(models)].notna().all(
                axis=1
            )
            b, c = b[keep], c[keep]
            y = b["home_win"].to_numpy()
            rows.append(
                {
                    "season": season if season == "pooled" else int(season),
                    "n": int(len(b)),
                    "brier": {
                        f"{m}_{v}": round(cm(y, f[m].to_numpy())["brier"], 5)
                        for m in models
                        for v, f in (("production", b), (name, c))
                    },
                    "minus_production": {
                        m: paired(y, c[m].to_numpy(), b[m].to_numpy()) for m in models
                    },
                }
            )
            print(json.dumps({name: rows[-1]}), flush=True)
        out[name] = rows
    (ROOT / "docs/results/rapm_equine.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["fetch", "stints", "tune", "rate", "equine"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--lams", type=float, nargs="+", default=[1, 3, 10, 30, 100, 300]
    )
    parser.add_argument("--ratings", nargs="+", default=["average"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.step == "fetch":
        fetch(args.smoke, args.workers)
    elif args.step == "stints":
        stints_step()
    elif args.step == "tune":
        tune(args.lams)
    elif args.step == "rate":
        rate()
    elif args.step == "equine":
        equine(args.ratings)


if __name__ == "__main__":
    main()
