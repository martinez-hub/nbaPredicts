"""NVIDIA Kumo-Forecast against TimesFM on the same forecasts (experiment).

Kumo-Forecast (nvidia/Kumo-Forecast, Apache-2.0) is a MOMENT-1-large encoder
with a forecasting head trained on the Monash archive; it reads 512 points of
history and takes no future-known covariates. On 2025-26, one step ahead,
every team's points and the published player-line sample (3,000 appearances,
seed 0, four stats), it is scored against:

- TimesFM 2.5 with XReg on home/away, opponent defense and rest (the page);
- TimesFM 2.5 alone (no covariates, the like-for-like comparison);
- the exponentially weighted average baseline.

Three steps, the middle one in Kumo-TS's own environment (it has its own
lockfile and a top-level ``sdk`` module):

    uv run python scripts/kumo_compare.py export
    (cd /Volumes/ExtremeSSD/Kumo-TS/Kumo-Forecast && \\
        .venv/bin/python /Volumes/ExtremeSSD/nbaPredicts/scripts/kumo_infer.py \\
        /Volumes/ExtremeSSD/nbaPredicts/cache.nosync/kumo)
    uv run python scripts/kumo_compare.py score
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from nbapredicts.collect import Store
from nbapredicts.features import player_values, team_state

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "cache.nosync/kumo"
SEASON = 2026
SEQ_LEN = 512  # Kumo-Forecast's head is trained on 512-point windows
RNG = np.random.default_rng(0)


def window(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The last SEQ_LEN values, left-padded with zeros, and the 0/1 mask."""
    v = values[-SEQ_LEN:].astype(np.float32)
    ctx = np.zeros(SEQ_LEN, np.float32)
    mask = np.zeros(SEQ_LEN, np.int8)
    ctx[SEQ_LEN - len(v) :], mask[SEQ_LEN - len(v) :] = v, 1
    return ctx, mask


def export() -> None:
    from nbapredicts.forecast import (
        MIN_GAMES,
        PLAYER_STATS,
        TimesFM,
        backtest_player_lines,
        backtest_team_points,
        player_frame,
        team_frame,
    )

    store = Store(ROOT / "data")
    games, tb, pb = (
        store.load("games"),
        store.load("team_box"),
        store.load("player_box"),
    )
    pv = player_values(pb, games)
    tf = team_frame(team_state(games, tb, pv, []))
    pf = player_frame(pv, tf)
    xreg = TimesFM(quantile_head=False)
    plain = TimesFM(quantile_head=False, xreg_mode=None)

    # Team points: the rows and windows of forecast.backtest_team_points.
    team = backtest_team_points(xreg, tf, SEASON).rename(columns={"pred": "tfm_xreg"})
    team["tfm_plain"] = backtest_team_points(plain, tf, SEASON)["pred"].to_numpy()
    ctx, mask = [], []
    for _team, rows in tf[tf["final"]].groupby("team_id", sort=False):
        rows = rows.reset_index(drop=True)
        for i in np.flatnonzero(rows["season"].to_numpy() == SEASON):
            if i < MIN_GAMES:
                continue
            c, m = window(rows["pts"].to_numpy()[max(0, i - SEQ_LEN) : i])
            ctx.append(c)
            mask.append(m)
    team = team.drop(columns=["lo", "hi"])

    # Player lines: forecast.backtest_player_lines' sample, in its order.
    players = backtest_player_lines(xreg, pf, SEASON).rename(
        columns={"pred": "tfm_xreg"}
    )
    players["tfm_plain"] = backtest_player_lines(plain, pf, SEASON)["pred"].to_numpy()
    players = players.drop(columns=["lo", "hi"])
    pf = pf.reset_index(drop=True)
    pf["n"] = pf.groupby("athlete_id").cumcount()
    candidates = pf[(pf["season"] == SEASON) & (pf["n"] >= MIN_GAMES)]
    picks = candidates.sample(min(3000, len(candidates)), random_state=0)
    history = dict(tuple(pf.groupby("athlete_id", sort=False)))
    pctx, pmask = [], []
    for p in picks.itertuples():
        hist = history[p.athlete_id]
        hist = hist[hist["n"] < p.n]
        for stat in PLAYER_STATS:
            c, m = window(hist.dropna(subset=[stat])[stat].to_numpy())
            pctx.append(c)
            pmask.append(m)

    WORK.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        WORK / "windows.npz",
        team_ctx=np.stack(ctx),
        team_mask=np.stack(mask),
        player_ctx=np.stack(pctx),
        player_mask=np.stack(pmask),
    )
    team.to_csv(WORK / "team_rows.csv", index=False)
    players.to_csv(WORK / "player_rows.csv", index=False)
    print(json.dumps({"team_rows": len(team), "player_rows": len(players)}))


def paired_abs(y, a, b, n=2000):
    """Mean |a - y| - |b - y| and its bootstrap 95% interval."""
    d = np.abs(a - y) - np.abs(b - y)
    boots = d[RNG.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [
        round(float(d.mean()), 4),
        *np.round(np.percentile(boots, [2.5, 97.5]), 4).tolist(),
    ]


def summarize(rows: pd.DataFrame, kumo_cols: list[str]) -> dict:
    y = rows["actual"].to_numpy()
    out = {"n": int(len(rows))}
    for col in ["tfm_xreg", "tfm_plain", "baseline", *kumo_cols]:
        out[f"mae_{col}"] = round(float(np.abs(rows[col] - y).mean()), 4)
    for col in kumo_cols:
        for ref in ("tfm_xreg", "tfm_plain"):
            out[f"{col}_minus_{ref}"] = paired_abs(
                y, rows[col].to_numpy(), rows[ref].to_numpy()
            )
    return out


def score() -> None:
    from nbapredicts.forecast import PLAYER_STATS

    preds = np.load(WORK / "kumo_preds.npz")
    meta = json.loads((WORK / "kumo_meta.json").read_text())
    variants = [k.removeprefix("team_") for k in preds.files if k.startswith("team_")]
    kumo_cols = [f"kumo_{v}" for v in variants]
    team = pd.read_csv(WORK / "team_rows.csv", dtype={"game_id": str, "team_id": str})
    players = pd.read_csv(WORK / "player_rows.csv", dtype={"game_id": str})
    for v in variants:
        team[f"kumo_{v}"] = preds[f"team_{v}"]
        players[f"kumo_{v}"] = preds[f"player_{v}"]
    windows = np.load(WORK / "windows.npz")
    players["context"] = windows["player_mask"].sum(axis=1)

    result: dict = {
        "season": SEASON,
        "kumo": meta,
        "team_points": summarize(team, kumo_cols),
    }

    # Game totals and the winner each model's two forecasts imply.
    games = Store(ROOT / "data").load("games").set_index("game_id")
    team["is_home"] = (
        team["team_id"].to_numpy()
        == games["home_id"].reindex(team["game_id"]).to_numpy()
    )
    cols = ["actual", "tfm_xreg", "tfm_plain", "baseline", *kumo_cols]
    home = team[team["is_home"]].set_index("game_id")[cols]
    away = team[~team["is_home"]].set_index("game_id")[cols]
    both = home.index.intersection(away.index)
    home, away = home.loc[both], away.loc[both]
    total = home + away
    result["game_total"] = summarize(total, kumo_cols)
    won = (home["actual"] > away["actual"]).to_numpy()
    result["winner_accuracy"] = {
        c: round(float(np.mean((home[c] > away[c]).to_numpy() == won)), 4)
        for c in cols[1:]
    }

    result["players"] = {}
    for stat in PLAYER_STATS:
        rows = players[players["stat"] == stat]
        result["players"][stat] = summarize(rows, kumo_cols)
        # Kumo's head was trained on full 512-point windows; most player
        # histories are shorter and padded.
        full = rows[rows["context"] >= 256]
        result["players"][stat]["n_context_256_plus"] = int(len(full))
        if len(full) > 50:
            result["players"][stat]["context_256_plus"] = summarize(full, kumo_cols)
    result["players_context_median"] = int(players["context"].median())

    out = ROOT / "docs/results/kumo_vs_timesfm.json"
    out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["export", "score"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    export() if args.step == "export" else score()


if __name__ == "__main__":
    main()
