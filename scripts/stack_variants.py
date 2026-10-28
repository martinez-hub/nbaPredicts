"""Which EQUINE stacking head uses the opening line best? (experiment)

A: EQUINE on [line log-odds, feature-model log-odds]
B: EQUINE on the out-of-fold log-odds of a logistic stack of those two
L: the logistic stack itself (reference)
All scored against the opening line, walk-forward, same games.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nbapredicts.backtest import classification_metrics as cm
from nbapredicts.collect import FIRST_SEASON, Store
from nbapredicts.features import (
    apply_pregame_availability,
    build_features,
    player_values,
    pregame_availability,
    team_state,
)
from nbapredicts.stack import STACK_CONFIG, Stage1, market_logit, out_of_fold_stage1
from nbapredicts.winner import fit

ROOT = Path(__file__).resolve().parents[1]


def oof_logistic(train: pd.DataFrame, cols: list[str]) -> np.ndarray:
    out = np.full(len(train), np.nan)
    seasons = train["season"].to_numpy()
    for s in np.unique(seasons):
        held = seasons == s
        lr = LogisticRegression(C=1e6).fit(
            train.loc[~held, cols], train.loc[~held, "home_win"]
        )
        out[held] = lr.decision_function(train.loc[held, cols])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--avail", choices=["hindsight", "pregame"], default="hindsight"
    )
    parser.add_argument("--train-from", type=int, default=FIRST_SEASON + 1)
    args = parser.parse_args()

    store = Store(ROOT / "data")
    games, tb, pb = (
        store.load("games"),
        store.load("team_box"),
        store.load("player_box"),
    )
    feats = build_features(
        games, team_state(games, tb, player_values(pb, games), []), []
    )
    if args.avail == "pregame":
        teams = json.loads((ROOT / "data" / "teams.json").read_text())
        pre = pregame_availability(
            player_values(pb, games),
            games,
            store.load("injury_reports"),
            {t["name"]: tid for tid, t in teams.items()},
        )
        feats = apply_pregame_availability(feats, pre)
    feats = feats[(feats["status"] == "final") & (feats["season"] > FIRST_SEASON)]
    feats = feats.merge(store.load("odds").drop(columns="season"), on="game_id")
    feats = feats[feats["p_open"].notna() & (feats["season"] >= args.train_from)]

    for season in (2025, 2026):
        tr, te = (
            feats[feats["season"] < season].copy(),
            feats[feats["season"] == season].copy(),
        )
        for d in (tr, te):
            d["mkt"] = market_logit(d["p_open"])
        tr["s1"] = out_of_fold_stage1(tr)
        te["s1"] = Stage1.fit(tr).logit(te)
        lr = LogisticRegression(C=1e6).fit(tr[["mkt", "s1"]], tr["home_win"])
        tr["z"] = oof_logistic(tr, ["mkt", "s1"])
        te["z"] = lr.decision_function(te[["mkt", "s1"]])
        y, p_open = te["home_win"].to_numpy(), te["p_open"].to_numpy()
        res = {"season": season, "n_train": len(tr), "open": cm(y, p_open)["brier"]}
        res["L"] = cm(y, lr.predict_proba(te[["mkt", "s1"]])[:, 1])["brier"]
        res["A"] = cm(
            y,
            fit(tr, STACK_CONFIG, features=["mkt", "s1"])
            .predict(te)["p_home"]
            .to_numpy(),
        )["brier"]
        from dataclasses import replace

        nonorm = replace(STACK_CONFIG, normalize_gp_features=False)
        for name, cols in (("A_nonorm", ["mkt", "s1"]), ("B_nonorm", ["z"])):
            res[name] = cm(
                y, fit(tr, nonorm, features=cols).predict(te)["p_home"].to_numpy()
            )["brier"]
        res["B"] = cm(
            y, fit(tr, STACK_CONFIG, features=["z"]).predict(te)["p_home"].to_numpy()
        )["brier"]
        print(
            json.dumps(
                {k: round(v, 5) if isinstance(v, float) else v for k, v in res.items()}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
