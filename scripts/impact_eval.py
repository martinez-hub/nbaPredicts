"""Does minutes-weighted player impact add information? (experiment)

Adds impact_diff (home minus away expected strength from plus-minus ratings,
projected minutes and the injury report) and each team's change from its
recent average, then compares, walk-forward on 2023-24..2025-26:

1. feature-only models with and without the impact features;
2. stacked models over the opening and closing lines with them.

    uv run python scripts/impact_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from nbapredicts.backtest import classification_metrics as cm
from nbapredicts.collect import FIRST_SEASON, Store
from nbapredicts.features import (
    FEATURES,
    apply_pregame_availability,
    build_features,
    player_values,
    pregame_availability,
    team_state,
)
from nbapredicts.impact import monthly_ratings, pregame_impact
from nbapredicts.stack import STACK_CONFIG, Stage1, market_logit, out_of_fold_stage1
from nbapredicts.winner import WinnerConfig, fit

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(0)
IMPACT = ["impact_diff", "impact_delta_home", "impact_delta_away"]


def paired(y, a, b, n=2000):
    d = (a - y) ** 2 - (b - y) ** 2
    boots = d[RNG.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [
        round(float(d.mean()), 5),
        *np.round(np.percentile(boots, [2.5, 97.5]), 5).tolist(),
    ]


def add_impact(feats: pd.DataFrame, imp: pd.DataFrame) -> pd.DataFrame:
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
    out["impact_diff"] = out["_home"] - out["_away"]
    return out.drop(columns=["_home", "_away"])


def main() -> None:
    store = Store(ROOT / "data")
    games, tb, pb = (
        store.load("games"),
        store.load("team_box"),
        store.load("player_box"),
    )
    reports = store.load("injury_reports")
    teams = json.loads((ROOT / "data" / "teams.json").read_text())
    team_ids = {t["name"]: tid for tid, t in teams.items()}
    pv = player_values(pb, games)
    feats = build_features(games, team_state(games, tb, pv, []), [])
    feats = apply_pregame_availability(
        feats, pregame_availability(pv, games, reports, team_ids)
    )
    ratings = monthly_ratings(pb, games)
    feats = add_impact(feats, pregame_impact(pv, pb, games, reports, team_ids, ratings))
    feats = feats[(feats["status"] == "final") & (feats["season"] > FIRST_SEASON)]
    feats = feats.merge(store.load("odds").drop(columns="season"), on="game_id")
    feats = feats[feats["impact_diff"].notna()]
    with_imp = [*FEATURES, *IMPACT]

    results = []
    for season in (2024, 2025, 2026):
        tr, te = (
            feats[feats["season"] < season].copy(),
            feats[feats["season"] == season].copy(),
        )
        y = te["home_win"].to_numpy()
        row: dict = {"season": season, "n": int(len(te))}
        for name, cols in (("base", FEATURES), ("impact", with_imp)):
            s1 = Stage1.fit(tr, cols)
            p = 1 / (1 + np.exp(-s1.logit(te)))
            row[f"logistic_{name}"] = round(cm(y, p)["brier"], 5)
            eq = fit(tr, WinnerConfig(), features=cols).predict(te)["p_home"].to_numpy()
            row[f"equine_{name}"] = round(cm(y, eq)["brier"], 5)
        for line in ("p_close", "p_open"):
            t2, e2 = tr[tr[line].notna()].copy(), te[te[line].notna()].copy()
            if len(e2) < 200 or len(t2) < 1000:
                continue
            for d in (t2, e2):
                d["mkt"] = market_logit(d[line])
            t2["s1"] = out_of_fold_stage1(t2, with_imp)
            e2["s1"] = Stage1.fit(t2, with_imp).logit(e2)
            ye, pl = e2["home_win"].to_numpy(), e2[line].to_numpy()
            lr = LogisticRegression(C=1e6).fit(t2[["mkt", "s1"]], t2["home_win"])
            p_lr = lr.predict_proba(e2[["mkt", "s1"]])[:, 1]
            p_eq = (
                fit(t2, STACK_CONFIG, features=["mkt", "s1"])
                .predict(e2)["p_home"]
                .to_numpy()
            )
            tag = line.removeprefix("p_")
            row[f"logistic_vs_{tag}"] = paired(ye, p_lr, pl)
            row[f"equine_vs_{tag}"] = paired(ye, p_eq, pl)
            row[f"weights_{tag}"] = lr.coef_[0].round(3).tolist()
        results.append(row)
        print(json.dumps(row), flush=True)
    out = ROOT / "docs/results/impact_eval.json"
    out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()
