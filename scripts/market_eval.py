"""Walk-forward comparison with the betting market, on the same games.

For each test season, models are trained on earlier seasons only and scored
by Brier score against the line they would be judged against, with a paired
bootstrap 95% interval of the difference (negative = better than the line):

- close:  stacked EQUINE over [closing line, feature model]  vs the closing line
- open:   stacked EQUINE over [opening line, feature model]  vs the opening line
          (trained on the seasons that have opening lines: 2013-14..2016-17
          and 2022-23 on)

Optionally ``--avail pregame`` swaps the hindsight availability feature for
the one built from the official injury report published before the games.

    uv run python scripts/market_eval.py [--seasons 2024 2025 2026]
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
RNG = np.random.default_rng(0)


def paired(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 2000) -> list[float]:
    """Mean Brier(a) - Brier(b) and its bootstrap 95% interval."""
    d = (a - y) ** 2 - (b - y) ** 2
    boots = d[RNG.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [float(d.mean()), *np.percentile(boots, [2.5, 97.5]).tolist()]


def stacked(
    train: pd.DataFrame, test: pd.DataFrame, line: str
) -> dict[str, np.ndarray]:
    """EQUINE and logistic stacks over [line log-odds, feature-model log-odds]."""
    train, test = train.copy(), test.copy()
    train["mkt"], test["mkt"] = market_logit(train[line]), market_logit(test[line])
    train["s1"] = out_of_fold_stage1(train)
    test["s1"] = Stage1.fit(train).logit(test)
    head = fit(train, STACK_CONFIG, features=["mkt", "s1"])
    lr = LogisticRegression(C=1e6).fit(train[["mkt", "s1"]], train["home_win"])
    return {
        "equine": head.predict(test)["p_home"].to_numpy(),
        "logistic": lr.predict_proba(test[["mkt", "s1"]])[:, 1],
        "weights": lr.coef_[0].round(3).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025, 2026])
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--avail",
        choices=["hindsight", "pregame"],
        default="hindsight",
        help="availability from the box score, or from the official injury report",
    )
    args = parser.parse_args()

    store = Store(ROOT / "data")
    games, tb, pb = (
        store.load("games"),
        store.load("team_box"),
        store.load("player_box"),
    )
    odds = store.load("odds")
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
    feats = feats.merge(odds.drop(columns="season"), on="game_id", how="inner")

    results = []
    for season in args.seasons:
        past, test = feats[feats["season"] < season], feats[feats["season"] == season]
        y = test["home_win"].to_numpy()
        row: dict = {"season": season, "n": int(len(test))}
        row["market_close"] = cm(y, test["p_close"].to_numpy())["brier"]

        close = stacked(past, test, "p_close")
        row["equine_vs_close"] = paired(y, close["equine"], test["p_close"].to_numpy())
        row["logistic_vs_close"] = paired(
            y, close["logistic"], test["p_close"].to_numpy()
        )
        row["close_weights"] = close["weights"]

        models = test["p_models"].notna()
        if models.sum() > 200:
            ym = y[models.to_numpy()]
            row["model_sites_vs_close"] = paired(
                ym,
                test.loc[models, "p_models"].to_numpy(),
                test.loc[models, "p_close"].to_numpy(),
            )

        has_open = test["p_open"].notna()
        train_open = past[past["p_open"].notna()]
        if has_open.sum() > 200 and len(train_open) > 1000:
            to = test[has_open]
            yo = to["home_win"].to_numpy()
            op = stacked(train_open, to, "p_open")
            row["n_open"] = int(len(to))
            row["market_open"] = cm(yo, to["p_open"].to_numpy())["brier"]
            row["close_vs_open"] = paired(
                yo, to["p_close"].to_numpy(), to["p_open"].to_numpy()
            )
            row["equine_vs_open"] = paired(yo, op["equine"], to["p_open"].to_numpy())
            row["logistic_vs_open"] = paired(
                yo, op["logistic"], to["p_open"].to_numpy()
            )
            row["open_weights"] = op["weights"]
        results.append(row)
        print(json.dumps(row, default=lambda v: round(v, 5)), flush=True)

    out = args.out or ROOT / f"docs/results/market_eval_{args.avail}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
