"""Does TimesFM's predicted score carry winner information? (experiment)

TimesFM forecasts each team's points from its own points series (plus
home/away, the opponent's defensive rating and rest), so the margin it
implies is a second, independent opinion about the winner. This compares it,
walk-forward and on the same games, with EQUINE and the closing line:

1. ``forecast``: one-step zero-shot team-points forecasts for every final
   game of each season, saved per season to
   ``docs/results/timesfm_margin/<season>.csv`` (seasons already there are
   skipped, so the slow step can be resumed);
2. ``eval``: winner accuracy and Brier of the TimesFM margin, of the
   published EQUINE stack and of the line; the stack with the TimesFM margin
   as a third input; who wins when TimesFM and EQUINE disagree; margin MAE
   against the closing spread.

    uv run python scripts/timesfm_margin.py forecast --seasons 2023 2024 2025 2026
    uv run python scripts/timesfm_margin.py eval
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
OUT = ROOT / "docs/results/timesfm_margin"
RNG = np.random.default_rng(0)


def forecast(seasons: list[int]) -> None:
    from nbapredicts.forecast import TimesFM, backtest_team_points, team_frame

    store = Store(ROOT / "data")
    games, tb, pb = (
        store.load("games"),
        store.load("team_box"),
        store.load("player_box"),
    )
    tf = team_frame(team_state(games, tb, player_values(pb, games), []))
    tfm = TimesFM(quantile_head=False)
    OUT.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        path = OUT / f"{season}.csv"
        if path.exists():
            continue
        rows = backtest_team_points(tfm, tf, season).assign(season=season)
        rows.to_csv(path, index=False)
        print(json.dumps({"season": season, "rows": len(rows)}), flush=True)


def paired(y, a, b, n=2000):
    """Mean Brier(a) - Brier(b) and its bootstrap 95% interval."""
    d = (a - y) ** 2 - (b - y) ** 2
    boots = d[RNG.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [
        round(float(d.mean()), 5),
        *np.round(np.percentile(boots, [2.5, 97.5]), 5).tolist(),
    ]


def features() -> pd.DataFrame:
    """Game features as the daily job builds them, with the line and TimesFM."""
    from nbapredicts.features import (
        apply_pregame_availability,
        build_features,
        pregame_availability,
    )
    from nbapredicts.impact import apply_impact, monthly_ratings, pregame_impact

    store = Store(ROOT / "data")
    games, tb, pb = (
        store.load("games"),
        store.load("team_box"),
        store.load("player_box"),
    )
    reports = store.load("injury_reports")
    teams = json.loads((ROOT / "data/teams.json").read_text())
    team_ids = {t["name"]: tid for tid, t in teams.items()}
    pv = player_values(pb, games)
    feats = build_features(games, team_state(games, tb, pv, []), [])
    feats = apply_pregame_availability(
        feats, pregame_availability(pv, games, reports, team_ids)
    )
    feats = apply_impact(
        feats,
        pregame_impact(pv, pb, games, reports, team_ids, monthly_ratings(pb, games)),
    )
    feats = feats[feats["status"] == "final"]
    feats = feats.merge(store.load("odds").drop(columns="season"), on="game_id")
    tfm = pd.concat(
        [
            pd.read_csv(p, dtype={"game_id": str, "team_id": str})
            for p in OUT.glob("*.csv")
        ]
    ).set_index(["game_id", "team_id"])["pred"]
    for side in ("home", "away"):
        keys = list(zip(feats["game_id"], feats[f"{side}_id"], strict=True))
        feats[f"tfm_{side}"] = tfm.reindex(keys).to_numpy()
    feats["m_tfm"] = feats["tfm_home"] - feats["tfm_away"]
    feats["margin"] = feats["home_pts"] - feats["away_pts"]
    return feats


def evaluate(test_seasons: list[int]) -> list[dict]:
    from sklearn.linear_model import LogisticRegression

    from nbapredicts.backtest import classification_metrics as cm
    from nbapredicts.collect import FIRST_SEASON
    from nbapredicts.pipeline import MODEL_FEATURES
    from nbapredicts.stack import STACK_CONFIG, Stage1, market_logit, out_of_fold_stage1
    from nbapredicts.winner import WinnerConfig, fit

    feats = features()
    feats = feats[feats["season"] > FIRST_SEASON]
    have_tfm = feats["m_tfm"].notna()
    print(
        json.dumps(
            {
                "tfm_seasons": sorted(feats.loc[have_tfm, "season"].unique().tolist()),
                "games_with_tfm": int(have_tfm.sum()),
            }
        ),
        flush=True,
    )
    results = []
    for season in test_seasons:
        tr = feats[(feats["season"] < season) & feats["p_close"].notna()].copy()
        te = feats[
            (feats["season"] == season)
            & feats["p_close"].notna()
            & feats["m_tfm"].notna()
        ].copy()
        y = te["home_win"].to_numpy()
        for d in (tr, te):
            d["mkt"] = market_logit(d["p_close"])
        tr["s1"] = out_of_fold_stage1(tr, MODEL_FEATURES)
        te["s1"] = Stage1.fit(tr, MODEL_FEATURES).logit(te)
        # Rows of the training seasons that have TimesFM forecasts: the
        # stacks with the TimesFM margin, and the margin's own calibration,
        # can only learn from those.
        trt = tr[tr["m_tfm"].notna()]
        p = {
            "market": te["p_close"].to_numpy(),
            "equine_stack": fit(tr, STACK_CONFIG, features=["mkt", "s1"])
            .predict(te)["p_home"]
            .to_numpy(),
            "equine_no_line": fit(tr, WinnerConfig(), features=MODEL_FEATURES)
            .predict(te)["p_home"]
            .to_numpy(),
            "timesfm": LogisticRegression(C=1e6)
            .fit(trt[["m_tfm"]], trt["home_win"])
            .predict_proba(te[["m_tfm"]])[:, 1],
        }
        weights = {}
        for name, cols in (
            ("logistic_stack", ["mkt", "s1"]),
            ("logistic_stack_tfm", ["mkt", "s1", "m_tfm"]),
            ("logistic_no_line", ["s1"]),
            ("logistic_no_line_tfm", ["s1", "m_tfm"]),
        ):
            lr = LogisticRegression(C=1e6).fit(trt[cols], trt["home_win"])
            p[name] = lr.predict_proba(te[cols])[:, 1]
            weights[name] = lr.coef_[0].round(4).tolist()
        p["equine_stack_tfm"] = (
            fit(trt, STACK_CONFIG, features=["mkt", "s1", "m_tfm"])
            .predict(te)["p_home"]
            .to_numpy()
        )
        row: dict = {
            "season": season,
            "n": int(len(te)),
            "train_with_tfm": int(len(trt)),
            "metrics": {
                k: {m: round(v, 4) for m, v in cm(y, v).items() if m != "n"}
                for k, v in p.items()
            },
            # The TimesFM pick is the side its predicted score favors.
            "timesfm_sign_accuracy": round(
                float(np.mean((te["m_tfm"] > 0) == (y == 1))), 4
            ),
            "weights": weights,
            "vs_published_stack": {
                k: paired(y, p[k], p["equine_stack"])
                for k in ("equine_stack_tfm", "timesfm", "market")
            },
            "tfm_added": {
                "stack": paired(y, p["logistic_stack_tfm"], p["logistic_stack"]),
                "no_line": paired(y, p["logistic_no_line_tfm"], p["logistic_no_line"]),
            },
        }
        # Games where TimesFM's score and the published pick disagree.
        eq_home, tfm_home = p["equine_stack"] > 0.5, te["m_tfm"].to_numpy() > 0
        split = eq_home != tfm_home
        row["disagree"] = {
            "n": int(split.sum()),
            "share": round(float(split.mean()), 4),
            "equine_right": int((eq_home[split] == (y[split] == 1)).sum()),
            "timesfm_right": int((tfm_home[split] == (y[split] == 1)).sum()),
            "mean_equine_conf": round(
                float(np.abs(p["equine_stack"][split] - 0.5).mean() + 0.5), 4
            ),
            "mean_abs_tfm_margin": round(
                float(np.abs(te["m_tfm"].to_numpy()[split]).mean()), 2
            ),
        }
        # Margin: TimesFM's against the line's (minus the home spread) and
        # against the published pick turned into points with the
        # spread-to-log-odds slope of the training seasons.
        k = np.polyfit(tr["mkt"], -tr["spread_close"].fillna(0), 1)
        m_eq = np.polyval(k, market_logit(p["equine_stack"]))
        actual = te["margin"].to_numpy()
        spread = -te["spread_close"].to_numpy()
        ok = ~np.isnan(spread)
        total = (te["tfm_home"] + te["tfm_away"]).to_numpy()
        home_rec, away_rec = (total + m_eq) / 2, (total - m_eq) / 2
        pts = np.concatenate([te["home_pts"], te["away_pts"]])
        row["margin_mae"] = {
            "timesfm": round(float(np.abs(te["m_tfm"] - actual).mean()), 3),
            "equine_stack_points": round(float(np.abs(m_eq - actual).mean()), 3),
            "closing_spread": round(float(np.abs(spread[ok] - actual[ok]).mean()), 3),
            "timesfm_on_spread_games": round(
                float(np.abs(te["m_tfm"].to_numpy()[ok] - actual[ok]).mean()), 3
            ),
        }
        row["team_points_mae"] = {
            "timesfm": round(
                float(
                    np.abs(
                        np.concatenate([te["tfm_home"], te["tfm_away"]]) - pts
                    ).mean()
                ),
                3,
            ),
            "timesfm_total_equine_margin": round(
                float(np.abs(np.concatenate([home_rec, away_rec]) - pts).mean()), 3
            ),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "docs/results/timesfm_margin.json").write_text(
        json.dumps(results, indent=1) + "\n"
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["forecast", "eval"])
    parser.add_argument(
        "--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026]
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.step == "forecast":
        forecast(args.seasons)
    else:
        evaluate(args.seasons)


if __name__ == "__main__":
    main()
