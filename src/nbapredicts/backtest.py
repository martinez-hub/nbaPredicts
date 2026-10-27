"""Walk-forward evaluation of both models against simple baselines.

Winner model: for each test season, train on every earlier season and predict
the whole test season. That is stricter than the daily job, which retrains
each morning on everything up to the night before.

TimesFM: one-step-ahead forecasts of every team game (and a sample of player
games) in a test season, each from the games before it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .collect import FIRST_SEASON
from .features import ELO_HFA, FEATURES
from .forecast import (
    PLAYER_STATS,
    TimesFM,
    backtest_player_lines,
    backtest_team_points,
)
from .stack import StackedWinner
from .winner import WinnerConfig, fit

log = logging.getLogger(__name__)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "n": int(len(y)),
        "accuracy": float(np.mean((p > 0.5) == (y == 1))),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
    }


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict[str, float]]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        mask = idx == b
        if mask.sum() >= 10:
            out.append(
                {
                    "predicted": float(p[mask].mean()),
                    "observed": float(y[mask].mean()),
                    "n": int(mask.sum()),
                }
            )
    return out


def elo_probability(df: pd.DataFrame) -> np.ndarray:
    hfa = np.where(df["neutral"] > 0, 0.0, ELO_HFA)
    return 1 / (1 + 10 ** (-(df["elo_diff"].to_numpy() + hfa) / 400))


def backtest_winner(
    feats: pd.DataFrame,
    test_seasons: list[int],
    config: WinnerConfig | None = None,
    odds: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Walk-forward winner metrics; with ``odds``, also the closing line and the
    market-anchored EQUINE, every model scored on the same games."""
    finals = feats.dropna(subset=["home_win"])
    finals = finals[finals["season"] > FIRST_SEASON]  # first season warms up state
    if odds is not None:
        finals = finals.merge(odds[["game_id", "p_close"]], on="game_id", how="inner")
    seasons, pooled, pooled_stack = [], [], []
    for season in test_seasons:
        train = finals[finals["season"] < season]
        test = finals[finals["season"] == season]
        if test.empty:
            continue
        log.info("winner backtest %d: train %d, test %d", season, len(train), len(test))
        wm = fit(train, config)
        pred = wm.predict(test)
        y = test["home_win"].to_numpy()

        x_train = np.nan_to_num((train[FEATURES].to_numpy() - wm.mean) / wm.std)
        x_test = np.nan_to_num((test[FEATURES].to_numpy() - wm.mean) / wm.std)
        logit = LogisticRegression(C=1.0, max_iter=2000).fit(
            x_train, train["home_win"].to_numpy()
        )
        home_rate = train["home_win"].mean()

        models = {
            "equine_gp": pred["p_home"].to_numpy(),
            "logistic": logit.predict_proba(x_test)[:, 1],
            "elo": elo_probability(test),
            "home_team": np.full(len(test), home_rate),
        }
        if odds is not None:
            stacked = StackedWinner.fit(train)
            p_stack = stacked.predict(test, test["p_close"])["p_home"].to_numpy()
            models = {
                "equine_market": p_stack,
                "market": test["p_close"].to_numpy(),
                **models,
            }
            pooled_stack.append(pd.DataFrame({"y": y, "p": p_stack}))
        seasons.append(
            {
                "season": int(season),
                **{name: classification_metrics(y, p) for name, p in models.items()},
            }
        )
        pooled.append(
            pd.DataFrame({"y": y, "p": models["equine_gp"], "novelty": pred["novelty"]})
        )

    allp = pd.concat(pooled, ignore_index=True)
    # Does EQUINE's uncertainty mean anything? Accuracy by novelty tercile.
    allp["tercile"] = pd.qcut(allp["novelty"].rank(method="first"), 3, labels=False)
    by_novelty = [
        {
            "tercile": int(t),
            **classification_metrics(g["y"].to_numpy(), g["p"].to_numpy()),
        }
        for t, g in allp.groupby("tercile")
    ]
    out = {
        "seasons": seasons,
        "reliability": reliability(allp["y"].to_numpy(), allp["p"].to_numpy()),
        "by_novelty": by_novelty,
    }
    if pooled_stack:
        st = pd.concat(pooled_stack, ignore_index=True)
        out["reliability_market_model"] = reliability(
            st["y"].to_numpy(), st["p"].to_numpy()
        )
    return out


def regression_metrics(df: pd.DataFrame) -> dict[str, float]:
    err = df["pred"] - df["actual"]
    base = df["baseline"] - df["actual"]
    return {
        "n": int(len(df)),
        "mae": float(err.abs().mean()),
        "baseline_mae": float(base.abs().mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "coverage_80": float(
            ((df["actual"] >= df["lo"]) & (df["actual"] <= df["hi"])).mean()
        ),
    }


def backtest_forecasts(
    tfm: TimesFM, tf: pd.DataFrame, pf: pd.DataFrame, season: int, sample: int = 3000
) -> dict[str, Any]:
    teams = backtest_team_points(tfm, tf, season)
    players = backtest_player_lines(tfm, pf, season, sample=sample)
    # A winner pick from the two point forecasts, scored like the classifier.
    side = tf[["game_id", "team_id", "is_home"]]
    t = teams.merge(side, on=["game_id", "team_id"])
    home, away = t[t["is_home"]], t[~t["is_home"]]
    both = home.merge(away, on="game_id", suffixes=("_h", "_a"))
    pick_acc = float(
        (
            (both["pred_h"] > both["pred_a"]) == (both["actual_h"] > both["actual_a"])
        ).mean()
    )
    return {
        "season": int(season),
        "team_points": regression_metrics(teams),
        "winner_from_points_accuracy": pick_acc,
        "players": {
            stat: regression_metrics(players[players["stat"] == stat])
            for stat in PLAYER_STATS
        },
    }
