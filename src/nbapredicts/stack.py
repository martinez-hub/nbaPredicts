"""EQUINE over the betting line: a two-stage, market-anchored winner model.

Stage 1 is an L2 logistic regression on the game features; for training
games its log-odds come out of fold (leave one season out), so stage 2 never
sees a stage-1 value fitted on the game itself. Stage 2 is EQUINE's GP head on
two inputs: the market's log-odds and stage 1's.

Why two stages: given the 25 features and the line together, EQUINE's
embedding and distance-based GP drowned the line (walk-forward Brier 0.212 vs
0.199 for the closing line in 2023-24). Stacked, it lands within about 0.0006
of the closing line on 2023-24..2025-26; given the line alone it matches it.
Everything the features know is already in the closing line (their stacked
weight is about -0.06), so the line is what makes this model good, and the
place left to gain is information the line does not have yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .features import FEATURES
from .winner import WinnerConfig, WinnerModel, fit

STACK_INPUTS = ["mkt", "s1"]
# Swept against the opening line on 2024-25/2025-26 (scripts/stack_sweep.py):
# without the GP-feature LayerNorm and with SNGP's mean-field factor (pi/8)
# this head beats the open by 0.00065 Brier on average; with EquineGP's
# defaults it did not beat it at all; wider heads lost the edge.
STACK_CONFIG = WinnerConfig(
    hidden=(8,),
    emb_dim=4,
    random_features=128,
    lr=1e-3,
    max_epochs=80,
    patience=12,
    weight_decay=1e-3,
    mean_field_factor=0.39269908169872414,  # pi / 8
    normalize_gp_features=False,
)


def market_logit(p: pd.Series | np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


@dataclass
class Stage1:
    model: LogisticRegression
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, train: pd.DataFrame) -> Stage1:
        x = train[FEATURES].to_numpy(dtype=float)
        mean, std = np.nanmean(x, axis=0), np.nanstd(x, axis=0) + 1e-9
        z = np.nan_to_num((x - mean) / std)
        model = LogisticRegression(C=1.0, max_iter=3000).fit(z, train["home_win"])
        return cls(model, mean, std)

    def logit(self, df: pd.DataFrame) -> np.ndarray:
        x = df[FEATURES].to_numpy(dtype=float)
        return self.model.decision_function(np.nan_to_num((x - self.mean) / self.std))


def out_of_fold_stage1(train: pd.DataFrame) -> np.ndarray:
    """Stage-1 log-odds for each training game from a model fit without it.

    Folds are seasons (leave one season out); with fewer than three seasons,
    five chronological blocks.
    """
    if train["season"].nunique() >= 3:
        folds = train["season"].to_numpy()
    else:
        order = np.argsort(train["start"].to_numpy(), kind="stable")
        folds = np.empty(len(train), dtype=int)
        folds[order] = np.arange(len(train)) * 5 // max(len(train), 1)
    out = np.full(len(train), np.nan)
    for fold in np.unique(folds):
        held = folds == fold
        out[held] = Stage1.fit(train[~held]).logit(train[held])
    return out


@dataclass
class StackedWinner:
    stage1: Stage1
    head: WinnerModel

    @classmethod
    def fit(
        cls, train: pd.DataFrame, config: WinnerConfig = STACK_CONFIG
    ) -> StackedWinner:
        """``train`` needs the features, ``home_win``, ``season`` and ``p_close``."""
        train = train.dropna(subset=["p_close", "home_win"]).copy()
        train["mkt"] = market_logit(train["p_close"])
        train["s1"] = out_of_fold_stage1(train)
        head = fit(train, config, features=STACK_INPUTS)
        return cls(Stage1.fit(train), head)

    def predict(self, df: pd.DataFrame, p_market: pd.Series) -> pd.DataFrame:
        x = df.copy()
        x["mkt"] = market_logit(p_market)
        x["s1"] = self.stage1.logit(x)
        return self.head.predict(x)
