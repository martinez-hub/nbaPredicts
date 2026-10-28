"""Sweep EQUINE stack settings (normalization off) against the opening line."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from nbapredicts.backtest import classification_metrics as cm
from nbapredicts.collect import FIRST_SEASON, Store
from nbapredicts.features import build_features, player_values, team_state
from nbapredicts.stack import STACK_CONFIG, Stage1, market_logit, out_of_fold_stage1
from nbapredicts.winner import fit

ROOT = Path(__file__).resolve().parents[1]
store = Store(ROOT / "data")
games, tb, pb = store.load("games"), store.load("team_box"), store.load("player_box")
feats = build_features(games, team_state(games, tb, player_values(pb, games), []), [])
feats = feats[(feats["status"] == "final") & (feats["season"] > FIRST_SEASON)]
feats = feats.merge(store.load("odds").drop(columns="season"), on="game_id")
feats = feats[feats["p_open"].notna()]
base = replace(STACK_CONFIG, normalize_gp_features=False)
configs = {
    "base": base,
    "mff_pi8": replace(base, mean_field_factor=float(np.pi / 8)),
    "mff_1": replace(base, mean_field_factor=1.0),
    "wider": replace(base, hidden=(16,), emb_dim=8, random_features=256),
    "wider_pi8": replace(
        base,
        hidden=(16,),
        emb_dim=8,
        random_features=256,
        mean_field_factor=float(np.pi / 8),
    ),
    "rff512_pi8": replace(
        base, random_features=512, mean_field_factor=float(np.pi / 8)
    ),
}
scores = {k: [] for k in configs}
for season in (2025, 2026):
    tr, te = (
        feats[feats["season"] < season].copy(),
        feats[feats["season"] == season].copy(),
    )
    for d in (tr, te):
        d["mkt"] = market_logit(d["p_open"])
    tr["s1"] = out_of_fold_stage1(tr)
    te["s1"] = Stage1.fit(tr).logit(te)
    y, p_open = te["home_win"].to_numpy(), te["p_open"].to_numpy()
    o = cm(y, p_open)["brier"]
    for k, c in configs.items():
        b = cm(y, fit(tr, c, features=["mkt", "s1"]).predict(te)["p_home"].to_numpy())[
            "brier"
        ]
        scores[k].append(b - o)
    print(season, {k: round(v[-1], 5) for k, v in scores.items()}, flush=True)
print(
    "MEAN vs open:",
    json.dumps({k: round(float(np.mean(v)), 5) for k, v in scores.items()}),
)
