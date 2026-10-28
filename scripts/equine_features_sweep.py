"""Features-only EQUINE settings with pre-game availability and impact (experiment)."""

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

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
from nbapredicts.stack import Stage1
from nbapredicts.winner import WinnerConfig, fit

sys.path.insert(0, str(Path(__file__).resolve().parent))
from impact_eval import IMPACT, add_impact  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
store = Store(ROOT / "data")
games, tb, pb = store.load("games"), store.load("team_box"), store.load("player_box")
reports = store.load("injury_reports")
teams = json.loads((ROOT / "data" / "teams.json").read_text())
team_ids = {t["name"]: tid for tid, t in teams.items()}
pv = player_values(pb, games)
feats = build_features(games, team_state(games, tb, pv, []), [])
feats = apply_pregame_availability(
    feats, pregame_availability(pv, games, reports, team_ids)
)
feats = add_impact(
    feats, pregame_impact(pv, pb, games, reports, team_ids, monthly_ratings(pb, games))
)
feats = feats[
    (feats["status"] == "final")
    & (feats["season"] > FIRST_SEASON)
    & feats["impact_diff"].notna()
]
cols = [*FEATURES, *IMPACT]
base = WinnerConfig()
configs = {
    "default": base,
    "nonorm_pi8": replace(
        base, normalize_gp_features=False, mean_field_factor=float(np.pi / 8)
    ),
    "nonorm_pi8_wide": replace(
        base,
        normalize_gp_features=False,
        mean_field_factor=float(np.pi / 8),
        hidden=(64,),
        emb_dim=16,
    ),
    "nonorm_pi8_rff1024": replace(
        base,
        normalize_gp_features=False,
        mean_field_factor=float(np.pi / 8),
        random_features=1024,
    ),
}
res = {k: [] for k in [*configs, "logistic"]}
for season in (2024, 2025, 2026):
    tr, te = feats[feats["season"] < season], feats[feats["season"] == season]
    y = te["home_win"].to_numpy()
    res["logistic"].append(
        cm(y, 1 / (1 + np.exp(-Stage1.fit(tr, cols).logit(te))))["brier"]
    )
    for k, c in configs.items():
        res[k].append(
            cm(y, fit(tr, c, features=cols).predict(te)["p_home"].to_numpy())["brier"]
        )
    print(season, {k: round(v[-1], 5) for k, v in res.items()}, flush=True)
print("MEAN", json.dumps({k: round(float(np.mean(v)), 5) for k, v in res.items()}))
