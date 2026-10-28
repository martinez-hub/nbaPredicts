"""Do old seasons hurt? Training on recent seasons only (experiment).

The game features are team-level, so retired players never enter a
prediction, but the league itself changes (pace, three-point volume), and
games from 2013-14 may teach relationships that no longer hold. For each test
season this trains on the last N seasons only and compares, on the same
games, with training on everything since 2013-14:

- the feature model alone (logistic stage 1, and features-only EQUINE);
- the published stack over [closing line, stage 1].

    uv run python scripts/train_window.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from timesfm_margin import features, paired  # noqa: E402

from nbapredicts.backtest import classification_metrics as cm  # noqa: E402
from nbapredicts.collect import FIRST_SEASON  # noqa: E402
from nbapredicts.pipeline import MODEL_FEATURES  # noqa: E402
from nbapredicts.stack import (  # noqa: E402
    STACK_CONFIG,
    Stage1,
    market_logit,
    out_of_fold_stage1,
)
from nbapredicts.winner import WinnerConfig, fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (None, 8, 5, 3)  # None: every season since 2013-14


def main() -> None:
    feats = features()
    feats = feats[(feats["season"] > FIRST_SEASON) & feats["p_close"].notna()].copy()
    feats["mkt"] = market_logit(feats["p_close"])
    results = []
    for season in (2024, 2025, 2026):
        te = feats[feats["season"] == season].copy()
        y = te["home_win"].to_numpy()
        p: dict[str, np.ndarray] = {}
        for w in WINDOWS:
            tag = "all" if w is None else f"last{w}"
            lo = -np.inf if w is None else season - w
            tr = feats[(feats["season"] < season) & (feats["season"] >= lo)].copy()
            s1 = Stage1.fit(tr, MODEL_FEATURES)
            te["s1"] = s1.logit(te)
            tr["s1"] = out_of_fold_stage1(tr, MODEL_FEATURES)
            p[f"logistic_{tag}"] = 1 / (1 + np.exp(-te["s1"].to_numpy()))
            p[f"equine_{tag}"] = (
                fit(tr, WinnerConfig(), features=MODEL_FEATURES)
                .predict(te)["p_home"]
                .to_numpy()
            )
            p[f"stack_{tag}"] = (
                fit(tr, STACK_CONFIG, features=["mkt", "s1"])
                .predict(te)["p_home"]
                .to_numpy()
            )
        row = {
            "season": season,
            "n": int(len(te)),
            "brier": {k: round(cm(y, v)["brier"], 5) for k, v in p.items()},
            "vs_all": {
                k: paired(y, v, p[k.rsplit("_", 1)[0] + "_all"])
                for k, v in p.items()
                if not k.endswith("_all")
            },
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "docs/results/train_window.json").write_text(
        json.dumps(results, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
