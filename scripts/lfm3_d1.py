"""Liquid AI's d1-2B decision model (LFM3 family) against EQUINE (experiment).

d1-2B reads a state (text or JSON) and a list of questions and answers them
in one forward pass, each yes/no with a probability Liquid AI calibrated on
its own data. Zero-shot here: no training on NBA games. Each game's state is
the same 28 pre-game features EQUINE reads, under descriptive names, and the
one question is whether the home team wins. Walk-forward is moot (nothing is
fit), but the comparison uses the same games and baselines as
``kumo_tabular.py``:

- without the line: against features-only EQUINE and the logistic stage 1;
- with the line: the features plus the closing line's home win probability,
  against the published EQUINE stack and the line itself.

LFM3 was pretrained on text through 2026, which can include these games'
results, so the state carries no team names, dates or game ids. The
``named`` variant adds them back for one season, only to see whether the
model recalls outcomes; it is never a result.

``score`` also refits each season's probabilities on the season before
(Platt: slope and intercept on the log-odds), in case the output is
informative but miscalibrated for this task.

Needs ``kumo_tabular.py export``'s files. ``infer`` runs in an environment
with transformers>=5.19 (it does not import this package):

    ~/.venvs/lfm3/bin/python scripts/lfm3_d1.py infer [--smoke]
    uv run python scripts/lfm3_d1.py score
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "cache.nosync/kumo_tab"  # kumo_tabular.py export
WORK = ROOT / "cache.nosync/lfm3"
REPO = "LiquidAI/d1-2B"
REVISION = "10e7671d44e4b37700020b1eed3f594c5f2af0b4"  # 2026-10-29
SEASONS = (2024, 2025, 2026)
CALIBRATION_SEASON = 2023  # predicted only to recalibrate 2023-24
BATCH = 32
RNG = np.random.default_rng(0)

# The features under names the model can read. "diff" is home minus away;
# ratings are per 100 possessions, averaged with more weight on recent games.
NAMES = {
    "elo_diff": "elo_rating_home_minus_away",
    "neutral": "neutral_site",
    "playoff": "playoff_game",
    "net_rtg_s_diff": "net_rating_last_8_games_home_minus_away",
    "net_rtg_l_diff": "net_rating_last_30_games_home_minus_away",
    "off_rtg_l_diff": "offensive_rating_home_minus_away",
    "def_rtg_l_diff": "defensive_rating_points_allowed_home_minus_away",
    "efg_l_diff": "effective_fg_pct_home_minus_away",
    "opp_efg_l_diff": "opponent_effective_fg_pct_allowed_home_minus_away",
    "tov_pct_l_diff": "turnover_pct_home_minus_away",
    "opp_tov_pct_l_diff": "opponent_turnover_pct_forced_home_minus_away",
    "orb_pct_l_diff": "offensive_rebound_pct_home_minus_away",
    "ftr_l_diff": "free_throw_rate_home_minus_away",
    "fg3a_rate_l_diff": "three_point_attempt_rate_home_minus_away",
    "pace_l_mean": "pace_average_of_both_teams",
    "win10_diff": "win_pct_last_10_games_home_minus_away",
    "season_win_diff": "season_win_pct_home_minus_away",
    "rest_home": "home_team_days_of_rest",
    "rest_away": "away_team_days_of_rest",
    "b2b_home": "home_team_on_back_to_back",
    "b2b_away": "away_team_on_back_to_back",
    "avail_diff": "available_players_strength_home_minus_away",
    "avail_delta_home": "home_available_strength_vs_its_usual",
    "avail_delta_away": "away_available_strength_vs_its_usual",
    "season_games_min": "games_played_this_season_fewer_of_the_two",
    "impact_diff": "expected_lineup_plus_minus_home_minus_away",
    "impact_delta_home": "home_expected_lineup_plus_minus_vs_its_usual",
    "impact_delta_away": "away_expected_lineup_plus_minus_vs_its_usual",
}
FLAGS = {"neutral", "playoff", "b2b_home", "b2b_away"}
QUESTION = {
    "type": "noul",
    "instructions": (
        "These are pre-game statistics for an NBA game. Fields ending in"
        " home_minus_away are the home team's value minus the away team's."
        " Lower is better for defensive rating, opponent shooting and turnover"
        " percentage. Will the home team win this game?"
    ),
    "criteria": {"true": "The home team wins", "false": "The away team wins"},
}


def state(row: pd.Series, feats: list[str], line: bool, named: dict | None) -> dict:
    out = {}
    if named is not None:
        out["date"] = str(row["start"])[:10]
        out["home_team"] = named[str(row["home_id"])]
        out["away_team"] = named[str(row["away_id"])]
    for c in feats:
        v = row[c]
        out[NAMES[c]] = bool(v) if c in FLAGS else round(float(v), 3)
    if line:
        out["betting_market_home_win_probability"] = round(float(row["p_close"]), 3)
    return out


# --------------------------------------------------------------------------
# infer (an environment with transformers>=5.19)
# --------------------------------------------------------------------------


def infer(
    smoke: bool, seasons: tuple[int, ...], settings: tuple[str, ...], dtype_name: str
) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # The M1 has no bfloat16 in hardware; float16 runs natively.
    dtype = getattr(torch, dtype_name) if device != "cpu" else torch.float32
    path = snapshot_download(REPO, revision=REVISION)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype)
    model = model.to(device)
    feats = json.loads((TAB / "features.json").read_text())
    games = pd.read_csv(TAB / "games.csv", dtype={"game_id": str})
    teams = json.loads((ROOT / "data/teams.json").read_text())
    names = {k: v["name"] for k, v in teams.items()}
    print(json.dumps({"device": device, "dtype": str(dtype), "rev": REVISION}))

    runs = WORK / ("smoke_runs" if smoke else "runs")
    runs.mkdir(parents=True, exist_ok=True)
    jobs = [(s, st, "anon") for s in seasons for st in settings]
    if smoke:
        jobs = [(SEASONS[0], st, "anon") for st in settings]
    else:
        jobs.append((SEASONS[0], "no_line", "named"))
    for season, setting, variant in jobs:
        out_path = runs / f"{season}_{setting}_{variant}.csv"
        if out_path.exists():
            continue
        line = setting == "with_line"
        part = games[games["season"] == season]
        if line:
            part = part[part["p_close"].notna()]
        if smoke:
            part = part.head(64)
        states = [
            state(r, feats, line, names if variant == "named" else None)
            for _, r in part.iterrows()
        ]
        # Answers so far, kept batch by batch: a season takes about 25 minutes
        # and a crash (the Mac failed to page libtorch back in from the SSD)
        # should not lose it.
        partial = out_path.with_suffix(".partial.npy")
        p = list(np.load(partial)) if partial.exists() else []
        started = time.time()
        for i in range(len(p), len(states), BATCH):
            batch = [(s, {"home_win": QUESTION}) for s in states[i : i + BATCH]]
            for out in model.system_one_batch(batch):
                p.append(float(out["answers"]["home_win"]["noul"]))
            np.save(partial, np.asarray(p))
        p = np.asarray(p)
        y = part["home_win"].to_numpy()
        info = {
            "season": season,
            "setting": setting,
            "variant": variant,
            "n": int(len(part)),
            "seconds": round(time.time() - started, 1),
            "brier": round(float(np.mean((p - y) ** 2)), 4),
            "mean_p": round(float(p.mean()), 3),
            "sd_p": round(float(p.std()), 3),
            "mean_p_wins": round(float(p[y == 1].mean()), 3),
            "mean_p_losses": round(float(p[y == 0].mean()), 3),
            # The answer should follow the strength gap, whatever its scale.
            "corr_elo": round(float(np.corrcoef(p, part["elo_diff"])[0, 1]), 3),
        }
        print(json.dumps(info), flush=True)
        if smoke:
            print(json.dumps(states[0], indent=1))
        pd.DataFrame(
            {
                "game_id": part["game_id"],
                "season": season,
                "setting": setting,
                "variant": variant,
                "p_home": p,
            }
        ).to_csv(out_path, index=False)
        partial.unlink()


# --------------------------------------------------------------------------
# score (this project's environment)
# --------------------------------------------------------------------------


def paired(y, a, b, n=2000):
    """Mean Brier(a) - Brier(b) and its bootstrap 95% interval."""
    d = (a - y) ** 2 - (b - y) ** 2
    boots = d[RNG.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return [
        round(float(d.mean()), 5),
        *np.round(np.percentile(boots, [2.5, 97.5]), 5).tolist(),
    ]


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def score() -> None:
    from sklearn.linear_model import LogisticRegression

    from nbapredicts.backtest import classification_metrics as cm

    eq = pd.read_csv(TAB / "equine_preds.csv", dtype={"game_id": str})
    games = pd.read_csv(TAB / "games.csv", dtype={"game_id": str})
    d1 = pd.concat(
        pd.read_csv(f, dtype={"game_id": str})
        for f in sorted((WORK / "runs").glob("*.csv"))
    ).merge(games[["game_id", "home_win"]], on="game_id")
    anon = d1[d1["variant"] == "anon"]
    out = []
    for setting in ("no_line", "with_line"):
        refs = (
            ["equine_no_line", "logistic"]
            if setting == "no_line"
            else ["equine_stack", "p_close"]
        )
        for season in SEASONS:
            cur = anon[(anon["season"] == season) & (anon["setting"] == setting)]
            prev = anon[(anon["season"] == season - 1) & (anon["setting"] == setting)]
            platt = LogisticRegression(C=1e6).fit(
                logit(prev[["p_home"]].to_numpy()), prev["home_win"]
            )
            df = (
                eq[eq["season"] == season]
                .drop(columns="home_win")
                .merge(cur, on="game_id")
                .dropna(subset=refs)
            )
            df["d1"] = df["p_home"]
            df["d1_recal"] = platt.predict_proba(logit(df[["p_home"]].to_numpy()))[:, 1]
            y = df["home_win"].to_numpy()
            row = {
                "season": season,
                "setting": setting,
                "n": int(len(df)),
                "platt": [
                    round(float(platt.coef_[0][0]), 3),
                    round(float(platt.intercept_[0]), 3),
                ],
                "metrics": {
                    c: {
                        k: round(v, 4)
                        for k, v in cm(y, df[c].to_numpy()).items()
                        if k != "n"
                    }
                    for c in refs + ["d1", "d1_recal"]
                },
                "vs": {
                    f"{c}_minus_{r}": paired(y, df[c].to_numpy(), df[r].to_numpy())
                    for c in ("d1", "d1_recal")
                    for r in refs
                },
            }
            out.append(row)
            print(json.dumps(row), flush=True)
    # d1's no-line answer next to the line: logistic on the line's log-odds
    # with and without it, fit on the d1 seasons before the tested one.
    wide = (
        anon[anon["setting"] == "no_line"][["game_id", "season", "home_win", "p_home"]]
        .merge(games[["game_id", "mkt"]], on="game_id")
        .dropna(subset=["mkt"])
    )
    wide["d1"] = logit(wide["p_home"].to_numpy())
    for season in SEASONS:
        tr, te = wide[wide["season"] < season], wide[wide["season"] == season]
        y, p, weights = te["home_win"].to_numpy(), {}, {}
        for name, cols in (("line", ["mkt"]), ("line_d1", ["mkt", "d1"])):
            lr = LogisticRegression(C=1e6).fit(tr[cols], tr["home_win"])
            p[name] = lr.predict_proba(te[cols])[:, 1]
            weights[name] = lr.coef_[0].round(3).tolist()
        row = {
            "season": season,
            "check": "d1 (no line) as an input next to the line",
            "n": int(len(te)),
            "train_games": int(len(tr)),
            "weights": weights,
            "brier": {k: round(cm(y, v)["brier"], 5) for k, v in p.items()},
            "d1_added": paired(y, p["line_d1"], p["line"]),
        }
        out.append(row)
        print(json.dumps(row), flush=True)
    # Names and dates back in: better only if the model recalls results.
    named = d1[d1["variant"] == "named"]
    if len(named):
        both = named.merge(
            anon[["game_id", "setting", "p_home"]],
            on=["game_id", "setting"],
            suffixes=("_named", "_anon"),
        )
        y = both["home_win"].to_numpy()
        row = {
            "season": int(both["season"].iloc[0]),
            "setting": "no_line",
            "check": "names and dates added",
            "n": int(len(both)),
            "brier_named": round(cm(y, both["p_home_named"].to_numpy())["brier"], 4),
            "brier_anon": round(cm(y, both["p_home_anon"].to_numpy())["brier"], 4),
            "named_minus_anon": paired(
                y, both["p_home_named"].to_numpy(), both["p_home_anon"].to_numpy()
            ),
        }
        out.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "docs/results/lfm3_d1.json").write_text(json.dumps(out, indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["infer", "score"])
    parser.add_argument("--smoke", action="store_true", help="infer on 64 games")
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="+",
        default=[CALIBRATION_SEASON, *SEASONS],
    )
    parser.add_argument("--settings", default="no_line,with_line")
    parser.add_argument("--dtype", default="float16", help="infer, on the GPU")
    args = parser.parse_args()
    if args.step == "infer":
        infer(
            args.smoke,
            tuple(args.seasons),
            tuple(args.settings.split(",")),
            args.dtype,
        )
    else:
        score()


if __name__ == "__main__":
    main()
