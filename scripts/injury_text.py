"""Does the injury report's reason, read by d1-2B, predict who plays? (experiment)

The models give each listed player a chance of playing from the status on
the day's official report (Out 0, Doubtful 0.03, Questionable 0.5, Probable
0.9; ``injuries.P_PLAY``) and ignore its reason column ("Injury/Illness -
Left Ankle; Sprain", "Rest", "G League - Two-Way"). This asks whether the
reason, or Liquid AI's d1-2B reading it, says more, before anything goes
into EQUINE.

Rows: players on a team's pre-game roster (``features.pregame_rosters``:
appeared in its previous 10 games) listed Questionable, Doubtful or Probable
on the last report published before tip-off, from 2019-20 (the reason's
standard format). Label: the player logged minutes in that game. Each test
season (2023-24..2025-26) is predicted from the seasons before it:

- the current status constants, and each status's rate in the training
  seasons;
- a logistic on the status plus the player's run of missed games,
  back-to-back, projected minutes and form (no reason);
- the same plus the reason split into category, body part and detail;
- d1-2B reading only the status and reason (no name, team or date: injury
  news is widely covered and LFM3's pretraining runs through 2026), raw,
  recalibrated, and as one more input to the logistic.

Three steps; ``infer`` runs in d1's environment (see ``lfm3_d1.py``):

    uv run python scripts/injury_text.py export
    ~/.venvs/lfm3/bin/python scripts/injury_text.py infer [--smoke]
    uv run python scripts/injury_text.py score

A fourth, ``equine``, puts the best of them (the logistic without the
reason) into EQUINE's availability and impact features in place of the
constants and reruns the published backtest both ways.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "cache.nosync/injury_text"
SEASONS = (2024, 2025, 2026)
FIRST = 2020  # 2019-20: the reason column's "Category - Body part; Detail" form
UNCERTAIN = ("Questionable", "Doubtful", "Probable")
REPO = "LiquidAI/d1-2B"
REVISION = "10e7671d44e4b37700020b1eed3f594c5f2af0b4"
BATCH = 64
RNG = np.random.default_rng(0)
QUESTION = {
    "type": "noul",
    "instructions": (
        "This NBA player is listed on today's official injury report with this"
        " status and reason. Will this player play in tonight's game?"
    ),
    "criteria": {
        "true": "The player plays in the game",
        "false": "The player does not play",
    },
}


def parse_reason(reason: object) -> tuple[str, str, str]:
    """'Injury/Illness - Left Ankle; Sprain' -> (injury/illness, ankle, sprain)."""
    text = re.sub(r"\s+", " ", str(reason) if isinstance(reason, str) else "").strip()
    m = re.match(r"^([^-]*?)\s*-\s*(.*)$", text)
    cat, rest = (m.group(1), m.group(2)) if m else (text, "")
    body, _, detail = rest.partition(";")
    body = re.sub(r"^(left|right|bilateral)\s+", "", body.strip().lower())
    detail = detail.strip().lower().rstrip(" -.")
    return cat.strip().lower() or "none", body or "none", detail or "none"


# --------------------------------------------------------------------------
# export (this project's environment)
# --------------------------------------------------------------------------


def listed_rows(first: int) -> pd.DataFrame:
    """Rostered players listed Questionable, Doubtful or Probable, ``first`` on.

    One row per player and game with the report's status and reason, the
    roster's ``date`` and name ``key``, the inputs of the models and whether
    the player logged minutes.
    """
    from nbapredicts.collect import Store
    from nbapredicts.features import player_values, pregame_rosters
    from nbapredicts.impact import projected_minutes

    store = Store(ROOT / "data")
    games, pb = store.load("games"), store.load("player_box")
    teams = json.loads((ROOT / "data/teams.json").read_text())
    pv = player_values(pb, games)
    app = pv.merge(
        projected_minutes(pb, games)[["game_id", "athlete_id", "proj_min"]],
        on=["game_id", "athlete_id"],
    )
    roster = pregame_rosters(
        app,
        games,
        store.load("injury_reports"),
        {t["name"]: tid for tid, t in teams.items()},
        columns=("proj_min", "post_value"),
        report_columns=("status", "reason"),
    )
    roster = roster.merge(games[["game_id", "season"]], on="game_id")
    rows = roster[roster["status"].isin(UNCERTAIN) & (roster["season"] >= first)]

    # Each team's finished games in order, to count the player's run of
    # games missed going in and whether the team played the day before.
    finals = games[games["status"] == "final"]
    tg = pd.concat(
        [
            finals[["game_id", "start", "date", "home_id"]].rename(
                columns={"home_id": "team_id"}
            ),
            finals[["game_id", "start", "date", "away_id"]].rename(
                columns={"away_id": "team_id"}
            ),
        ]
    ).sort_values("start")
    tg["k"] = tg.groupby("team_id").cumcount()
    day = pd.to_datetime(tg["date"])
    tg["b2b"] = (day - day.groupby(tg["team_id"]).shift(1)).dt.days.eq(1).astype(int)
    played = pb[(~pb["dnp"]) & (pb["min"] > 0)][["game_id", "team_id", "athlete_id"]]
    played = played.merge(tg[["game_id", "team_id", "k"]], on=["game_id", "team_id"])
    rows = rows.merge(tg[["game_id", "team_id", "k", "b2b"]], on=["game_id", "team_id"])
    rows["played"] = (
        pd.MultiIndex.from_frame(rows[["game_id", "athlete_id"]])
        .isin(pd.MultiIndex.from_frame(played[["game_id", "athlete_id"]]))
        .astype(int)
    )
    last = pd.merge_asof(
        rows.sort_values("k"),
        played[["team_id", "athlete_id", "k"]]
        .rename(columns={"k": "k_last"})
        .assign(k=lambda d: d["k_last"])
        .sort_values("k"),
        on="k",
        by=["team_id", "athlete_id"],
        allow_exact_matches=False,
    )
    last["missed"] = (last["k"] - last["k_last"] - 1).fillna(10).clip(0, 10)
    parts = last["reason"].map(parse_reason)
    last["cat"] = parts.str[0]
    last["body"] = parts.str[1]
    last["detail"] = parts.str[2]
    last["reason"] = last["reason"].fillna("")
    return last


def export() -> None:
    last = listed_rows(FIRST)
    keep = [
        "game_id",
        "season",
        "team_id",
        "athlete_id",
        "status",
        "reason",
        "cat",
        "body",
        "detail",
        "missed",
        "b2b",
        "proj_min",
        "post_value",
        "p_play",
        "played",
    ]
    WORK.mkdir(parents=True, exist_ok=True)
    last[keep].to_csv(WORK / "rows.csv", index=False)
    print(
        json.dumps(
            {
                "rows": int(len(last)),
                "by_season": last.groupby("season").size().to_dict(),
                "played_by_status": last.groupby("status")["played"]
                .mean()
                .round(3)
                .to_dict(),
                "pairs": int(last[["status", "reason"]].drop_duplicates().shape[0]),
            }
        )
    )


# --------------------------------------------------------------------------
# infer (d1's environment)
# --------------------------------------------------------------------------


def infer(smoke: bool) -> None:
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device != "cpu" else torch.float32
    path = snapshot_download(REPO, revision=REVISION)
    model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=dtype)
    model = model.to(device)
    rows = pd.read_csv(WORK / "rows.csv", keep_default_na=False)
    pairs = rows[["status", "reason"]].drop_duplicates().reset_index(drop=True)
    if smoke:
        pairs = pairs.sample(96, random_state=0).reset_index(drop=True)
    out_path = WORK / ("smoke_d1.csv" if smoke else "d1.csv")
    started, p = time.time(), []
    for i in range(0, len(pairs), BATCH):
        batch = [
            (
                {"status": s, "injury_report_reason": r or "none given"},
                {"plays": QUESTION},
            )
            for s, r in pairs.iloc[i : i + BATCH].itertuples(index=False)
        ]
        for out in model.system_one_batch(batch):
            p.append(float(out["answers"]["plays"]["noul"]))
    pairs["p_d1"] = p
    pairs.to_csv(out_path, index=False)
    print(
        json.dumps(
            {
                "pairs": int(len(pairs)),
                "seconds": round(time.time() - started, 1),
                "mean_by_status": pairs.groupby("status")["p_d1"]
                .mean()
                .round(3)
                .to_dict(),
            }
        )
    )
    if smoke:
        print(pairs.sort_values("p_d1").to_string(max_colwidth=50))


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


NUMERIC = ["b2b", "proj_min", "post_value"]
NO_REASON = (["status", "missed_bucket"], NUMERIC)


def missed_bucket(missed: pd.Series) -> pd.Series:
    return pd.cut(missed, [-1, 0, 1, 2, 5, 9, 10], labels=False).astype(str)


def model(cats: list[str], nums: list[str]):
    """Logistic on one-hot categories (rare ones pooled) and scaled numbers."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    return make_pipeline(
        ColumnTransformer(
            [
                (
                    "c",
                    OneHotEncoder(
                        min_frequency=20, handle_unknown="infrequent_if_exist"
                    ),
                    cats,
                ),
                ("n", StandardScaler(), nums),
            ]
        ),
        LogisticRegression(C=1.0, max_iter=2000),
    )


def score() -> None:
    from nbapredicts.backtest import classification_metrics as cm

    rows = pd.read_csv(WORK / "rows.csv", keep_default_na=False)
    d1 = pd.read_csv(WORK / "d1.csv", keep_default_na=False)
    rows = rows.merge(d1, on=["status", "reason"], how="left")
    rows["d1"] = logit(rows["p_d1"].to_numpy())
    rows["missed_bucket"] = missed_bucket(rows["missed"])
    designs = {
        "no_reason": NO_REASON,
        "reason": (["status", "missed_bucket", "cat", "body", "detail"], NUMERIC),
        "reason_d1": (
            ["status", "missed_bucket", "cat", "body", "detail"],
            [*NUMERIC, "d1"],
        ),
    }

    out = []
    for season in SEASONS:
        tr, te = rows[rows["season"] < season], rows[rows["season"] == season].copy()
        y = te["played"].to_numpy()
        p = {"constants": te["p_play"].to_numpy()}
        rate = tr.groupby("status")["played"].mean()
        p["status_rate"] = te["status"].map(rate).to_numpy()
        for name, (cats, nums) in designs.items():
            m = model(cats, nums).fit(tr[cats + nums], tr["played"])
            p[name] = m.predict_proba(te[cats + nums])[:, 1]
        p["d1"] = te["p_d1"].to_numpy()
        # d1 recalibrated within each status: its reading of the reason, not
        # of the status the constants already use.
        cal = model(["status"], ["d1"]).fit(tr[["status", "d1"]], tr["played"])
        p["d1_recal"] = cal.predict_proba(te[["status", "d1"]])[:, 1]
        # Rare reasons: the category-body-detail combination seen fewer than
        # 20 times in the training seasons.
        combo = tr.groupby(["cat", "body", "detail"]).size()
        seen = te.set_index(["cat", "body", "detail"]).index.map(combo).fillna(0)
        rare = np.asarray(seen < 20)
        row = {
            "season": season,
            "n": int(len(te)),
            "train_rows": int(len(tr)),
            "played": round(float(y.mean()), 3),
            "brier": {k: round(cm(y, v)["brier"], 5) for k, v in p.items()},
            "log_loss": {k: round(cm(y, v)["log_loss"], 5) for k, v in p.items()},
            "vs": {
                "status_rate_minus_constants": paired(
                    y, p["status_rate"], p["constants"]
                ),
                "no_reason_minus_status_rate": paired(
                    y, p["no_reason"], p["status_rate"]
                ),
                "reason_minus_no_reason": paired(y, p["reason"], p["no_reason"]),
                "d1_recal_minus_status_rate": paired(
                    y, p["d1_recal"], p["status_rate"]
                ),
                "reason_d1_minus_reason": paired(y, p["reason_d1"], p["reason"]),
            },
            "by_status": {
                s: {
                    "n": int((te["status"] == s).sum()),
                    "played": round(float(y[te["status"] == s].mean()), 3),
                    **{
                        k: round(float(np.mean((v - y)[te["status"] == s] ** 2)), 4)
                        for k, v in p.items()
                        if k in ("status_rate", "no_reason", "reason", "reason_d1")
                    },
                }
                for s in UNCERTAIN
            },
            "rare_reasons": {
                "n": int(rare.sum()),
                "reason_d1_minus_reason": paired(
                    y[rare], p["reason_d1"][rare], p["reason"][rare]
                )
                if rare.sum() > 20
                else None,
            },
        }
        out.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "docs/results/injury_text.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


# --------------------------------------------------------------------------
# equine (this project's environment)
# --------------------------------------------------------------------------


def equine_preds(feats: pd.DataFrame) -> pd.DataFrame:
    """EQUINE and its baselines game by game, as in the published backtest."""
    from nbapredicts.pipeline import MODEL_FEATURES
    from nbapredicts.stack import STACK_CONFIG, Stage1, out_of_fold_stage1
    from nbapredicts.winner import WinnerConfig, fit

    rows = []
    for season in SEASONS:
        tr = feats[feats["season"] < season].copy()
        te = feats[feats["season"] == season].copy()
        s1 = Stage1.fit(tr, MODEL_FEATURES)
        out = te[["game_id", "season", "home_win", "p_close"]].copy()
        out["logistic"] = 1 / (1 + np.exp(-s1.logit(te)))
        out["equine_no_line"] = (
            fit(tr, WinnerConfig(), features=MODEL_FEATURES)
            .predict(te)["p_home"]
            .to_numpy()
        )
        trl, tel = tr[tr["p_close"].notna()].copy(), te["p_close"].notna()
        trl["s1"] = out_of_fold_stage1(trl, MODEL_FEATURES)
        te["s1"] = s1.logit(te)
        head = fit(trl, STACK_CONFIG, features=["mkt", "s1"])
        out["equine_stack"] = np.nan
        out.loc[tel, "equine_stack"] = head.predict(te[tel])["p_home"].to_numpy()
        rows.append(out)
    return pd.concat(rows)


def equine() -> None:
    """The learned chance of playing as EQUINE's input, against the constants.

    Each season's listed players get the no-reason logistic fit on the
    seasons before (2018-19, with none before it, keeps the constants); the
    availability and impact features are rebuilt with those chances and
    EQUINE is refit, everything else as published.
    """
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    from timesfm_margin import features

    from nbapredicts.backtest import classification_metrics as cm
    from nbapredicts.collect import FIRST_SEASON, Store
    from nbapredicts.injuries import normalize_name
    from nbapredicts.stack import market_logit

    rows = listed_rows(2019)
    rows["missed_bucket"] = missed_bucket(rows["missed"])
    cats, nums = NO_REASON
    rows["p_learned"] = np.nan
    fit_quality = {}
    for season in sorted(rows["season"].unique()):
        tr, at = rows[rows["season"] < season], rows["season"] == season
        if tr.empty:
            continue
        m = model(cats, nums).fit(tr[cats + nums], tr["played"])
        rows.loc[at, "p_learned"] = m.predict_proba(rows.loc[at, cats + nums])[:, 1]
        y = rows.loc[at, "played"].to_numpy()
        fit_quality[int(season)] = {
            "n": int(at.sum()),
            "constants": round(cm(y, rows.loc[at, "p_play"].to_numpy())["brier"], 4),
            "learned": round(cm(y, rows.loc[at, "p_learned"].to_numpy())["brier"], 4),
        }
    print(json.dumps({"p_play_brier": fit_quality}), flush=True)

    store = Store(ROOT / "data")
    reports = store.load("injury_reports")
    teams = json.loads((ROOT / "data/teams.json").read_text())
    team_ids = {t["name"]: tid for tid, t in teams.items()}
    keys = reports.assign(
        team_id=reports["team"].map(team_ids),
        key=reports["player"].map(normalize_name),
        date=pd.to_datetime(reports["game_date"].fillna(reports["day"])).dt.date,
    )
    learned = rows.dropna(subset=["p_learned"]).drop_duplicates(
        ["team_id", "date", "key"]
    )[["team_id", "date", "key", "p_learned"]]
    keys = keys.merge(learned, on=["team_id", "date", "key"], how="left")
    with_learned = reports.assign(p_play=keys["p_learned"].to_numpy())
    print(
        json.dumps(
            {"report_rows_with_learned": int(with_learned["p_play"].notna().sum())}
        ),
        flush=True,
    )

    preds, feats_by = {}, {}
    for name, reps in (("constants", reports), ("learned", with_learned)):
        feats = features(reps)
        feats = feats[feats["season"] > FIRST_SEASON].copy()
        feats["mkt"] = np.where(
            feats["p_close"].notna(),
            market_logit(feats["p_close"].fillna(0.5)),
            np.nan,
        )
        feats_by[name] = feats.set_index("game_id")
        preds[name] = equine_preds(feats)
        print(json.dumps({"built": name}), flush=True)

    moved = {}
    for c in ("avail_diff", "impact_diff"):
        d = (feats_by["learned"][c] - feats_by["constants"][c]).abs()
        season = feats_by["constants"]["season"]
        moved[c] = d.groupby(season).mean().round(4).loc[2019:2026].to_dict()
    both = preds["constants"].merge(
        preds["learned"].drop(columns=["season", "home_win", "p_close"]),
        on="game_id",
        suffixes=("_constants", "_learned"),
    )
    out = {"p_play_brier": fit_quality, "feature_change": moved, "seasons": []}
    models = ("logistic", "equine_no_line", "equine_stack")
    for season, part in [*both.groupby("season"), ("pooled", both)]:
        part = part.dropna(
            subset=[f"{m}_{v}" for m in models for v in ("constants", "learned")]
        )
        y = part["home_win"].to_numpy()
        out["seasons"].append(
            {
                "season": season if season == "pooled" else int(season),
                "n": int(len(part)),
                "brier": {
                    f"{m}_{v}": round(cm(y, part[f"{m}_{v}"].to_numpy())["brier"], 5)
                    for m in models
                    for v in ("constants", "learned")
                },
                "learned_minus_constants": {
                    m: paired(
                        y,
                        part[f"{m}_learned"].to_numpy(),
                        part[f"{m}_constants"].to_numpy(),
                    )
                    for m in models
                },
            }
        )
        print(json.dumps(out["seasons"][-1]), flush=True)
    (ROOT / "docs/results/injury_pplay_equine.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["export", "infer", "score", "equine"])
    parser.add_argument("--smoke", action="store_true", help="infer: 96 pairs")
    args = parser.parse_args()
    if args.step == "export":
        export()
    elif args.step == "infer":
        infer(args.smoke)
    elif args.step == "equine":
        equine()
    else:
        score()


if __name__ == "__main__":
    main()
