"""NVIDIA's tabular and relational foundation models against EQUINE (experiment).

Kumo Tabular and TabICLv2 (both in ``structured-data-models``) are in-context
learners: given labeled rows they predict new ones without training. Kumo
Relational also reads rows of related tables. Walk-forward on 2023-24..
2025-26, each test season predicted from the seasons before it, on the same
pre-game inputs as EQUINE:

- without the line: the 28 game features, against features-only EQUINE and
  the logistic stage 1;
- with the line: the features plus the closing line's log-odds, against the
  published EQUINE stack and the line itself;
- Kumo Relational gets the game features plus one row per player on each
  roster (pre-game form, plus-minus rating, projected minutes, chance of
  playing), to see whether it can do better with players than the summed
  availability and impact features.

Three steps; ``infer`` runs in an environment with structured-data-models
(it does not import this package):

    uv run python scripts/kumo_tabular.py export
    /Volumes/ExtremeSSD/kumo-sdm/bin/python scripts/kumo_tabular.py infer [--smoke]
    uv run python scripts/kumo_tabular.py score
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "cache.nosync/kumo_tab"
SEASONS = (2024, 2025, 2026)
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# export (this project's environment)
# --------------------------------------------------------------------------


def export() -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    from timesfm_margin import features

    from nbapredicts.collect import FIRST_SEASON, Store
    from nbapredicts.features import player_values, pregame_rosters
    from nbapredicts.impact import monthly_ratings, projected_minutes
    from nbapredicts.pipeline import MODEL_FEATURES
    from nbapredicts.stack import STACK_CONFIG, Stage1, market_logit, out_of_fold_stage1
    from nbapredicts.winner import WinnerConfig, fit

    feats = features()
    feats = feats[feats["season"] > FIRST_SEASON].copy()
    feats["mkt"] = np.where(
        feats["p_close"].notna(), market_logit(feats["p_close"].fillna(0.5)), np.nan
    )
    cols = [
        "game_id",
        "season",
        "start",
        "home_id",
        "away_id",
        "home_win",
        "p_close",
        "mkt",
    ]
    WORK.mkdir(parents=True, exist_ok=True)
    feats[cols + MODEL_FEATURES].to_csv(WORK / "games.csv", index=False)
    (WORK / "features.json").write_text(json.dumps(MODEL_FEATURES) + "\n")

    # EQUINE and its baselines, game by game, as in the published backtest.
    rows = []
    for season in SEASONS:
        tr, te = (
            feats[feats["season"] < season].copy(),
            feats[feats["season"] == season].copy(),
        )
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
        print(json.dumps({"season": season, "games": int(len(out))}), flush=True)
    pd.concat(rows).to_csv(WORK / "equine_preds.csv", index=False)

    # One row per player on each roster, as knowable before tip-off.
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
    )
    roster["month"] = (
        pd.to_datetime(roster["start"], utc=True)
        .dt.tz_localize(None)
        .dt.to_period("M")
        .dt.to_timestamp()
        .dt.tz_localize("UTC")
    )
    roster = roster.merge(
        monthly_ratings(pb, games), on=["month", "athlete_id"], how="left"
    )
    roster = roster[roster["game_id"].isin(feats["game_id"])]
    home = roster["game_id"].map(feats.set_index("game_id")["home_id"])
    roster["side"] = np.where(roster["team_id"] == home, 1.0, -1.0)
    roster["rating"] = roster["rating"].fillna(0.0)
    keep = ["game_id", "side", "post_value", "proj_min", "rating", "p_play"]
    roster[keep].to_csv(WORK / "players.csv", index=False)
    print(json.dumps({"player_rows": int(len(roster))}))


# --------------------------------------------------------------------------
# infer (structured-data-models' environment)
# --------------------------------------------------------------------------


def _probs(out) -> np.ndarray:
    """P(home win): the output column named for class 1.

    Classification outputs hold one column per class in the order the
    context's categories were seen (not sorted), each named by its value.
    """
    if isinstance(out, dict):
        out = next(iter(out.values()))
    df = out.to_pandas()
    for name in df.columns:
        if str(name) in ("1", "1.0", "True"):
            return df[name].to_numpy(dtype=float)
    raise ValueError(f"no column for class 1 in {list(df.columns)}")


def infer(smoke: bool, models: list[str], context_seasons: int) -> None:
    import sdm
    import torch
    from sdm import RelatedTables, TableTensor
    from sdm.models import KumoRelational, KumoTabular, TabICLv2

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    feats = json.loads((WORK / "features.json").read_text())
    games = pd.read_csv(WORK / "games.csv", dtype={"game_id": str})
    players = pd.read_csv(WORK / "players.csv", dtype={"game_id": str})
    print(
        json.dumps({"device": str(device), "sdm": getattr(sdm, "__version__", "?")}),
        flush=True,
    )

    def table(df: pd.DataFrame, cols: list[str], target: bool) -> TableTensor:
        # MPS has no float64: features as float32, the label as an integer.
        out = df[cols].astype("float32").reset_index(drop=True)
        stypes = {c: "numerical" for c in cols}
        if target:
            out["home_win"] = df["home_win"].astype("int64").to_numpy()
            stypes["home_win"] = "categorical"
        return TableTensor.from_pandas(df=out, stypes=stypes, device=device)

    def tabular(model, ctx, qry, cols) -> np.ndarray:
        t = table(ctx, cols, True)
        model.fit(x=t.drop_columns("home_win"), y=t[:, "home_win"], num_estimators=8)
        q = table(qry, cols, False)
        out = [_probs(model.predict(q[i : i + 512])) for i in range(0, len(qry), 512)]
        model.clear()
        return np.concatenate(out)

    def relational(model, ctx, qry, cols) -> np.ndarray:
        def tables(df: pd.DataFrame, target: bool):
            task = df[["game_id", *cols] + (["home_win"] if target else [])].copy()
            task["game_key"] = pd.factorize(task["game_id"])[0]
            ids = task["game_key"].to_numpy()
            key = dict(zip(task["game_id"], ids, strict=True))
            p = players[players["game_id"].isin(key)]
            data = {c: task[c].astype(float).tolist() for c in cols}
            stypes = {c: "numerical" for c in cols}
            data["game_key"], stypes["game_key"] = ids.tolist(), "id"
            if target:
                data["home_win"], stypes["home_win"] = (
                    task["home_win"].astype(int).tolist(),
                    "categorical",
                )
            t = TableTensor.from_columns(data, stypes=stypes, device=device)
            pcols = ["side", "post_value", "proj_min", "rating", "p_play"]
            related = {
                "games": TableTensor.from_columns(
                    {
                        "game_key": ids.tolist(),
                        "n_players": p.groupby("game_id")
                        .size()
                        .reindex(task["game_id"])
                        .fillna(0)
                        .astype(float)
                        .tolist(),
                    },
                    stypes={"game_key": "id", "n_players": "numerical"},
                    device=device,
                ),
                "players": TableTensor.from_columns(
                    {
                        "game_key": p["game_id"].map(key).astype(int).tolist(),
                        **{c: p[c].astype(float).tolist() for c in pcols},
                    },
                    stypes={"game_key": "id", **{c: "numerical" for c in pcols}},
                    device=device,
                ),
            }
            rel = RelatedTables(
                tables=related,
                relationships=[
                    {
                        "left_table": "players",
                        "left_columns": "game_key",
                        "right_table": "games",
                        "right_columns": "game_key",
                    }
                ],
                task_links=[
                    {
                        "task_columns": "game_key",
                        "table": "games",
                        "table_columns": "game_key",
                    }
                ],
            )
            return t, rel

        t, rc = tables(ctx, True)
        q, rq = tables(qry, False)
        model.fit(t.drop_columns("home_win"), t[:, "home_win"], rc, num_hops=1)
        out = _probs(model.predict(q, rq))
        model.clear()
        return out

    builders = {
        "kumo_tabular": lambda: KumoTabular(task="classification", device=device),
        "tabicl_v2": lambda: TabICLv2(device=device),
        "kumo_relational": lambda: KumoRelational(task="classification", device=device),
    }
    results = []
    for season in SEASONS:
        for setting in ("no_line", "with_line"):
            cols = feats + (["mkt"] if setting == "with_line" else [])
            pool = games if setting == "no_line" else games[games["mkt"].notna()]
            ctx, qry = pool[pool["season"] < season], pool[pool["season"] == season]
            if smoke:
                ctx, qry = ctx.tail(2000), qry.head(300)
            for name in models:
                c = ctx[ctx["season"] >= season - context_seasons]
                started = time.time()
                model = builders[name]()
                run = relational if name == "kumo_relational" else tabular
                p = run(model, c, qry, cols)
                y = qry["home_win"].to_numpy()
                info = {
                    "season": season,
                    "setting": setting,
                    "model": name,
                    "context": int(len(c)),
                    "query": int(len(qry)),
                    "seconds": round(time.time() - started, 1),
                    "brier": round(float(np.mean((p - y) ** 2)), 4),
                    "mean_p_wins": round(float(p[y == 1].mean()), 3),
                    "mean_p_losses": round(float(p[y == 0].mean()), 3),
                }
                print(json.dumps(info), flush=True)
                results.append(
                    pd.DataFrame(
                        {
                            "game_id": qry["game_id"],
                            "season": season,
                            "setting": setting,
                            "model": name,
                            "p_home": p,
                        }
                    )
                )
                del model
            if smoke:
                break
        if smoke:
            break
    out = WORK / ("smoke_preds.csv" if smoke else "fm_preds.csv")
    pd.concat(results).to_csv(out, index=False)


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


def score() -> None:
    from nbapredicts.backtest import classification_metrics as cm

    eq = pd.read_csv(WORK / "equine_preds.csv", dtype={"game_id": str})
    fm = pd.read_csv(WORK / "fm_preds.csv", dtype={"game_id": str})
    out = []
    for (season, setting), part in fm.groupby(["season", "setting"]):
        wide = part.pivot(index="game_id", columns="model", values="p_home")
        df = eq[eq["season"] == season].set_index("game_id").join(wide, how="inner")
        refs = (
            ["equine_no_line", "logistic"]
            if setting == "no_line"
            else ["equine_stack", "p_close"]
        )
        df = df.dropna(subset=refs + list(wide.columns))
        y = df["home_win"].to_numpy()
        row = {
            "season": int(season),
            "setting": setting,
            "n": int(len(df)),
            "metrics": {},
        }
        for c in refs + list(wide.columns):
            m = cm(y, df[c].to_numpy())
            row["metrics"][c] = {k: round(v, 4) for k, v in m.items() if k != "n"}
        row["vs"] = {
            f"{c}_minus_{r}": paired(y, df[c].to_numpy(), df[r].to_numpy())
            for c in wide.columns
            for r in refs
        }
        out.append(row)
        print(json.dumps(row), flush=True)
    (ROOT / "docs/results/kumo_tabular.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["export", "infer", "score"])
    parser.add_argument("--smoke", action="store_true", help="infer on a small slice")
    parser.add_argument(
        "--models",
        default="kumo_tabular,tabicl_v2,kumo_relational",
        help="infer: comma list",
    )
    parser.add_argument(
        "--context-seasons",
        type=int,
        default=4,
        help="infer: labeled seasons each model reads (in-context cost grows with the"
        " square of the rows; EQUINE fit on the last five seasons matched all of them)",
    )
    args = parser.parse_args()
    if args.step == "export":
        export()
    elif args.step == "infer":
        infer(args.smoke, args.models.split(","), args.context_seasons)
    else:
        score()


if __name__ == "__main__":
    main()
