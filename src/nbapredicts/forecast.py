"""TimesFM 2.5 forecasts of team points and player stat lines.

Each team's points, and each player's per-game stats, form a series indexed by
game. XReg first fits an in-context ridge regression of the series on
covariates known before tip-off (home/away, the opponent's defensive rating
going in, rest days); TimesFM then forecasts, zero-shot, what the regression
leaves over, and the two are added back together. The checkpoint is
``google/timesfm-2.5-200m-pytorch``, whose weights are Apache-2.0; TimesFM 3.0's
weights are licensed for non-production use only, which a public page is not.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODEL_ID = "google/timesfm-2.5-200m-pytorch"
PLAYER_STATS = ("pts", "reb", "ast", "fg3m")
COVARIATES = ("home", "opp_def", "rest_c")
MIN_GAMES = 10  # a player needs this many appearances to be forecast
PLAYERS_PER_TEAM = 10
Q_LO, Q_HI = 1, 9  # quantile channels for the 10th and 90th percentiles
MIN_XREG_CONTEXT = 64  # TimesFM's patch is 32 points; XReg fits on the rest


@dataclass
class Job:
    context: np.ndarray
    covariates: dict[str, np.ndarray]  # each len(context) + horizon
    horizon: int


class TimesFM:
    """A compiled TimesFM 2.5 model, CPU by default."""

    def __init__(
        self,
        max_context: int = 512,
        max_horizon: int = 8,
        batch_size: int = 64,
        # Backtested on 2025-26: "xreg + timesfm" beat "timesfm + xreg" and
        # plain TimesFM on team points (MAE 9.79 vs 9.90 and 10.37) and on all
        # four player stats, with 80% intervals covering ~78% of outcomes.
        xreg_mode: str | None = "xreg + timesfm",
        ridge: float = 1.0,
        quantile_head: bool = True,
    ) -> None:
        import timesfm  # heavy import; keep the rest of the package light

        self.max_context = max_context
        self.max_horizon = max_horizon
        self.xreg_mode = xreg_mode
        self.ridge = ridge
        self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_ID)
        self.model.compile(
            timesfm.ForecastConfig(
                max_context=max_context,
                max_horizon=max_horizon,
                normalize_inputs=True,
                # Off when the point head is fine-tuned: its own nine
                # deciles then give the intervals, and the separate quantile
                # head (untrained on NBA data) is not consulted.
                use_continuous_quantile_head=quantile_head,
                force_flip_invariance=True,
                infer_is_positive=True,
                fix_quantile_crossing=True,
                return_backcast=True,  # required by forecast_with_covariates
                per_core_batch_size=batch_size,
            )
        )

    def run(self, jobs: Sequence[Job]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(point forecast, quantiles) per job, each covering ``job.horizon`` games.

        Jobs are batched by horizon: TimesFM's "timesfm + xreg" mode stacks its
        regression outputs into one array, so a batch must share a horizon.
        """
        for job in jobs:
            if not 1 <= job.horizon <= self.max_horizon:
                raise ValueError(f"horizon {job.horizon} outside 1..{self.max_horizon}")
        results: list[tuple[np.ndarray, np.ndarray] | None] = [None] * len(jobs)
        groups: dict[tuple[int, bool], list[int]] = {}
        for i, j in enumerate(jobs):
            # XReg's in-context regression needs some history to fit (and in
            # "timesfm + xreg" mode skips TimesFM's first 32-point patch), so a
            # short context (a rookie's first games) gets the plain forecast.
            xreg = self.xreg_mode is not None and len(j.context) >= MIN_XREG_CONTEXT
            groups.setdefault((j.horizon, xreg), []).append(i)
        for (horizon, xreg), idx in sorted(groups.items()):
            batch = self._run_batch([jobs[i] for i in idx], horizon, xreg)
            for i, result in zip(idx, batch, strict=True):
                results[i] = result
        return [r for r in results if r is not None]

    def _run_batch(
        self, jobs: list[Job], horizon: int, xreg: bool
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        contexts = [j.context[-self.max_context :].astype(np.float64) for j in jobs]
        if not xreg:
            points, quantiles = self.model.forecast(
                horizon=horizon, inputs=list(contexts)
            )
            # With return_backcast the output is backcast + forecast.
            return [
                (points[i, -horizon:], quantiles[i, -horizon:])
                for i in range(len(jobs))
            ]
        covs = {
            name: [
                list(job.covariates[name][-(len(ctx) + horizon) :].astype(np.float64))
                for job, ctx in zip(jobs, contexts, strict=True)
            ]
            for name in COVARIATES
        }
        points, quantiles = self.model.forecast_with_covariates(
            inputs=[list(c) for c in contexts],
            dynamic_numerical_covariates=covs,
            xreg_mode=self.xreg_mode,
            ridge=self.ridge,
            force_on_cpu=True,
        )
        return [
            (np.asarray(p), np.asarray(q))
            for p, q in zip(points, quantiles, strict=True)
        ]


# --------------------------------------------------------------------------
# Covariates
# --------------------------------------------------------------------------


def team_frame(state: pd.DataFrame) -> pd.DataFrame:
    """``team_state`` rows plus the covariates every series uses."""
    opp = state[["game_id", "team_id", "def_rtg_l"]].rename(
        columns={"team_id": "opp_id", "def_rtg_l": "opp_def"}
    )
    tf = state.merge(opp, on=["game_id", "opp_id"], how="left")
    neutral = tf["neutral"].astype(bool)
    tf["home"] = np.where(neutral, 0.0, np.where(tf["is_home"], 1.0, -1.0))
    tf["opp_def"] = tf["opp_def"].fillna(tf["opp_def"].mean())
    tf["rest_c"] = tf["rest"].clip(0, 4).astype(float)
    tf["final"] = tf["pts"].notna()
    return tf.sort_values(["start", "game_id"]).reset_index(drop=True)


def _job(hist: pd.DataFrame, fut: pd.DataFrame, target: str) -> Job:
    return Job(
        context=hist[target].to_numpy(dtype=np.float64),
        covariates={
            c: np.concatenate([hist[c].to_numpy(float), fut[c].to_numpy(float)])
            for c in COVARIATES
        },
        horizon=len(fut),
    )


def _summarize(result: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, float]:
    point, quantiles = result
    return float(point[0]), float(quantiles[0, Q_LO]), float(quantiles[0, Q_HI])


# --------------------------------------------------------------------------
# Upcoming games
# --------------------------------------------------------------------------


def forecast_team_points(
    tfm: TimesFM, tf: pd.DataFrame, upcoming_ids: Sequence[str]
) -> pd.DataFrame:
    """Points forecast for each team in each upcoming game."""
    jobs, keys = [], []
    for team_id, rows in tf.groupby("team_id", sort=False):
        fut = rows[rows["game_id"].isin(upcoming_ids)].head(tfm.max_horizon)
        hist = rows[rows["final"]]
        if fut.empty or len(hist) < MIN_GAMES:
            continue
        jobs.append(_job(hist, fut, "pts"))
        keys.append((team_id, fut["game_id"].tolist()))
    rows = []
    for (team_id, game_ids), (point, q) in zip(keys, tfm.run(jobs), strict=True):
        for i, gid in enumerate(game_ids):
            rows.append(
                {
                    "game_id": gid,
                    "team_id": team_id,
                    "pts": float(point[i]),
                    "pts_lo": float(q[i, Q_LO]),
                    "pts_hi": float(q[i, Q_HI]),
                }
            )
    return pd.DataFrame(rows)


def player_frame(pv: pd.DataFrame, tf: pd.DataFrame) -> pd.DataFrame:
    """Every appearance joined with its team's covariates for that game."""
    cols = ["game_id", "team_id", *COVARIATES]
    return pv.merge(tf[cols], on=["game_id", "team_id"], how="inner").sort_values(
        ["start", "game_id"]
    )


def forecast_player_lines(
    tfm: TimesFM,
    pf: pd.DataFrame,
    tf: pd.DataFrame,
    rosters: pd.DataFrame,
    upcoming_ids: Sequence[str],
    injuries: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Stat-line forecasts for each rotation player in each of his team's games.

    A team with several games in the window gets a multi-step forecast, one
    step per game, so every game on the slate has projections.
    """
    upcoming = tf[tf["game_id"].isin(upcoming_ids)]
    fut_by_team = {
        tid: rows.head(tfm.max_horizon) for tid, rows in upcoming.groupby("team_id")
    }
    status = {}
    if injuries is not None and not injuries.empty:
        status = dict(zip(injuries["athlete_id"], injuries["status"], strict=False))

    rotation = (
        rosters[rosters["team_id"].isin(fut_by_team)]
        .sort_values("post_value", ascending=False)
        .groupby("team_id")
        .head(PLAYERS_PER_TEAM)
    )
    history = dict(tuple(pf.groupby("athlete_id", sort=False)))
    jobs, keys = [], []
    for p in rotation.itertuples(index=False):
        hist = history.get(p.athlete_id)
        if hist is None or len(hist) < MIN_GAMES:
            continue
        if str(status.get(p.athlete_id, "")).lower() == "out":
            continue
        fut = fut_by_team[p.team_id]
        for stat in PLAYER_STATS:
            jobs.append(_job(hist.dropna(subset=[stat]), fut, stat))
            keys.append((p, stat, fut["game_id"].tolist()))

    log.info("player forecasts: %d series", len(jobs))
    out: dict[tuple[str, str], dict] = {}
    for (p, stat, game_ids), (point, q) in zip(keys, tfm.run(jobs), strict=True):
        for i, gid in enumerate(game_ids):
            row = out.setdefault(
                (p.athlete_id, gid),
                {
                    "game_id": gid,
                    "team_id": p.team_id,
                    "athlete_id": p.athlete_id,
                    "name": p.name,
                    "value": float(p.post_value),
                    "status": status.get(p.athlete_id),
                },
            )
            row[stat] = max(float(point[i]), 0.0)
            row[f"{stat}_lo"] = max(float(q[i, Q_LO]), 0.0)
            row[f"{stat}_hi"] = max(float(q[i, Q_HI]), 0.0)
    return pd.DataFrame(list(out.values()))


# --------------------------------------------------------------------------
# Backtests: one-step-ahead forecasts of games that already happened
# --------------------------------------------------------------------------


def backtest_team_points(tfm: TimesFM, tf: pd.DataFrame, season: int) -> pd.DataFrame:
    """Forecast every final game of ``season`` from the games before it."""
    jobs, rows = [], []
    for team_id, team_rows in tf[tf["final"]].groupby("team_id", sort=False):
        team_rows = team_rows.reset_index(drop=True)
        for i in np.flatnonzero(team_rows["season"].to_numpy() == season):
            if i < MIN_GAMES:
                continue
            target = team_rows.iloc[[i]]
            jobs.append(
                _job(team_rows.iloc[max(0, i - tfm.max_context) : i], target, "pts")
            )
            rows.append(
                {
                    "game_id": target["game_id"].iloc[0],
                    "team_id": team_id,
                    "actual": float(target["pts"].iloc[0]),
                    "baseline": float(target["pts_l"].iloc[0]),
                }
            )
    for row, result in zip(rows, tfm.run(jobs), strict=True):
        row["pred"], row["lo"], row["hi"] = _summarize(result)
    return pd.DataFrame(rows)


def backtest_player_lines(
    tfm: TimesFM,
    pf: pd.DataFrame,
    season: int,
    sample: int = 3000,
    seed: int = 0,
) -> pd.DataFrame:
    """One-step forecasts for a random sample of ``season`` appearances."""
    pf = pf.reset_index(drop=True)
    pf["n"] = pf.groupby("athlete_id").cumcount()
    candidates = pf[(pf["season"] == season) & (pf["n"] >= MIN_GAMES)]
    picks = candidates.sample(min(sample, len(candidates)), random_state=seed)
    history = dict(tuple(pf.groupby("athlete_id", sort=False)))
    jobs, rows = [], []
    for p in picks.itertuples():
        hist_all = history[p.athlete_id]
        hist = hist_all[hist_all["n"] < p.n]
        target = hist_all[hist_all["n"] == p.n]
        for stat in PLAYER_STATS:
            h = hist.dropna(subset=[stat]).tail(tfm.max_context)
            jobs.append(_job(h, target, stat))
            rows.append(
                {
                    "athlete_id": p.athlete_id,
                    "game_id": p.game_id,
                    "stat": stat,
                    "actual": float(getattr(p, stat)),
                    "baseline": float(h[stat].ewm(span=10).mean().iloc[-1]),
                }
            )
    for row, result in zip(rows, tfm.run(jobs), strict=True):
        row["pred"], row["lo"], row["hi"] = _summarize(result)
    return pd.DataFrame(rows)
