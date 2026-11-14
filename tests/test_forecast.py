from __future__ import annotations

import numpy as np
import pandas as pd

from nbapredicts.forecast import (
    COVARIATES,
    MIN_GAMES,
    PLAYER_STATS,
    PLAYERS_PER_TEAM,
    Job,
    TimesFM,
    forecast_player_lines,
    margin_per_logit,
    split_total,
)


class FakeModel:
    """Stands in for TimesFM: forecasts the last value, one batch at a time."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []
        self.plain: list[int] = []

    def forecast(self, horizon, inputs):
        self.plain.append(len(inputs))
        # Mirrors return_backcast: a 480-point backcast, then the forecast.
        points = np.stack([np.full(480 + horizon, x[-1]) for x in inputs])
        return points, np.repeat(points[..., None], 10, axis=2)

    def forecast_with_covariates(self, inputs, dynamic_numerical_covariates, **_):
        horizons = [
            len(cov) - len(ctx)
            for ctx, cov in zip(
                inputs, dynamic_numerical_covariates["home"], strict=True
            )
        ]
        assert len(set(horizons)) == 1, "a batch must share one horizon"
        self.batches.append(horizons)
        points = [np.full(h, ctx[-1]) for ctx, h in zip(inputs, horizons, strict=True)]
        quantiles = [np.tile(p[:, None], (1, 10)) for p in points]
        return points, quantiles


def fake_timesfm() -> TimesFM:
    tfm = object.__new__(TimesFM)
    tfm.max_context, tfm.max_horizon = 70, 8
    tfm.xreg_mode, tfm.ridge = "timesfm + xreg", 1.0
    tfm.model = FakeModel()
    return tfm


def job(last: float, horizon: int, n: int = 80) -> Job:
    context = np.arange(n, dtype=float) + last - (n - 1)
    covs = {c: np.zeros(len(context) + horizon) for c in COVARIATES}
    return Job(context=context, covariates=covs, horizon=horizon)


def test_mixed_horizons_are_batched_and_returned_in_order():
    tfm = fake_timesfm()
    jobs = [job(100, 2), job(200, 1), job(300, 2), job(400, 3)]
    out = tfm.run(jobs)
    assert [len(p) for p, _ in out] == [2, 1, 2, 3]
    assert [p[0] for p, _ in out] == [100, 200, 300, 400]
    assert sorted(len(b) for b in tfm.model.batches) == [1, 1, 2]


def test_context_is_trimmed_to_max_context():
    tfm = fake_timesfm()
    seen = []
    real = tfm.model.forecast_with_covariates

    def spy(inputs, dynamic_numerical_covariates, **kw):
        seen.extend(len(i) for i in inputs)
        return real(inputs, dynamic_numerical_covariates, **kw)

    tfm.model.forecast_with_covariates = spy
    tfm.run([job(1, 1)])
    assert seen == [70]


def test_short_contexts_skip_xreg():
    tfm = fake_timesfm()
    out = tfm.run([job(5, 1, n=20), job(7, 1)])
    assert [p[0] for p, _ in out] == [5, 7]
    assert tfm.model.plain == [1] and tfm.model.batches == [[1]]


def test_split_total_keeps_the_total_and_follows_the_pick():
    team_pts = pd.DataFrame(
        {
            "game_id": ["g1", "g1", "g2"],
            "team_id": ["h", "a", "x"],  # g2's opponent has no forecast
            "pts": [111.0, 112.0, 100.0],
            "pts_lo": [96.0, 97.0, 90.0],
            "pts_hi": [126.0, 127.0, 110.0],
        }
    )
    picks = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "home_id": ["h", "x"],
            "away_id": ["a", "y"],
            "p_home": [0.7, 0.9],
        }
    )
    out = split_total(team_pts, picks, (8.0, 0.0)).set_index("team_id")
    margin = 8.0 * np.log(0.7 / 0.3)
    assert out.loc["h", "pts"] + out.loc["a", "pts"] == 223.0
    assert np.isclose(out.loc["h", "pts"] - out.loc["a", "pts"], margin)
    # Intervals move with the forecast; TimesFM's own split is kept.
    assert np.isclose(out.loc["h", "pts_hi"] - out.loc["h", "pts"], 15.0)
    assert out.loc["a", "pts_tfm"] == 112.0
    assert out.loc["x", "pts"] == 100.0


def test_margin_per_logit_recovers_the_spread_line():
    p = np.array([0.2, 0.4, 0.5, 0.65, 0.85])
    odds = pd.DataFrame(
        {"p_close": p, "spread_close": -7.0 * np.log(p / (1 - p)) - 2.0}
    )
    slope, intercept = margin_per_logit(odds)
    assert np.isclose(slope, 7.0) and np.isclose(intercept, 2.0)


def test_players_out_or_without_history_leave_room_for_the_next_ones():
    # 14 players ranked by value: the top 3 are out and the 4th has too few
    # games, so the list should reach down to the 14th to fill 10 places.
    ids = [f"p{i}" for i in range(14)]
    rosters = pd.DataFrame(
        {
            "team_id": "t",
            "athlete_id": ids,
            "name": ids,
            "post_value": np.arange(14, 0, -1, dtype=float),
        }
    )
    games = {pid: MIN_GAMES - 7 if pid == "p3" else MIN_GAMES + 5 for pid in ids}
    pf = pd.DataFrame(
        [
            {
                "athlete_id": pid,
                "team_id": "t",
                "game_id": f"old{k}",
                **dict.fromkeys(PLAYER_STATS, 10.0),
                **dict.fromkeys(COVARIATES, 0.0),
            }
            for pid, n in games.items()
            for k in range(n)
        ]
    )
    tf = pd.DataFrame(
        {"game_id": ["next"], "team_id": ["t"], **{c: [0.0] for c in COVARIATES}}
    )
    injuries = pd.DataFrame({"athlete_id": ["p0", "p1", "p2"], "status": "Out"})
    lines = forecast_player_lines(fake_timesfm(), pf, tf, rosters, ["next"], injuries)
    assert sorted(lines["athlete_id"], key=lambda a: int(a[1:])) == ids[4:]
    assert len(lines) == PLAYERS_PER_TEAM
