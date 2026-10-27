from __future__ import annotations

import numpy as np

from nbapredicts.forecast import COVARIATES, Job, TimesFM


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
