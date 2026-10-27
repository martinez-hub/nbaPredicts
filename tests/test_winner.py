from __future__ import annotations

import numpy as np
import pandas as pd

from nbapredicts.collect import Store
from nbapredicts.features import build_features, player_values, team_state
from nbapredicts.pipeline import log_predictions
from nbapredicts.winner import WinnerConfig, WinnerModel, fit

FAST = WinnerConfig(
    hidden=(16,), emb_dim=8, random_features=64, max_epochs=5, batch_size=64
)


def league_features(league):
    games, team_box, player_box = league
    pv = player_values(player_box, games)
    return build_features(games, team_state(games, team_box, pv, []), [])


def test_fit_predicts_probabilities_and_novelty(league):
    feats = league_features(league)
    wm = fit(feats, FAST)
    out = wm.predict(feats)
    assert out["p_home"].between(0, 1).all()
    assert out["novelty"].between(0, 1).all()
    assert (out["gp_var"] > 0).all()


def test_learns_the_strength_signal():
    from tests.conftest import make_league

    feats = league_features(make_league(n_teams=10, n_days=150, seed=1))
    train = feats[feats["season"] == 2024]
    test = feats[feats["season"] == 2025]
    wm = fit(train, WinnerConfig(max_epochs=15, random_features=256))
    p = wm.predict(test)["p_home"].to_numpy()
    acc = np.mean((p > 0.5) == (test["home_win"].to_numpy() == 1))
    assert acc > 0.6  # coin flip is 0.5; the synthetic gap is learnable


def test_saved_model_round_trips(league, tmp_path):
    feats = league_features(league)
    wm = fit(feats, FAST)
    wm.save(tmp_path)
    loaded = WinnerModel.load(tmp_path)
    np.testing.assert_allclose(
        wm.predict(feats)["p_home"], loaded.predict(feats)["p_home"], rtol=1e-5
    )


def test_prediction_log_freezes_started_games(tmp_path):
    store = Store(tmp_path)
    games = pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "status": ["scheduled", "scheduled"],
            "season": [2027] * 2,
        }
    )
    first = pd.DataFrame(
        {"game_id": ["1", "2"], "season": [2027] * 2, "p_home": [0.6, 0.4]}
    )
    log_predictions(store, "predictions", first, ["game_id"], games)

    games.loc[games["game_id"] == "1", "status"] = "final"
    second = pd.DataFrame(
        {"game_id": ["1", "2"], "season": [2027] * 2, "p_home": [0.9, 0.1]}
    )
    log_predictions(store, "predictions", second, ["game_id"], games)

    logged = store.load("predictions").set_index("game_id")["p_home"]
    assert logged["1"] == 0.6  # started: kept the pre-tip prediction
    assert logged["2"] == 0.1  # not started: refreshed


def test_prediction_log_ignores_an_empty_slate(tmp_path):
    store = Store(tmp_path)
    games = pd.DataFrame({"game_id": ["1"], "status": ["final"], "season": [2027]})
    out = log_predictions(store, "predictions", pd.DataFrame(), ["game_id"], games)
    assert out.empty
