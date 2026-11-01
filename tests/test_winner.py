from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nbapredicts.collect import Store
from nbapredicts.features import build_features, player_values, team_state
from nbapredicts.pipeline import log_predictions
from nbapredicts.winner import WinnerConfig, WinnerModel, fit, mirror_rule, mirrored

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


def test_every_model_feature_reads_from_the_away_side():
    from nbapredicts.pipeline import MODEL_FEATURES

    assert {mirror_rule(f) for f in MODEL_FEATURES} == {"flip", "swap", "same"}
    # A new feature has to say how it mirrors before the model will train.
    with pytest.raises(ValueError, match="no mirror rule"):
        mirror_rule("brand_new_feature")


def test_mirroring_twice_gives_the_game_back(league):
    feats = league_features(league)
    from nbapredicts.features import FEATURES

    back = mirrored(mirrored(feats, FEATURES), FEATURES)
    pd.testing.assert_frame_equal(
        back[FEATURES + ["home_win"]], feats[FEATURES + ["home_win"]]
    )
    assert (back["home_side"] == 1.0 - feats["neutral"].astype(float)).all()


def test_the_away_side_gets_the_other_chance(league):
    feats = league_features(league)
    wm = fit(feats, FAST)
    p = wm.predict(feats)["p_home"].to_numpy()
    p_away = wm.predict(mirrored(feats, wm.features))["p_home"].to_numpy()
    np.testing.assert_allclose(p + p_away, 1.0, atol=1e-6)


def test_the_stack_keeps_a_plain_head(league):
    import equine as eq

    from nbapredicts.stack import STACK_CONFIG

    assert not STACK_CONFIG.mirror and STACK_CONFIG.encoder == "mlp"
    feats = league_features(league).assign(mkt=0.0)
    feats["s1"] = feats["elo_diff"] / 100
    wm = fit(feats, STACK_CONFIG, ["mkt", "s1"])
    assert isinstance(wm.model.embedding_model, eq.MLP) and not wm.mirror


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


def test_stacked_model_follows_an_informative_line():
    from nbapredicts.backtest import elo_probability
    from nbapredicts.stack import StackedWinner
    from tests.conftest import make_league

    feats = league_features(make_league(n_teams=10, n_days=150, seed=2))
    # A sharp "market": the Elo probability, which the features only partly see.
    feats["p_close"] = elo_probability(feats)
    train = feats[feats["season"] == 2024]
    test = feats[feats["season"] == 2025]
    stacked = StackedWinner.fit(train)
    p = stacked.predict(test, test["p_close"])["p_home"].to_numpy()
    assert ((p > 0) & (p < 1)).all()
    assert np.corrcoef(p, test["p_close"])[0, 1] > 0.8


def test_backtest_runs_with_the_mirrored_model(league):
    from nbapredicts.backtest import backtest_winner

    feats = league_features(league)
    out = backtest_winner(feats, [2025], config=FAST)
    season = out["seasons"][0]
    assert {"equine_gp", "logistic", "elo"} <= set(season)
    assert len(out["by_novelty"]) == 3
