from __future__ import annotations

import numpy as np
import pandas as pd

from nbapredicts.pipeline import _clean, select_upcoming, track_record


def schedule(starts: list[str], status: str = "scheduled") -> pd.DataFrame:
    start = pd.to_datetime(starts, utc=True)
    return pd.DataFrame(
        {
            "game_id": [str(i) for i in range(len(starts))],
            "start": start,
            "date": start.tz_convert("America/New_York").date,
            "status": status,
        }
    )


def test_upcoming_window_is_three_eastern_days():
    now = pd.Timestamp("2026-10-27T14:00Z")  # 10:00 ET on the 27th
    games = schedule(
        [
            "2026-10-27T23:30Z",  # tonight
            "2026-10-29T02:00Z",  # 10 PM ET on the 28th
            "2026-10-30T03:30Z",  # 11:30 PM ET on the 29th: still day three
            "2026-10-30T23:00Z",  # the 30th: outside
        ]
    )
    assert select_upcoming(games, now)["game_id"].tolist() == ["0", "1", "2"]


def test_upcoming_falls_back_to_the_next_games_between_slates():
    now = pd.Timestamp("2027-02-13T14:00Z")  # All-Star break
    games = schedule([f"2027-02-{d}T00:30Z" for d in range(19, 21)])
    assert len(select_upcoming(games, now)) == 2


def test_track_record_scores_pre_tip_predictions():
    games = schedule(["2026-10-25T23:30Z", "2026-10-26T23:30Z"], status="final")
    games["home_pts"] = [110.0, 100.0]
    games["away_pts"] = [100.0, 104.0]
    preds = pd.DataFrame(
        {
            "game_id": ["0", "1"],
            "season": [2027, 2027],
            "start": games["start"],
            "home_id": ["a", "b"],
            "away_id": ["b", "a"],
            "p_home": [0.7, 0.6],
            "pred_home_pts": [112.0, 108.0],
            "pred_away_pts": [104.0, 101.0],
        }
    )
    record, results = track_record(
        preds,
        pd.DataFrame(),
        games,
        pd.DataFrame(),
        2027,
        pd.Timestamp("2026-10-27T12:00Z"),
    )
    assert record["n"] == 2 and record["correct"] == 1
    assert np.isclose(record["brier"], ((0.7 - 1) ** 2 + (0.6 - 0) ** 2) / 2)
    assert [r["correct"] for r in results] == [False, True]  # newest first


def test_clean_drops_nan_and_numpy_types():
    out = _clean(
        {
            "a": np.float32(0.5),
            "b": float("nan"),
            "c": [np.int64(3)],
            "d": np.bool_(True),
        }
    )
    assert out == {"a": 0.5, "b": None, "c": [3], "d": True}


def test_track_record_compares_with_the_market_on_the_same_games():
    games = schedule(
        ["2026-10-25T23:30Z", "2026-10-26T23:30Z", "2026-10-26T23:45Z"], status="final"
    )
    games["home_pts"] = [110.0, 100.0, 99.0]
    games["away_pts"] = [100.0, 104.0, 90.0]
    preds = pd.DataFrame(
        {
            "game_id": ["0", "1", "2"],
            "season": [2027] * 3,
            "start": games["start"],
            "home_id": ["a", "b", "c"],
            "away_id": ["b", "a", "d"],
            "p_home": [0.7, 0.6, 0.4],
            "p_market": [0.8, 0.3, np.nan],  # no line for the third game
            "pred_home_pts": [112.0, 108.0, 100.0],
            "pred_away_pts": [104.0, 101.0, 101.0],
        }
    )
    record, _ = track_record(
        preds,
        pd.DataFrame(),
        games,
        pd.DataFrame(),
        2027,
        pd.Timestamp("2026-10-27T12:00Z"),
    )
    vs = record["vs_market"]
    assert vs["n"] == 2
    assert vs["equine_correct"] == 1 and vs["market_correct"] == 2
    assert np.isclose(vs["market_brier"], ((0.8 - 1) ** 2 + 0.3**2) / 2)


def test_track_record_separates_backfilled_predictions():
    games = schedule(
        ["2026-10-24T23:30Z", "2026-10-25T23:30Z", "2026-10-26T23:30Z"],
        status="final",
    )
    games["home_pts"] = [110.0, 100.0, 99.0]
    games["away_pts"] = [100.0, 104.0, 90.0]
    preds = pd.DataFrame(
        {
            "game_id": ["0", "1", "2"],
            "season": [2027] * 3,
            "start": games["start"],
            "home_id": ["a", "b", "c"],
            "away_id": ["b", "c", "a"],
            "p_home": [0.7, 0.6, 0.4],
            "pred_home_pts": [112.0, 108.0, 100.0],
            "pred_away_pts": [104.0, 101.0, 101.0],
            # As read back from the CSV: rows logged before the flag existed are NaN.
            "backfilled": [True, True, np.nan],
        }
    )
    record, results = track_record(
        preds,
        pd.DataFrame(),
        games,
        pd.DataFrame(),
        2027,
        pd.Timestamp("2026-10-27T12:00Z"),
    )
    assert record["n"] == 3 and record["correct"] == 1
    assert record["backfilled"] == 2
    assert record["before_tipoff"] == {"n": 1, "correct": 0}
    assert [r["backfilled"] for r in results] == [False, True, True]
