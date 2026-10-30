from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from nbapredicts.collect import ID_COLS, LINEUP_COLS, Store
from nbapredicts.espn import ESPNError
from nbapredicts.stints import SUB, build_stints, game_stints, update

ROOT = Path(__file__).parents[1]
GAME_ID = "401160628"  # MIN 127 at BKN 126 in overtime, 2019-10-23
FIXTURE = Path(__file__).parent / "fixtures" / f"pbp_{GAME_ID}.json.gz"


@pytest.fixture
def plays() -> dict:
    with gzip.open(FIXTURE, "rt") as f:
        return json.load(f)


@pytest.fixture
def box() -> pd.DataFrame:
    pb = pd.read_csv(ROOT / "data/player_box/2020.csv", dtype={c: str for c in ID_COLS})
    return pb[pb["game_id"] == GAME_ID]


def test_a_game_rebuilds_into_stints_that_match_its_box_score(plays, box):
    # The fixture has two substitutions naming only the player leaving.
    assert sum(p[3] == SUB and len(p[5]) == 1 for p in plays["plays"]) == 2
    stints, check = game_stints(plays, box, (126, 127))
    assert check["status"] == "ok"
    # Back to back from tip-off to the end of overtime, five a side.
    assert stints[0][0] == 0 and stints[-1][1] == 4 * 720 + 300
    assert all(a[1] == b[0] for a, b in zip(stints, stints[1:], strict=False))
    assert all(len(s[2]) == len(s[3]) == 5 for s in stints)
    assert (sum(s[4] for s in stints), sum(s[5] for s in stints)) == (126, 127)
    # A wrong final score fails the check, and then there are no stints.
    none, wrong = game_stints(plays, box, (126, 128))
    assert none == [] and wrong["status"] == "failed" and not wrong["points_ok"]


def test_a_substitution_naming_only_who_left_finds_who_came_in():
    home = [f"h{k}" for k in range(5)]
    away = [f"a{k}" for k in range(5)]
    team_of = {**dict.fromkeys([*home, "h5"], "1"), **dict.fromkeys(away, "2")}
    game = {
        "home_id": "1",
        "starters": {"1": home, "2": away},
        "plays": [
            [1, 1, 720.0, "1", "1", ["h0"], 2, 0],
            [2, 1, 600.0, SUB, "1", ["h2"], 2, 0],  # h2 leaves; who came in?
            [3, 1, 590.0, "1", "1", ["h5"], 4, 0],  # h5, off the floor until now
        ],
    }
    stints, issues = build_stints(game, team_of)
    assert not issues
    assert [s[2] for s in stints] == [
        tuple(home),
        ("h0", "h1", "h5", "h3", "h4"),
    ]
    assert [(s[0], s[1], s[4]) for s in stints] == [(0, 120, 2), (120, 720, 2)]


def test_update_checks_each_game_once_and_retries_failed_downloads(
    tmp_path, plays, box
):
    games = pd.read_csv(ROOT / "data/games/2020.csv", dtype={c: str for c in ID_COLS})
    game = games[games["game_id"] == GAME_ID].assign(
        start=lambda g: pd.to_datetime(g["start"], utc=True)
    )
    ids = {"ok": GAME_ID, "flaky": "1", "none": "2"}
    games = pd.concat([game.assign(game_id=g) for g in ids.values()])
    player_box = pd.concat([box.assign(game_id=g) for g in ids.values()])
    tip = game["start"].iloc[0]
    asked: list[str] = []

    def plays_for(gid: str) -> dict | None:
        asked.append(gid)
        if gid == ids["flaky"]:
            raise ESPNError("HTTP 503")
        return plays if gid == ids["ok"] else None

    store = Store(tmp_path)
    day_after = tip + pd.Timedelta(days=1)
    assert update(store, games, player_box, plays_for, day_after) == {
        "checked": 1,
        "ok": 1,
        "stints": len(game_stints(plays, box, (126, 127))[0]),
    }
    saved = store.load("stints")
    # Athlete ids are digits but must come back as text to match the box.
    assert saved[list(LINEUP_COLS)].map(lambda a: isinstance(a, str)).all().all()
    assert saved["t0"].is_monotonic_increasing

    # Games without plays yet (a failed or an empty download) are asked
    # for again, and only they.
    asked.clear()
    update(store, games, player_box, plays_for, day_after)
    assert sorted(asked) == sorted([ids["flaky"], ids["none"]])
    # Once a game is a few days old it is given up on.
    update(store, games, player_box, plays_for, tip + pd.Timedelta(days=5))
    status = store.load("stint_checks").set_index("game_id")["status"]
    assert status.to_dict() == {
        ids["ok"]: "ok",
        ids["flaky"]: "missing",
        ids["none"]: "missing",
    }
    asked.clear()
    update(store, games, player_box, plays_for, tip + pd.Timedelta(days=6))
    assert asked == []
