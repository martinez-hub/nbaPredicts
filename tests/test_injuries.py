from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nbapredicts import pipeline
from nbapredicts.injuries import combine, normalize_name, parse_report

FIXTURE = Path(__file__).parent / "fixtures" / "injury_report_2026-10-27_12_45AM.pdf"
TEAMS = json.loads((Path(__file__).parents[1] / "data" / "teams.json").read_text())


def test_parses_the_official_report():
    rows = parse_report(FIXTURE.read_bytes(), [t["name"] for t in TEAMS.values()])
    assert len(rows) == 27
    assert rows["status"].value_counts().to_dict() == {
        "Out": 20,
        "Questionable": 6,
        "Available": 1,
    }
    duren = rows[rows["player"] == "Duren, Jalen"].iloc[0]
    assert (duren["team"], duren["status"]) == ("Detroit Pistons", "Out")
    # A row below a team's first line still belongs to that team.
    assert rows[rows["player"] == "Saraf, Ben"].iloc[0]["team"] == "Brooklyn Nets"


def test_names_match_across_sources():
    assert normalize_name("Butler III, Jimmy") == normalize_name("Jimmy Butler III")
    assert normalize_name("Dončić, Luka") == "luka doncic"


def test_official_status_overrides_espn():
    espn = pd.DataFrame(
        {
            "team_id": ["1", "1"],
            "athlete_id": ["a", "b"],
            "name": ["A", "B"],
            "status": ["Day-To-Day", "Out"],
        }
    )
    official = pd.DataFrame(
        {"team_id": ["1"], "athlete_id": ["a"], "name": ["A"], "status": ["Doubtful"]}
    )
    both = combine(espn, official).set_index("athlete_id")
    assert both.loc["a", "status"] == "Doubtful" and both.loc["a", "p_play"] == 0.25
    assert both.loc["b", "p_play"] == 0.0


def test_a_failing_official_report_falls_back_to_espn(monkeypatch):
    espn = pd.DataFrame(
        {"team_id": ["1"], "athlete_id": ["a"], "name": ["A"], "status": ["Out"]}
    )
    monkeypatch.setattr(pipeline, "fetch_injuries", lambda client, ids: espn)

    def blocked(*args, **kwargs):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr(pipeline, "official_injuries", blocked)
    injuries, source = pipeline.gather_injuries(
        None, ["g1"], 2027, pd.Timestamp("2026-10-27T20:00Z"), TEAMS, pd.DataFrame()
    )
    assert source == "ESPN"
    assert injuries["p_play"].tolist() == [0.0]
