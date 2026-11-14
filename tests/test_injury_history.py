from __future__ import annotations

import pandas as pd
import pytest

from nbapredicts import injury_history as ih


def release(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    """Rows of (version time, player, status), all for one team's game that day."""
    return pd.DataFrame(
        {
            "report_datetime": pd.to_datetime([r[0] for r in rows]),
            "team": "Boston Celtics",
            "report_player_name": [r[1] for r in rows],
            "current_status": [r[2] for r in rows],
            "reason": "Injury/Illness - Left Ankle; Sprain",
            "game_date": [r[0][:10] for r in rows],
            "season": 2024,
        }
    )


def test_each_day_keeps_its_last_version_by_the_cutoff():
    rel = release(
        [
            # Hourly era: the hh:30 version is the file named for hh:00.
            ("2024-01-15 17:30", "Doe, Jon", "Questionable"),
            ("2024-01-15 18:30", "Doe, Jon", "Out"),  # the 06PM file: kept
            ("2024-01-15 19:30", "Doe, Jon", "Available"),  # 07PM: too late
            # 15-minute era: names match the times.
            ("2026-01-15 18:30", "Roe, Al", "Doubtful"),  # kept
            ("2026-01-15 18:45", "Roe, Al", "Out"),  # after 6:30 PM
            ("2026-01-15 18:30", None, "NOT YET SUBMITTED"),  # no player
        ]
    )
    out = ih.daily_reports(rel).sort_values("day").reset_index(drop=True)
    assert out["day"].tolist() == ["2024-01-15", "2026-01-15"]
    assert out["status"].tolist() == ["Out", "Doubtful"]
    assert out["published"].tolist() == ["2024-01-15 18:00", "2026-01-15 18:30"]
    assert list(out.columns) == list(ih.COLUMNS)


def games_on(days: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"date": pd.to_datetime(days).date, "status": "final", "game_id": days}
    )


def test_recent_days_come_from_the_league_when_it_has_them():
    # The copy stops at its last refresh: its 09:00 version of Jan 13 is all
    # it has, and Jan 14 is missing. The league has both days' 6:30 PM files.
    rel = release(
        [
            ("2026-01-10 18:30", "Old, Ray", "Out"),
            ("2026-01-13 09:00", "Roe, Al", "Questionable"),
        ]
    )
    games = games_on(["2026-01-10", "2026-01-13", "2026-01-14"])
    asked = []

    def fetch(day: str) -> pd.DataFrame:
        asked.append(day)
        return ih.daily_reports(release([(f"{day} 18:30", "Roe, Al", "Out")]))

    out = ih.injury_history(
        games, pd.Timestamp("2026-01-15T03:00Z"), rel, fetch, first_day="2026-01-01"
    )
    assert asked == ["2026-01-10", "2026-01-13", "2026-01-14"]
    by_day = out.set_index("day")
    assert by_day.loc["2026-01-13", "status"] == "Out"
    assert by_day.loc["2026-01-13", "published"] == "2026-01-13 18:30"
    assert sorted(out["day"].unique()) == ["2026-01-10", "2026-01-13", "2026-01-14"]


def test_the_copy_stands_in_when_the_league_has_nothing():
    rel = release([("2026-01-13 18:30", "Roe, Al", "Doubtful")])
    out = ih.injury_history(
        games_on(["2026-01-13"]),
        pd.Timestamp("2026-01-14T03:00Z"),
        rel,
        lambda day: pd.DataFrame(columns=list(ih.COLUMNS)),
        first_day="2026-01-01",
    )
    assert out["status"].tolist() == ["Doubtful"]


def test_a_short_history_stops_the_run():
    rel = release([("2026-01-10 18:30", "Roe, Al", "Out")])
    games = games_on([f"2026-01-{d:02d}" for d in range(2, 12)])
    with pytest.raises(RuntimeError, match="injury history"):
        ih.injury_history(
            games,
            pd.Timestamp("2026-01-12T03:00Z"),
            rel,
            lambda day: pd.DataFrame(columns=ih.COLUMNS),
            first_day="2026-01-01",
        )
