"""Stints from ESPN play-by-play: who was on the floor for each scoring run.

A stint is a stretch of a game between substitutions with the same ten
players on the floor, about 30 a game. ``impact.monthly_ratings`` fits
player ratings on them as well as on whole games.

Two tables, one CSV per season under ``data/``:

- ``stints``: each stint of every game whose stints pass the box-score
  check: when it starts and ends (``t0``, ``t1``, seconds into the game),
  the home margin over it, and the home (``h0``..``h4``) and away
  (``a0``..``a4``) players on the floor.
- ``stint_checks``: one row per final game whose play-by-play was read and
  how its check went, so each game is fetched once. A game that fails the
  check has no stints and is rated on its whole-game row.

The summary endpoint the box scores come from also carries the plays, but
``ESPNClient.summary`` drops them before caching, so they are fetched again,
once per new game.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .collect import LINEUP_COLS, Store, _upsert
from .espn import ESPNClient, ESPNError

log = logging.getLogger(__name__)

SUB = "584"  # ESPN's play type id for a substitution
RETRY_DAYS = 3  # a game whose plays fail to download is retried this long
STINT_COLS = ["game_id", "season", "t0", "t1", "margin", *LINEUP_COLS]
CHECK_COLS = ["game_id", "season", "status", "mean_err", "max_err"]


def clock_seconds(display: str) -> float:
    """'11:36' -> 696.0; '45.2' (the last minute) -> 45.2."""
    if ":" in display:
        m, s = display.split(":", 1)
        return int(m) * 60 + float(s)
    return float(display or 0)


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    """What stints need from a summary: starters and a slim play list.

    A play: [sequence, period, clock seconds left, type id, team id,
    participant ids, home score, away score]. A substitution's participants
    are the player entering, then the player leaving.
    """
    comp = summary["header"]["competitions"][0]
    home = next(c["team"]["id"] for c in comp["competitors"] if c["homeAway"] == "home")
    starters = {}
    for team in summary["boxscore"].get("players", []):
        athletes = team["statistics"][0]["athletes"] if team["statistics"] else []
        starters[team["team"]["id"]] = [
            a["athlete"]["id"] for a in athletes if a.get("starter")
        ]
    plays = [
        [
            int(p.get("sequenceNumber") or 0),
            int(p["period"]["number"]),
            clock_seconds(p.get("clock", {}).get("displayValue", "0")),
            p.get("type", {}).get("id", ""),
            (p.get("team") or {}).get("id"),
            [a["athlete"]["id"] for a in p.get("participants", []) if "athlete" in a],
            int(p.get("homeScore") or 0),
            int(p.get("awayScore") or 0),
        ]
        for p in summary.get("plays", [])
    ]
    return {"home_id": home, "starters": starters, "plays": plays}


def fetch_plays(client: ESPNClient, game_id: str) -> dict[str, Any] | None:
    """One game's compact play-by-play; None when ESPN has no plays for it."""
    data = client.get("summary", event=game_id)
    return compact(data) if data.get("plays") else None


def period_length(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def period_start(period: int) -> float:
    return sum(period_length(p) for p in range(1, period))


def build_stints(
    game: dict[str, Any], team_of: dict[str, str]
) -> tuple[list[tuple], dict[str, Any]]:
    """Stints of one game, and what went wrong if anything.

    A stint: (start s, end s, home lineup, away lineup, home pts, away pts).
    Period 1 opens with the box score's starters. Later periods open with
    the players who act in the period before entering it (older feeds do
    not log substitutions between periods), filled to five from those on
    the floor at the end of the period before who are not seen entering.
    Substitutions at the same moment change the lineup at once; points
    scored at that moment after the change count for the new lineup.
    """
    home = game["home_id"]
    away = next(t for t in game["starters"] if t != home)
    plays = [p for p in game["plays"] if p[1] >= 1]
    issues: dict[str, Any] = {}
    by_period: dict[int, list] = {}
    for p in plays:
        by_period.setdefault(p[1], []).append(p)
    on = {t: list(game["starters"].get(t, [])) for t in (home, away)}
    stints: list[tuple] = []
    score = (0, 0)
    for period in sorted(by_period):
        rows = by_period[period]
        if period > 1:
            first: dict[str, list[str]] = {home: [], away: []}
            entered: set[str] = set()
            for p in rows:
                if p[3] == SUB and len(p[5]) == 2:
                    came, left = p[5]
                    t = team_of.get(left)
                    if left not in entered and t in first and left not in first[t]:
                        first[t].append(left)
                    entered.add(came)
                    continue
                for a in p[5]:
                    t = team_of.get(a)
                    if a not in entered and t in first and a not in first[t]:
                        first[t].append(a)
            for t in (home, away):
                for a in on[t]:
                    if len(first[t]) >= 5:
                        break
                    if a not in first[t] and a not in entered:
                        first[t].append(a)
                if len(first[t]) != 5:
                    issues.setdefault("period_start", []).append((period, t))
                on[t] = first[t][:5]
        base, length = period_start(period), period_length(period)
        t0 = base
        s0 = score
        for i, p in enumerate(rows):
            now = base + length - p[2]
            score = (p[6], p[7])
            if p[3] != SUB or len(p[5]) not in (1, 2):
                continue
            if len(p[5]) == 2:
                came, left = p[5]
            else:
                # "enters the game for X" with the entering player missing:
                # the first teammate to act off the floor came in here.
                left = p[5][0]
                came = next_off_floor(rows[i + 1 :], team_of, left, on)
                if came is None:
                    issues["unknown_entry"] = issues.get("unknown_entry", 0) + 1
                    continue
            t = team_of.get(left, p[4])
            if t not in on:
                continue
            if now > t0:
                stints.append(
                    (
                        t0,
                        now,
                        tuple(on[home]),
                        tuple(on[away]),
                        score[0] - s0[0],
                        score[1] - s0[1],
                    )
                )
                t0, s0 = now, score
            if left in on[t]:
                on[t] = [came if a == left else a for a in on[t]]
            else:
                issues["sub_not_on_floor"] = issues.get("sub_not_on_floor", 0) + 1
                if came not in on[t]:
                    on[t] = [*on[t], came][-5:]
        end = base + length
        if end > t0:
            stints.append(
                (
                    t0,
                    end,
                    tuple(on[home]),
                    tuple(on[away]),
                    score[0] - s0[0],
                    score[1] - s0[1],
                )
            )
    return stints, issues


def next_off_floor(
    rest: list, team_of: dict[str, str], left: str, on: dict[str, list[str]]
) -> str | None:
    """The first player of ``left``'s team to act while off the floor."""
    team = team_of.get(left)
    if team not in on:
        return None
    came_in: set[str] = set()
    for q in rest:
        if q[3] == SUB and len(q[5]) == 2:
            q_came, q_left = q[5]
            if team_of.get(q_left) == team and q_left not in on[team]:
                if q_left != left and q_left not in came_in:
                    return q_left
            came_in.add(q_came)
            continue
        for a in q[5]:
            if team_of.get(a) == team and a not in on[team] and a != left:
                if a not in came_in:
                    return a
    return None


def reconcile(
    stints: list[tuple], box: pd.DataFrame, final: tuple[int, int]
) -> dict[str, Any]:
    """Rebuilt minutes and points against the box score."""
    mins: dict[str, float] = {}
    for t0, t1, h, a, _, _ in stints:
        for p in (*h, *a):
            mins[p] = mins.get(p, 0.0) + (t1 - t0) / 60
    # Some 2012-13 box scores have no minutes: then only the roster checks.
    has_minutes = bool(box["min"].notna().any())
    played = box[(~box["dnp"]) & (box["min"] > 0)] if has_minutes else box
    err = np.array(
        [
            abs(mins.get(a, 0.0) - m)
            for a, m in zip(played["athlete_id"], played["min"], strict=True)
        ]
    )
    # On the floor for under a minute with 0 in the box (rounded down).
    extra = {a for a in set(mins) - set(played["athlete_id"]) if mins[a] > 1.0}
    if not has_minutes:
        err = np.array([])
    pts = (sum(s[4] for s in stints), sum(s[5] for s in stints))
    sizes_ok = all(len(s[2]) == 5 and len(s[3]) == 5 for s in stints)
    return {
        "mean_err": float(err.mean()) if len(err) else np.nan,
        "max_err": float(err.max()) if len(err) else np.nan,
        "extra_players": len(extra),
        "points_ok": pts == final,
        "sizes_ok": sizes_ok,
        "has_minutes": has_minutes,
    }


def game_stints(
    game: dict[str, Any] | None, box: pd.DataFrame, final: tuple[int, int]
) -> tuple[list[tuple], dict[str, Any]]:
    """One game's stints if they pass the box-score check, and the check.

    Passing: five a side throughout, the points add up to the final score,
    nobody on the floor for over a minute who is not in the box score, and
    each player's minutes within 3 of the box score's (0.6 on average).
    """
    if game is None or len(game["starters"]) != 2:
        return [], {"status": "missing"}
    team_of = dict(zip(box["athlete_id"], box["team_id"], strict=True))
    try:
        stints, issues = build_stints(game, team_of)
    except (StopIteration, KeyError, IndexError, ValueError) as err:
        return [], {"status": f"error {err}"}
    rec = reconcile(stints, box, final)
    minutes_ok = (
        rec["mean_err"] <= 0.6 and rec["max_err"] <= 3.0
        if rec["has_minutes"]
        else not issues
    )
    ok = rec["sizes_ok"] and rec["points_ok"] and rec["extra_players"] == 0
    ok = ok and minutes_ok
    check = {"status": "ok" if ok else "failed", **rec, "issues": issues}
    return (stints if ok else []), check


def update(
    store: Store,
    games: pd.DataFrame,
    player_box: pd.DataFrame,
    plays_for: Callable[[str], dict[str, Any] | None],
    now: pd.Timestamp | None = None,
) -> dict[str, int]:
    """Stints for every final game with a box score not checked yet.

    ``plays_for(game_id)``: the game's compact play-by-play (``fetch_plays``
    in the daily job). A download that fails leaves no check, so the next
    run tries again, until the game is ``RETRY_DAYS`` old.
    """
    now = now or pd.Timestamp.now(tz="UTC")
    checks = store.load("stint_checks")
    done = set(checks["game_id"]) if not checks.empty else set()
    finals = games[(games["status"] == "final") & ~games["game_id"].isin(done)]
    finals = finals[finals["game_id"].isin(set(player_box["game_id"]))]
    box = player_box[player_box["game_id"].isin(finals["game_id"])]
    box_by_game = {gid: part for gid, part in box.groupby("game_id")}
    rows, new_checks = [], []
    for g in finals.itertuples(index=False):
        try:
            game = plays_for(g.game_id)
        except ESPNError as err:
            log.warning("plays for %s failed: %s", g.game_id, err)
            if now - pd.Timestamp(g.start) < pd.Timedelta(days=RETRY_DAYS):
                continue
            game = None
        except (KeyError, IndexError, TypeError, StopIteration) as err:
            log.warning("plays for %s unreadable: %s", g.game_id, err)
            game = None
        final = (int(g.home_pts), int(g.away_pts))
        stints, check = game_stints(game, box_by_game[g.game_id], final)
        new_checks.append(
            {
                "game_id": g.game_id,
                "season": g.season,
                "status": check["status"],
                "mean_err": check.get("mean_err"),
                "max_err": check.get("max_err"),
            }
        )
        rows += [
            (g.game_id, g.season, t0, t1, hp - ap, *h, *a)
            for t0, t1, h, a, hp, ap in stints
        ]
    if new_checks:
        store.save(
            "stints",
            _upsert(
                store.load("stints"),
                pd.DataFrame(rows, columns=STINT_COLS),
                ["game_id", "t0"],
            ),
        )
        store.save(
            "stint_checks",
            _upsert(checks, pd.DataFrame(new_checks, columns=CHECK_COLS), ["game_id"]),
        )
    statuses = pd.Series([c["status"] for c in new_checks], dtype=str)
    return {
        "checked": len(new_checks),
        "ok": int((statuses == "ok").sum()),
        "stints": len(rows),
    }
