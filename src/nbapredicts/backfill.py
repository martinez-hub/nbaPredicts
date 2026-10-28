"""Predictions, after the fact, for finished games the daily job never saw.

The season can start before the job does (2026-27 began on Oct 20; the job's
first slate was Oct 27), and a scheduled run can be dropped. For each such
game this reproduces what the job would have published, with nothing from
its day or later:

- the injury history gets each missing day's report, the latest the league's
  CDN has by 6:30 PM ET, the same rule as the archive and the daily job;
- both EQUINE models are refit on the games before the day;
- the stack uses the closing line (the job freezes a game's prediction at
  its last run before tip-off, late in the afternoon);
- TimesFM forecasts each team's points from its games before this one,
  zero-shot, and the total is split by the pick as on the page.

Rows are flagged ``backfilled`` so the track record can tell them from
predictions published before tip-off, and never replace one of those.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .collect import FIRST_SEASON, Store
from .features import (
    apply_pregame_availability,
    build_features,
    player_values,
    pregame_availability,
    team_state,
)
from .forecast import (
    TimesFM,
    backtest_team_points,
    margin_per_logit,
    split_total,
    team_frame,
)
from .impact import apply_impact, monthly_ratings, pregame_impact
from .injuries import _HEADERS, REPORT_CUTOFF, parse_report, record_report
from .stack import StackedWinner
from .winner import WinnerConfig, fit

log = logging.getLogger(__name__)

REPORT_URL = "https://ak-static.cms.nba.com/referee/injury/Injury-Report_{stamp}.pdf"


def report_by_cutoff(day: str) -> tuple[str, bytes] | None:
    """The day's latest report published by ``REPORT_CUTOFF``, from the CDN.

    The season's index page lists only the current day, but the PDFs stay up;
    they come out every 15 minutes, so the candidates are walked back from
    the cutoff.
    """
    t = datetime.strptime(
        f"{day} {REPORT_CUTOFF[0]}:{REPORT_CUTOFF[1]}", "%Y-%m-%d %H:%M"
    )
    start = t.replace(hour=0, minute=0)
    session = requests.Session()
    session.headers.update(_HEADERS)
    while t >= start:
        url = REPORT_URL.format(stamp=t.strftime("%Y-%m-%d_%I_%M%p"))
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code == 200 and resp.content[:4] == b"%PDF":
                return url, resp.content
        except requests.RequestException as err:
            log.warning("report %s: %s", url, err)
        t -= timedelta(minutes=15)
    return None


def fill_reports(
    store: Store, teams: dict[str, Any], days: list[str], season: int
) -> list[str]:
    """Record the cutoff report for each of ``days``; returns the days added."""
    names = [t["name"] for t in teams.values()]
    added = []
    for day in days:
        found = report_by_cutoff(day)
        if found is None:
            log.warning("no injury report for %s", day)
            continue
        url, pdf = found
        if record_report(store, parse_report(pdf, names), url, season):
            added.append(day)
            log.info("recorded %s", url.rsplit("/", 1)[-1])
    return added


def backfill(
    root: Path, store: Store, season: int, now: pd.Timestamp
) -> dict[str, Any]:
    from .pipeline import MODEL_FEATURES

    games = store.load("games")
    preds = store.load("predictions")
    done = set(preds["game_id"]) if not preds.empty else set()
    finals = games[(games["status"] == "final") & (games["season"] == season)]
    todo = finals[~finals["game_id"].isin(done)]
    if todo.empty:
        return {"games": 0}
    days = sorted(str(d) for d in todo["date"].unique())
    teams = json.loads((root / "data/teams.json").read_text())
    # Every finished game day, so later games' availability history is whole too.
    reports = fill_reports(
        store, teams, sorted(str(d) for d in finals["date"].unique()), season
    )

    team_box, player_box = store.load("team_box"), store.load("player_box")
    injury_reports, odds = store.load("injury_reports"), store.load("odds")
    team_ids = {t["name"]: tid for tid, t in teams.items()}
    pv = player_values(player_box, games)
    state = team_state(games, team_box, pv, [])
    feats = build_features(games, state, [])
    feats = apply_pregame_availability(
        feats, pregame_availability(pv, games, injury_reports, team_ids)
    )
    feats = apply_impact(
        feats,
        pregame_impact(
            pv,
            player_box,
            games,
            injury_reports,
            team_ids,
            monthly_ratings(player_box, games),
        ),
    )
    feats = feats.merge(
        odds[["game_id", "p_close", "p_open", "spread_close"]], on="game_id", how="left"
    )

    picks = []
    for day in days:
        d = pd.Timestamp(day).date()
        train = feats[
            (feats["status"] == "final")
            & (feats["season"] > FIRST_SEASON)
            & (feats["date"] < d)
        ]
        test = feats[
            feats["game_id"].isin(todo["game_id"]) & (feats["date"] == d)
        ].reset_index(drop=True)
        model = fit(train, WinnerConfig(), features=MODEL_FEATURES)
        out = pd.concat([test, model.predict(test)], axis=1)
        out["p_model"] = out["p_home"]
        has_line = out["p_close"].notna().to_numpy()
        out["uses_line"] = has_line
        if has_line.any():
            stacked = StackedWinner.fit(
                train.dropna(subset=["p_close"]), features=MODEL_FEATURES
            )
            p = stacked.predict(out[has_line], out.loc[has_line, "p_close"])
            out.loc[has_line, "p_home"] = p["p_home"].to_numpy()
        picks.append(out)
        log.info("backfill %s: %d games, trained on %d", day, len(out), len(train))
    picks = pd.concat(picks, ignore_index=True)

    # TimesFM: each team's points from its games before this one.
    tfm = TimesFM(quantile_head=False)
    pts = backtest_team_points(tfm, team_frame(state), season)
    pts = pts[pts["game_id"].isin(picks["game_id"])].assign(pts=lambda d: d["pred"])
    pts = pts.rename(columns={"lo": "pts_lo", "hi": "pts_hi"})
    pts = split_total(
        pts[["game_id", "team_id", "pts", "pts_lo", "pts_hi"]],
        picks[["game_id", "home_id", "away_id", "p_home"]],
        margin_per_logit(odds[odds["season"] < season]),
    ).set_index(["game_id", "team_id"])

    def team_pred(gid: str, tid: str, col: str) -> float:
        return (
            float(pts.loc[(gid, tid), col]) if (gid, tid) in pts.index else float("nan")
        )

    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = pd.DataFrame(
        [
            {
                "game_id": r.game_id,
                "season": r.season,
                "start": r.start,
                "home_id": r.home_id,
                "away_id": r.away_id,
                "predicted_at": stamp,
                "p_home": r.p_home,
                "novelty": r.novelty,
                "pred_home_pts": team_pred(r.game_id, r.home_id, "pts"),
                "pred_away_pts": team_pred(r.game_id, r.away_id, "pts"),
                "p_market": r.p_close,
                "spread": r.spread_close,
                "p_model": r.p_model,
                "uses_line": bool(r.uses_line),
                "p_open": r.p_open,
                "tfm_home_pts": team_pred(r.game_id, r.home_id, "pts_tfm"),
                "tfm_away_pts": team_pred(r.game_id, r.away_id, "pts_tfm"),
                "backfilled": True,
            }
            for r in picks.itertuples()
        ]
    )
    # Never replace a prediction that was published before tip-off.
    preds = store.load("predictions")
    if not preds.empty:
        rows = rows[~rows["game_id"].isin(preds["game_id"])]
    store.save("predictions", pd.concat([preds, rows], ignore_index=True))
    return {"games": int(len(rows)), "days": days, "reports_added": reports}
