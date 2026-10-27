"""The daily job: collect, retrain, forecast, score and publish.

Every run is idempotent. Predictions are logged per game and overwritten until
tip-off, so the logged prediction for a finished game is the last one made
before it started; the live track record on the page is computed from that log.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .collect import (
    EASTERN,
    FIRST_SEASON,
    Store,
    current_season,
    fetch_injuries,
    update,
)
from .espn import ESPNClient, ESPNError, parse_odds
from .features import (
    FEATURES,
    build_features,
    current_rosters,
    player_values,
    team_state,
)
from .finetune import daily_finetune
from .forecast import (
    COVARIATES,
    MODEL_ID,
    PLAYER_STATS,
    TimesFM,
    forecast_player_lines,
    forecast_team_points,
    player_frame,
    team_frame,
)
from .injuries import combine, official_injuries
from .winner import WinnerConfig, WinnerModel, fit

log = logging.getLogger(__name__)

WINDOW_DAYS = 3  # upcoming games shown: today and the next two days (ET)
RESULT_DAYS = 7  # finished games listed under recent results


def season_label(season: int) -> str:
    return f"{season - 1}-{str(season)[-2:]}"


def select_upcoming(games: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    sched = games[(games["status"] == "scheduled") & (games["start"] > now)]
    today = now.tz_convert(EASTERN).date()
    last = (now.tz_convert(EASTERN) + pd.Timedelta(days=WINDOW_DAYS - 1)).date()
    window = sched[(sched["date"] >= today) & (sched["date"] <= last)]
    if window.empty:  # between slates (All-Star break, offseason): the next 10
        window = sched.head(10)
    return window.sort_values(["start", "game_id"])


def load_teams(client: ESPNClient, path: Path) -> dict[str, dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text())
    teams = {
        str(t["id"]): {
            "abbr": t["abbreviation"],
            "name": t["displayName"],
            "short": t.get("shortDisplayName") or t["name"],
            "color": "#" + (t.get("color") or "888888"),
        }
        for t in client.teams()
    }
    path.write_text(json.dumps(teams, indent=1, sort_keys=True) + "\n")
    return teams


def gather_injuries(
    client: ESPNClient,
    upcoming_ids: list[str],
    season: int,
    now: pd.Timestamp,
    teams: dict[str, dict[str, Any]],
    rosters: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    """ESPN's injury list, refined by the NBA's official report when reachable.

    Never raises: if the official report is blocked or does not parse, the run
    goes on with ESPN's list alone.
    """
    espn = fetch_injuries(client, upcoming_ids)
    official, source = pd.DataFrame(), "ESPN"
    if upcoming_ids:
        try:
            official, url = official_injuries(season, now, teams, rosters)
            if url and not official.empty:
                source = "NBA official report " + url.rsplit("/", 1)[-1]
        except Exception as err:  # network, WAF, or PDF layout changes
            log.warning("official injury report unavailable: %s", err)
    return combine(espn, official), source


def fetch_market(client: ESPNClient, upcoming: pd.DataFrame) -> dict[str, dict]:
    """Sportsbook lines ESPN shows for the slate, keyed by game id.

    Shown next to EQUINE's pick and logged for the season comparison; the
    model never sees them.
    """
    lines: dict[str, dict] = {}
    for day in sorted(set(upcoming["date"])):
        try:
            events = client.scoreboard(pd.Timestamp(day).strftime("%Y%m%d"))
        except ESPNError as err:
            log.warning("odds for %s failed: %s", day, err)
            continue
        for event in events:
            odds = parse_odds(event)
            if odds is not None and odds["p_market"] is not None:
                lines[odds["game_id"]] = odds
    log.info("market lines for %d of %d games", len(lines), len(upcoming))
    return lines


# --------------------------------------------------------------------------
# Prediction log and track record
# --------------------------------------------------------------------------


def log_predictions(
    store: Store, name: str, new: pd.DataFrame, keys: list[str], games: pd.DataFrame
) -> pd.DataFrame:
    """Upsert ``new`` into the ``name`` log, never touching started games."""
    old = store.load(name)
    if new.empty:  # no slate (offseason, or nothing scheduled yet)
        return old
    if not old.empty:
        started = set(games.loc[games["status"] != "scheduled", "game_id"])
        # Rows for games that have started are frozen; the rest are replaced.
        replace = new[~new["game_id"].isin(started)]
        keep = old.merge(replace[keys], on=keys, how="left", indicator=True)
        old = old[(keep["_merge"] == "left_only").to_numpy()]
        merged = pd.concat([old, replace], ignore_index=True)
    else:
        merged = new
    if not merged.empty:
        store.save(name, merged)
    return merged


def _finite(x: Any) -> Any:
    if isinstance(x, float) and not math.isfinite(x):
        return None
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else round(float(x), 4)
    if isinstance(x, float):
        return round(x, 4)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def _clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return _finite(obj)


def track_record(
    preds: pd.DataFrame,
    player_preds: pd.DataFrame,
    games: pd.DataFrame,
    player_box: pd.DataFrame,
    season: int,
    now: pd.Timestamp,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    finals = games[games["status"] == "final"]
    if preds.empty:
        return {}, []
    scored = preds.merge(
        finals[["game_id", "date", "home_pts", "away_pts"]], on="game_id", how="inner"
    )
    scored = scored[scored["season"] == season]
    if scored.empty:
        return {"season": season_label(season), "n": 0}, []
    y = (scored["home_pts"] > scored["away_pts"]).astype(float)
    correct = (scored["p_home"] > 0.5) == (y == 1)
    pts_err = pd.concat(
        [
            scored["pred_home_pts"] - scored["home_pts"],
            scored["pred_away_pts"] - scored["away_pts"],
        ]
    ).abs()
    record: dict[str, Any] = {
        "season": season_label(season),
        "n": int(len(scored)),
        "correct": int(correct.sum()),
        "accuracy": float(correct.mean()),
        "brier": float(((scored["p_home"] - y) ** 2).mean()),
        "team_pts_mae": float(pts_err.mean()),
    }
    if "p_market" in scored:
        both = scored.dropna(subset=["p_market"])
        if not both.empty:
            yb = (both["home_pts"] > both["away_pts"]).astype(float)
            record["vs_market"] = {
                "n": int(len(both)),
                "equine_correct": int(((both["p_home"] > 0.5) == (yb == 1)).sum()),
                "market_correct": int(((both["p_market"] > 0.5) == (yb == 1)).sum()),
                "equine_brier": float(((both["p_home"] - yb) ** 2).mean()),
                "market_brier": float(((both["p_market"] - yb) ** 2).mean()),
            }
    if not player_preds.empty:
        pp = player_preds.merge(
            player_box[["game_id", "athlete_id", "dnp", *PLAYER_STATS]],
            on=["game_id", "athlete_id"],
            how="inner",
            suffixes=("", "_actual"),
        )
        pp = pp[(~pp["dnp"]) & pp["game_id"].isin(scored["game_id"])]
        record["player_mae"] = {
            s: float((pp[s] - pp[f"{s}_actual"]).abs().mean()) for s in PLAYER_STATS
        }
        record["player_n"] = int(len(pp))

    recent_cut = (now.tz_convert(EASTERN) - pd.Timedelta(days=RESULT_DAYS)).date()
    recent = scored[scored["date"] >= recent_cut].sort_values("start", ascending=False)
    results = [
        {
            "game_id": r.game_id,
            "date": str(r.date),
            "home": r.home_id,
            "away": r.away_id,
            "home_pts": r.home_pts,
            "away_pts": r.away_pts,
            "p_home": r.p_home,
            "p_market": getattr(r, "p_market", None),
            "pred_home_pts": r.pred_home_pts,
            "pred_away_pts": r.pred_away_pts,
            "correct": bool((r.p_home > 0.5) == (r.home_pts > r.away_pts)),
        }
        for r in recent.itertuples()
    ]
    return record, results


# --------------------------------------------------------------------------
# Daily run
# --------------------------------------------------------------------------


def daily(
    root: Path,
    store: Store,
    client: ESPNClient,
    now: pd.Timestamp | None = None,
    collect: bool = True,
    config: WinnerConfig | None = None,
    finetune: bool = True,
) -> dict[str, Any]:
    now = now or pd.Timestamp.now(tz="UTC")
    season = current_season(now.to_pydatetime())
    if collect:
        log.info("collect: %s", update(store, client, [season]))

    games = store.load("games")
    team_box = store.load("team_box")
    player_box = store.load("player_box")
    teams = load_teams(client, store.data_dir / "teams.json")

    upcoming = select_upcoming(games, now)
    upcoming_ids = upcoming["game_id"].tolist()
    pv = player_values(player_box, games)
    injuries, injury_source = gather_injuries(
        client, upcoming_ids, season, now, teams, current_rosters(pv)
    )
    market = fetch_market(client, upcoming)
    log.info("%d upcoming games, %d injury entries", len(upcoming_ids), len(injuries))

    state = team_state(games, team_box, pv, upcoming_ids, injuries)
    feats = build_features(games, state, upcoming_ids)

    train = feats[(feats["status"] == "final") & (feats["season"] > FIRST_SEASON)]
    wm = fit(train, config)
    wm.save(root / "models")
    WinnerModel.load(root / "models")  # the saved file must round-trip
    up = feats[feats["game_id"].isin(upcoming_ids)].reset_index(drop=True)
    winner = pd.concat([up, wm.predict(up)], axis=1)

    # The point head's own deciles give the intervals (better calibrated in
    # the 2025-26 backtest than the separate quantile head), and that head is
    # what the daily fine-tune trains.
    tfm = TimesFM(quantile_head=False)
    tf = team_frame(state)
    pf = player_frame(pv, tf)
    fine_tune = daily_finetune(tfm, tf, pf) if finetune else {"used": False}
    team_pts = forecast_team_points(tfm, tf, upcoming_ids)
    lines = forecast_player_lines(
        tfm, pf, tf, current_rosters(pv), upcoming_ids, injuries
    )

    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    pts = team_pts.set_index(["game_id", "team_id"]) if not team_pts.empty else None

    def team_pred(gid: str, tid: str, col: str) -> float:
        if pts is None or (gid, tid) not in pts.index:
            return float("nan")
        return float(pts.loc[(gid, tid), col])

    pred_rows = [
        {
            "game_id": r.game_id,
            "season": r.season,
            "start": r.start,
            "home_id": r.home_id,
            "away_id": r.away_id,
            "predicted_at": stamp,
            "p_home": r.p_home,
            "novelty": r.novelty,
            "p_market": market.get(r.game_id, {}).get("p_market"),
            "spread": market.get(r.game_id, {}).get("spread"),
            "pred_home_pts": team_pred(r.game_id, r.home_id, "pts"),
            "pred_away_pts": team_pred(r.game_id, r.away_id, "pts"),
        }
        for r in winner.itertuples()
    ]
    preds = log_predictions(
        store, "predictions", pd.DataFrame(pred_rows), ["game_id"], games
    )
    if not lines.empty:
        pl = lines.merge(upcoming[["game_id", "season"]], on="game_id")
        pl["predicted_at"] = stamp
        player_preds = log_predictions(
            store,
            "player_predictions",
            pl[
                [
                    "game_id",
                    "season",
                    "athlete_id",
                    "team_id",
                    "name",
                    "predicted_at",
                    *PLAYER_STATS,
                ]
            ],
            ["game_id", "athlete_id"],
            games,
        )
    else:
        player_preds = store.load("player_predictions")

    record, results = track_record(preds, player_preds, games, player_box, season, now)

    inj_by_team: dict[str, list[dict[str, str]]] = {}
    for r in injuries.itertuples():
        inj_by_team.setdefault(r.team_id, []).append(
            {"name": r.name, "status": r.status}
        )

    def players_for(gid: str, tid: str) -> list[dict[str, Any]]:
        if lines.empty:
            return []
        rows = lines[(lines["game_id"] == gid) & (lines["team_id"] == tid)]
        rows = rows.sort_values("pts", ascending=False)
        keep = ["name", "status"] + [
            f"{s}{x}" for s in PLAYER_STATS for x in ("", "_lo", "_hi")
        ]
        return rows[keep].to_dict("records")

    out_games = []
    for r in winner.itertuples():
        out_games.append(
            {
                "game_id": r.game_id,
                "start": r.start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "date": str(r.date),
                "playoff": bool(r.playoff),
                "home": r.home_id,
                "away": r.away_id,
                "p_home": r.p_home,
                "novelty": r.novelty,
                "market": market.get(r.game_id),
                "elo": {"home": r.home_elo, "away": r.away_elo},
                "rest": {"home": r.rest_home, "away": r.rest_away},
                "points": {
                    side: {
                        "mid": team_pred(r.game_id, tid, "pts"),
                        "lo": team_pred(r.game_id, tid, "pts_lo"),
                        "hi": team_pred(r.game_id, tid, "pts_hi"),
                    }
                    for side, tid in (("home", r.home_id), ("away", r.away_id))
                },
                "players": {
                    "home": players_for(r.game_id, r.home_id),
                    "away": players_for(r.game_id, r.away_id),
                },
                "injuries": {
                    "home": inj_by_team.get(r.home_id, []),
                    "away": inj_by_team.get(r.away_id, []),
                },
            }
        )

    # Written by scripts/equine_stack_tip.py alongside the pyproject pin.
    stack = json.loads((root / "equine_stack.json").read_text())
    cfg = config or WinnerConfig()
    widths = [len(FEATURES), *cfg.hidden, cfg.emb_dim]
    embedding_desc = "MLP " + "→".join(map(str, widths))
    finals = games[games["status"] == "final"]
    payload = {
        "generated_at": stamp,
        "season": season_label(season),
        "data": {
            "first_season": season_label(FIRST_SEASON),
            "games": int(len(finals)),
            "player_games": int((~player_box["dnp"]).sum()),
            "through": str(finals["date"].max()),
            "source": "ESPN public JSON API",
            "injuries": injury_source,
        },
        "models": {
            "winner": {
                "library": "EQUINE",
                "version": version("equine"),
                "commit": stack["commit"],
                "branch": stack["branch"],
                "prs": stack["prs"],
                "head": "EquineGP: Laplace GP over random Fourier features",
                "embedding": embedding_desc,
                "features": FEATURES,
                "trained_on": int(len(train)),
                "epochs": wm.epochs_trained,
            },
            "forecast": {
                "library": "TimesFM",
                "version": version("timesfm"),
                "checkpoint": MODEL_ID,
                "xreg_mode": tfm.xreg_mode,
                "covariates": list(COVARIATES),
                "context": tfm.max_context,
                "fine_tune": fine_tune,
            },
        },
        "teams": teams,
        "games": out_games,
        "results": results,
        "record": record,
    }
    site_data = root / "site" / "data"
    site_data.mkdir(parents=True, exist_ok=True)
    (site_data / "latest.json").write_text(
        json.dumps(_clean(payload), separators=(",", ":"), allow_nan=False)
    )
    return {"upcoming": len(out_games), "record": record, "generated_at": stamp}


def write_backtest(root: Path, result: dict[str, Any]) -> None:
    site_data = root / "site" / "data"
    site_data.mkdir(parents=True, exist_ok=True)
    result = {
        "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **result,
    }
    (site_data / "backtest.json").write_text(
        json.dumps(_clean(result), indent=1, allow_nan=False) + "\n"
    )
