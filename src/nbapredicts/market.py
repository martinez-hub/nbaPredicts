"""Historical and current betting lines as market win probabilities.

ESPN's core API keeps every sportsbook's line for each game back to 2013-14:
one moneyline per book through about 2020-21, separate opening and closing
lines from 2023-24 on. Per book, the two moneylines become a win probability
with the margin removed by the power method (raw implied probabilities
raised to the power k that makes them sum to one), which does not overstate
longshots the way proportional rescaling does. The game's market probability
is the median across books, skipping live-odds feeds.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from .espn import ESPNClient, ESPNError

log = logging.getLogger(__name__)

CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
ODDS_COLS = [
    "game_id", "season", "books", "p_close", "p_open", "spread_close", "spread_open",
]  # fmt: skip


def decimal_from_american(american: float) -> float:
    return 1 + (american / 100 if american > 0 else 100 / -american)


def devig_power(dec_home: float, dec_away: float) -> float:
    """Home win probability from two decimal odds, margin removed (power)."""
    a, b = 1 / dec_home, 1 / dec_away
    lo, hi = 0.5, 3.0
    for _ in range(60):
        k = (lo + hi) / 2
        if a**k + b**k > 1:
            lo = k
        else:
            hi = k
    k = (lo + hi) / 2
    return a**k / (a**k + b**k)


def _american(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("american", value.get("alternateDisplayValue"))
    try:
        number = float(str(value).replace("+", ""))
    except (TypeError, ValueError):
        return None
    return number if abs(number) >= 100 else None


def _number(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value", value.get("american"))
    try:
        return float(str(value).replace("+", ""))
    except (TypeError, ValueError):
        return None


def _line(side: dict[str, Any], phase: str | None) -> tuple[float | None, float | None]:
    """(american moneyline, home spread in points) at ``phase`` (open/close/None)."""
    block = side.get(phase) if phase else side
    if not isinstance(block, dict):
        return None, None
    return _american(block.get("moneyLine")), _number(block.get("pointSpread"))


def parse_core_odds(game_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    close_p, open_p, close_s, open_s = [], [], [], []
    for item in payload.get("items", []):
        name = (item.get("provider") or {}).get("name", "")
        if "live" in name.lower():
            continue
        home, away = item.get("homeTeamOdds") or {}, item.get("awayTeamOdds") or {}
        has_phases = isinstance(home.get("close"), dict)
        for phase, probs, spreads in (
            ("close" if has_phases else None, close_p, close_s),
            ("open", open_p, open_s),
        ):
            if phase == "open" and not isinstance(home.get("open"), dict):
                continue
            ml_h, sp = _line(home, phase)
            ml_a, _ = _line(away, phase)
            if ml_h is None or ml_a is None:
                continue
            probs.append(
                devig_power(decimal_from_american(ml_h), decimal_from_american(ml_a))
            )
            if sp is None and phase is None:  # older seasons: one line per book
                sp = _number(item.get("spread"))
            # Spreads move in half points; anything else is a price that
            # leaked into the field (seen in some books' opening lines).
            if sp is not None and abs(sp) <= 30 and float(sp * 2).is_integer():
                spreads.append(sp)
    if not close_p:
        return None
    median = lambda xs: float(np.median(xs)) if xs else None  # noqa: E731
    return {
        "game_id": game_id,
        "books": len(close_p),
        "p_close": median(close_p),
        "p_open": median(open_p),
        "spread_close": median(close_s),
        "spread_open": median(open_s),
    }


def fetch_odds(
    client: ESPNClient, game_ids: list[str], workers: int = 6
) -> pd.DataFrame:
    def one(gid: str) -> dict[str, Any] | None:
        try:
            payload = client.get_url(f"{CORE}/events/{gid}/competitions/{gid}/odds")
        except ESPNError as err:
            log.warning("odds for %s failed: %s", gid, err)
            return None
        return parse_core_odds(gid, payload)

    with ThreadPoolExecutor(workers) as pool:
        rows = [r for r in pool.map(one, game_ids) if r is not None]
    log.info("odds: %d of %d games have a line", len(rows), len(game_ids))
    return pd.DataFrame(rows, columns=[c for c in ODDS_COLS if c != "season"])
