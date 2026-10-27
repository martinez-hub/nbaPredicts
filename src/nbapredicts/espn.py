"""Client for ESPN's public NBA JSON endpoints.

Three endpoints are used, all keyless:

- ``teams``: the 30 franchises and their ESPN ids.
- ``teams/{id}/schedule``: one team's games for a season and season type. This
  is how the backfill enumerates games, because the scoreboard does not accept
  date ranges.
- ``summary?event={id}``: one game's header, team box score, player box score
  and injury report.

ESPN labels a season by the calendar year it ends in: ``season=2021`` is the
2020-21 season. The rest of the package keeps that convention and calls it
``season``.
"""

from __future__ import annotations

import gzip
import json
import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

# ESPN season types. Preseason (1) and the All-Star game are never fetched.
REGULAR, POSTSEASON, PLAY_IN = 2, 3, 5
SEASON_TYPES = (REGULAR, POSTSEASON, PLAY_IN)

# Only these parts of a summary are read; the rest (play-by-play, news, win
# probability) is ~90% of the payload and is dropped before caching.
_SUMMARY_KEEP = ("header", "boxscore", "gameInfo", "injuries")


class ESPNError(RuntimeError):
    """A request that still failed after every retry."""


class ESPNClient:
    """Thread-safe, retrying ESPN client with an optional on-disk cache.

    Only completed games are cached: a finished box score never changes, while
    a scheduled game's summary does.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        timeout: float = 20.0,
        max_retries: int = 6,
        min_interval: float = 0.05,
    ) -> None:
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_interval = min_interval
        self._local = threading.local()
        self._lock = threading.Lock()
        self._last_request = 0.0

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "User-Agent": "nbaPredicts/0.1 (+https://martinez-hub.github.io)",
                    "Accept": "application/json",
                }
            )
            self._local.session = session
        return session

    def _throttle(self) -> None:
        with self._lock:
            wait = self._last_request + self.min_interval - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        url = f"{BASE}/{path}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self._session().get(url, params=params, timeout=self.timeout)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    last_error = ESPNError(f"HTTP {resp.status_code} for {resp.url}")
                else:
                    raise ESPNError(f"HTTP {resp.status_code} for {resp.url}")
            except (requests.ConnectionError, requests.Timeout, ValueError) as err:
                last_error = err
            backoff = min(60.0, 2.0**attempt) + random.uniform(0, 1)
            log.warning(
                "retry %d for %s in %.1fs: %s", attempt + 1, url, backoff, last_error
            )
            time.sleep(backoff)
        raise ESPNError(f"giving up on {url} {params}: {last_error}")

    def teams(self) -> list[dict[str, Any]]:
        data = self.get("teams")
        return [t["team"] for t in data["sports"][0]["leagues"][0]["teams"]]

    def team_schedule(
        self, team_id: str, season: int, season_type: int
    ) -> list[dict[str, Any]]:
        data = self.get(
            f"teams/{team_id}/schedule", season=season, seasontype=season_type
        )
        return data.get("events", [])

    def scoreboard(self, yyyymmdd: str) -> list[dict[str, Any]]:
        return self.get("scoreboard", dates=yyyymmdd).get("events", [])

    def summary(self, event_id: str, use_cache: bool = True) -> dict[str, Any]:
        path = self._cache_path(event_id)
        if use_cache and path is not None and path.exists():
            with gzip.open(path, "rt") as f:
                return json.load(f)
        data = self.get("summary", event=event_id)
        trimmed = {k: data[k] for k in _SUMMARY_KEEP if k in data}
        if path is not None and _is_final(trimmed):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            with gzip.open(tmp, "wt") as f:
                json.dump(trimmed, f, separators=(",", ":"))
            tmp.replace(path)
        return trimmed

    def _cache_path(self, event_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / "summary" / event_id[-2:] / f"{event_id}.json.gz"


def _is_final(summary: dict[str, Any]) -> bool:
    try:
        status = summary["header"]["competitions"][0]["status"]["type"]
    except (KeyError, IndexError):
        return False
    return bool(status.get("completed")) and status.get("name") == "STATUS_FINAL"


def _american(odds: object) -> float | None:
    try:
        value = float(str(odds).replace("+", ""))
    except (TypeError, ValueError):
        return None
    return value if value <= -100 or value >= 100 else None


def implied_probability(american: float) -> float:
    """Win probability an American moneyline implies, margin included."""
    return 100 / (american + 100) if american > 0 else -american / (-american + 100)


def parse_odds(event: dict[str, Any]) -> dict[str, Any] | None:
    """The first sportsbook line on a scoreboard event, home-team perspective.

    ``p_home`` is the moneyline probability with the bookmaker's margin
    removed (the two implied probabilities rescaled to sum to one).
    """
    lines = event.get("competitions", [{}])[0].get("odds") or []
    if not lines:
        return None
    line = lines[0]
    ml = line.get("moneyline") or {}
    home = _american(((ml.get("home") or {}).get("close") or {}).get("odds"))
    away = _american(((ml.get("away") or {}).get("close") or {}).get("odds"))
    p_home = None
    if home is not None and away is not None:
        ph, pa = implied_probability(home), implied_probability(away)
        p_home = ph / (ph + pa)
    return {
        "game_id": str(event["id"]),
        "provider": (line.get("provider") or {}).get("name"),
        "spread": line.get("spread"),
        "total": line.get("overUnder"),
        "ml_home": home,
        "ml_away": away,
        "p_market": p_home,
    }
