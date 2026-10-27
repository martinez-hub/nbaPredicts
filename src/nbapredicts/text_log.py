"""Time-stamped pre-game text, logged at every run.

ESPN's injury notes and news are current-only (there is no archive), so the
only way to have a leakage-free history of what was known before each game,
for the text-reading model, is to record it as it appears. Each run appends
new injury notes and news items; nothing here feeds the models yet.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from .collect import Store
from .espn import ESPNClient

log = logging.getLogger(__name__)

_ATHLETE_ID = re.compile(r"/id/(\d+)")


def _athlete_id(athlete: dict[str, Any]) -> str | None:
    if athlete.get("id"):
        return str(athlete["id"])
    for link in athlete.get("links", []) or []:
        m = _ATHLETE_ID.search(link.get("href", ""))
        if m:
            return m.group(1)
    return None


def injury_notes(client: ESPNClient, fetched_at: str, season: int) -> pd.DataFrame:
    rows = []
    for team in client.get("injuries").get("injuries", []):
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete", {}) or {}
            details = inj.get("details", {}) or {}
            rows.append(
                {
                    "season": season,
                    "fetched_at": fetched_at,
                    "note_date": inj.get("date"),
                    "team_id": str(team.get("id")),
                    "athlete_id": _athlete_id(athlete),
                    "name": athlete.get("displayName"),
                    "status": inj.get("status"),
                    "fantasy": (details.get("fantasyStatus") or {}).get("abbreviation"),
                    "body": details.get("type"),
                    "detail": details.get("detail"),
                    "side": details.get("side"),
                    "return_date": details.get("returnDate"),
                    "short_comment": inj.get("shortComment"),
                    "long_comment": inj.get("longComment"),
                }
            )
    return pd.DataFrame(rows)


def news_items(client: ESPNClient, fetched_at: str, season: int) -> pd.DataFrame:
    rows = []
    for a in client.get("news", limit=50).get("articles", []):
        cats = a.get("categories", []) or []
        rows.append(
            {
                "season": season,
                "fetched_at": fetched_at,
                "news_id": str(a.get("id")),
                "published": a.get("published"),
                "headline": a.get("headline"),
                "description": a.get("description"),
                "team_ids": " ".join(str(c["teamId"]) for c in cats if c.get("teamId")),
                "athlete_ids": " ".join(
                    str(c["athleteId"]) for c in cats if c.get("athleteId")
                ),
            }
        )
    return pd.DataFrame(rows)


def _append(store: Store, name: str, new: pd.DataFrame, keys: list[str]) -> int:
    if new.empty:
        return 0
    old = store.load(name)
    if not old.empty:
        known = set(map(tuple, old[keys].astype(str).to_numpy()))
        fresh = ~new[keys].astype(str).apply(tuple, axis=1).isin(known)
        new = new[fresh]
        if new.empty:
            return 0
        new = pd.concat([old, new], ignore_index=True)
    store.save(name, new)
    return int(len(new) - len(old))


def log_text(store: Store, client: ESPNClient, now: pd.Timestamp, season: int) -> dict:
    """Append new injury notes and news; never raises."""
    stamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    added = {}
    for name, fetch, keys in (
        ("text_injuries", injury_notes, ["name", "note_date", "short_comment"]),
        ("text_news", news_items, ["news_id"]),
    ):
        try:
            added[name] = _append(store, name, fetch(client, stamp, season), keys)
        except Exception as err:  # text logging must never stop a run
            log.warning("%s logging failed: %s", name, err)
            added[name] = 0
    log.info("text log: %s", added)
    return added
