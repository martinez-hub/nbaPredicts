"""What was known when: a logger for feeds that keep only their current state.

The hypothesis the project is testing is that any edge over the betting
market sits between news breaking and prices moving. Testing it needs, with
timestamps: who is starting and who is out (the NBA's daily lineups file),
what reporters and news feeds said and when (Bluesky, RSS, ESPN), and what
the prices were (Polymarket's order book, ESPN's current odds). None of these
keep a history, so the Mac records them as they change, every two minutes.

What can be fetched again later (the 15-minute injury-report PDFs, settled
Polymarket history) is not logged here. Everything stays local: third-party
data, never committed. A host that refuses a plain client is skipped from
then on, never worked around.

launchd runs ``nbapredicts --root ~/.local/share/nbapredicts live-log`` from
the internal disk (launchd jobs may not read the external SSD), and
``nbapredicts live-sweep`` moves finished days to the SSD archive.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import logging
import re
import shutil
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests

from .collect import Store
from .espn import ESPNClient
from .text_log import log_text

log = logging.getLogger(__name__)

USER_AGENT = "research-client/1.0"
MIN_FREE_BYTES = 5 * 1024**3  # stop writing below 5 GB free on the disk
EASTERN = "America/New_York"

LINEUPS = "https://stats.nba.com/js/data/leaders/00_daily_lineups_{day}.json"
BLUESKY = "https://public.api.bsky.app/xrpc/app.bsky.feed.getAuthorFeed"
# Active reporters, plus shamsbot (an unofficial relay of Shams Charania's
# posts), which the user chose to include on 2026-11-14.
BLUESKY_ACCOUNTS = (
    "shamsbot.bsky.social",
    "thesteinline.bsky.social",
    "jakelfischer.bsky.social",
    "mikeascotto.bsky.social",
    "howardbeck.bsky.social",
    "anthonyvslater.bsky.social",
    "hoopsrumors.bsky.social",
)
RSS_FEEDS = {
    "rotowire": "https://www.rotowire.com/rss/news.php?sport=NBA",
    "cbs": "https://www.cbssports.com/rss/headlines/nba/",
    "google_injury": "https://news.google.com/rss/search?q=NBA+injury+when:1d"
    "&hl=en-US&gl=US&ceid=US:en",
}
GAMMA_EVENTS = "https://gamma-api.polymarket.com/events"
CLOB_BOOKS = "https://clob.polymarket.com/books"
NBA_SERIES = 10345
ESPN_CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"

# Minutes between runs of each task (launchd starts a pass every 2 minutes).
CADENCE = {
    "polymarket_games": 60,  # before "polymarket", which reads what it finds
    "polymarket": 2,
    "lineups": 2,
    "bluesky": 2,
    "rss": 2,
    "espn_odds": 10,
    "espn_text": 10,
    "espn_feeds": 60,
}


class Refused(RuntimeError):
    """A host refused a plain client; it is skipped from then on."""


# --------------------------------------------------------------------------
# Small pieces (tested)
# --------------------------------------------------------------------------


def due(state: dict, task: str, now: pd.Timestamp, minutes: int) -> bool:
    last = state.get("last_run", {}).get(task)
    if last is None:
        return True
    # launchd's interval drifts by seconds; allow a little early.
    return now - pd.Timestamp(last) >= pd.Timedelta(minutes=minutes, seconds=-15)


def mark(state: dict, task: str, now: pd.Timestamp) -> None:
    state.setdefault("last_run", {})[task] = now.isoformat()


def rss_items(text: str) -> list[dict]:
    items = []
    for item in ET.fromstring(text).iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        guid = (item.findtext("guid") or "").strip()
        items.append(
            {
                "key": guid or link,
                "title": title,
                "link": link,
                "published": (item.findtext("pubDate") or "").strip(),
            }
        )
    return items


def bluesky_posts(feed: dict, handle: str) -> list[dict]:
    posts = []
    for entry in feed.get("feed", []):
        post = entry.get("post", {})
        record = post.get("record", {})
        reason = (entry.get("reason") or {}).get("$type", "")
        posts.append(
            {
                "key": post.get("uri"),
                "handle": handle,
                "created_at": record.get("createdAt"),
                "indexed_at": post.get("indexedAt"),
                "repost": reason.endswith("reasonRepost"),
                "text": record.get("text", ""),
            }
        )
    return posts


def append_new(path: Path, rows: list[dict], first_seen: str) -> int:
    """Append rows whose ``key`` is not in the file yet, stamped with when
    they were first seen."""
    known: set[str] = set()
    if path.exists():
        with open(path, newline="") as f:
            known = {r["key"] for r in csv.DictReader(f)}
    fresh = [r for r in rows if r.get("key") and r["key"] not in known]
    if not fresh:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["first_seen", *fresh[0].keys()]
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new_file:
            w.writeheader()
        for r in fresh:
            w.writerow({"first_seen": first_seen, **r})
            known.add(r["key"])
    return len(fresh)


def save_if_changed(folder: Path, name: str, text: str, state: dict, key: str) -> bool:
    digest = hashlib.sha256(text.encode()).hexdigest()
    hashes = state.setdefault("hashes", {})
    if hashes.get(key) == digest:
        return False
    folder.mkdir(parents=True, exist_ok=True)
    with gzip.open(folder / f"{name}.json.gz", "wt") as f:
        f.write(text)
    hashes[key] = digest
    return True


def book_rows(books: list[dict], meta: dict, seen_at: str) -> list[dict]:
    rows = []
    for b in books:
        token = b.get("asset_id")
        if token not in meta:
            continue
        slug, market, outcome = meta[token]
        bids = sorted(b.get("bids") or [], key=lambda x: -float(x["price"]))
        asks = sorted(b.get("asks") or [], key=lambda x: float(x["price"]))
        last = b.get("last_trade_price")
        rows.append(
            {
                "seen_at": seen_at,
                "book_ts": b.get("timestamp"),
                "slug": slug,
                "market": market,
                "outcome": outcome,
                "best_bid": float(bids[0]["price"]) if bids else None,
                "bid_size": float(bids[0]["size"]) if bids else None,
                "best_ask": float(asks[0]["price"]) if asks else None,
                "ask_size": float(asks[0]["size"]) if asks else None,
                "last_trade": float(last) if last not in (None, "") else None,
            }
        )
    return rows


_DAY = re.compile(r"(\d{4}-\d{2}-\d{2})")
_MONTH = re.compile(r"(?<!\d)(\d{4}-\d{2})(?!-\d)")


def sweep(staging: Path, archive: Path, today: date) -> int:
    """Move finished days (and months) from the internal-disk staging folder
    to the SSD archive; today's files and the state stay."""
    moved = 0
    month = today.strftime("%Y-%m")
    for path in sorted(p for p in staging.rglob("*") if p.is_file()):
        rel = path.relative_to(staging)
        text = str(rel)
        day, mon = _DAY.search(text), _MONTH.search(text)
        if day:
            done = date.fromisoformat(day.group(1)) < today
        elif mon:
            done = mon.group(1) < month
        else:
            continue
        if done:
            target = archive / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(path, target)
            moved += 1
    return moved


def _moneyline(odds: dict, side: str) -> str | None:
    """ESPN's current moneyline for one side ("close" is the latest line)."""
    line = ((odds.get("moneyline") or {}).get(side) or {}).get("close") or {}
    return line.get("odds")


def _append_rows(path: Path, rows: list[dict]) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new_file:
            w.writeheader()
        w.writerows(rows)
    return len(rows)


# --------------------------------------------------------------------------
# The logger
# --------------------------------------------------------------------------


class Logger:
    def __init__(self, root: Path, now: pd.Timestamp | None = None) -> None:
        self.root = root
        self.out = root / "data_local" / "live"
        self.now = now or pd.Timestamp.now(tz="UTC")
        self.stamp = self.now.strftime("%Y-%m-%dT%H:%M:%SZ")
        self.today = self.now.tz_convert(EASTERN).date()
        self.state_path = self.out / "state.json"
        self.state = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else {}
        )
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.espn = ESPNClient(cache_dir=None, max_retries=3)

    def get(self, url: str, **kwargs) -> requests.Response:
        host = urlparse(url).netloc
        if host in self.state.get("refused", {}):
            raise Refused(host)
        resp = self.session.get(url, timeout=30, **kwargs)
        if resp.status_code in (401, 403, 451):
            self.state.setdefault("refused", {})[host] = self.stamp
            log.error(
                "%s refused a plain client (%d); skipping it", host, resp.status_code
            )
            raise Refused(host)
        return resp

    def post(self, url: str, body: Any) -> requests.Response:
        host = urlparse(url).netloc
        if host in self.state.get("refused", {}):
            raise Refused(host)
        resp = self.session.post(url, json=body, timeout=30)
        if resp.status_code in (401, 403, 451):
            self.state.setdefault("refused", {})[host] = self.stamp
            raise Refused(host)
        return resp

    # Tasks ---------------------------------------------------------------

    def lineups(self) -> int:
        resp = self.get(
            LINEUPS.format(day=self.today.strftime("%Y%m%d")), allow_redirects=False
        )
        if resp.status_code != 200:  # not posted yet (a redirect) or no games
            return 0
        folder = self.out / "lineups" / str(self.today)
        name = self.now.strftime("%H%M%S")
        return int(save_if_changed(folder, name, resp.text, self.state, "lineups"))

    def bluesky(self) -> int:
        added = 0
        path = self.out / "bluesky" / f"{self.now:%Y-%m}.csv"
        for handle in BLUESKY_ACCOUNTS:
            resp = self.get(
                BLUESKY,
                params={"actor": handle, "limit": 30, "filter": "posts_no_replies"},
            )
            if resp.status_code == 200:
                added += append_new(
                    path, bluesky_posts(resp.json(), handle), self.stamp
                )
        return added

    def rss(self) -> int:
        added = 0
        path = self.out / "rss" / f"{self.now:%Y-%m}.csv"
        for name, url in RSS_FEEDS.items():
            try:
                resp = self.get(url)
                if resp.status_code != 200:
                    continue
                rows = [{**item, "feed": name} for item in rss_items(resp.text)]
                rows = [{**r, "key": f"{name}:{r['key']}"} for r in rows]
                added += append_new(path, rows, self.stamp)
            except Refused:
                continue
            except ET.ParseError as err:
                log.warning("%s: unreadable feed: %s", name, err)
        return added

    def polymarket_games(self) -> int:
        """Moneyline tokens for today's and tomorrow's games, refreshed hourly."""
        events, offset = [], 0
        while True:  # every open game event, a page at a time
            resp = self.get(
                GAMMA_EVENTS,
                params={
                    "series_id": NBA_SERIES,
                    "closed": "false",
                    "limit": 100,
                    "offset": offset,
                },
            )
            resp.raise_for_status()
            page = resp.json()
            if not page:
                break
            events += page
            offset += len(page)
        days = {str(self.today), str(self.today + pd.Timedelta(days=1))}
        tokens = {}
        for e in events:
            if e.get("slug", "")[-10:] not in days:
                continue
            for m in e.get("markets", []):
                if m.get("sportsMarketType") != "moneyline":
                    continue
                outcomes = json.loads(m.get("outcomes") or "[]")
                for tok, outcome in zip(
                    json.loads(m.get("clobTokenIds") or "[]"), outcomes, strict=False
                ):
                    tokens[tok] = [e["slug"], m["id"], outcome]
        self.state["polymarket_tokens"] = tokens
        return len(tokens)

    def polymarket(self) -> int:
        meta = {k: tuple(v) for k, v in self.state.get("polymarket_tokens", {}).items()}
        if not meta:
            return 0
        rows = []
        tokens = list(meta)
        for k in range(0, len(tokens), 20):
            resp = self.post(CLOB_BOOKS, [{"token_id": t} for t in tokens[k : k + 20]])
            if resp.status_code == 200:
                rows += book_rows(resp.json(), meta, self.stamp)
        return _append_rows(self.out / "polymarket" / f"{self.today}.csv", rows)

    def espn_odds(self) -> int:
        rows = []
        for day in (self.today, self.today + pd.Timedelta(days=1)):
            for ev in self.espn.scoreboard(day.strftime("%Y%m%d")):
                comp = (ev.get("competitions") or [{}])[0]
                for o in comp.get("odds") or []:
                    rows.append(
                        {
                            "seen_at": self.stamp,
                            "game_id": ev.get("id"),
                            "provider": (o.get("provider") or {}).get("name"),
                            "details": o.get("details"),
                            "spread": o.get("spread"),
                            "over_under": o.get("overUnder"),
                            "home_ml": _moneyline(o, "home"),
                            "away_ml": _moneyline(o, "away"),
                        }
                    )
        return _append_rows(self.out / "espn_odds" / f"{self.today}.csv", rows)

    def espn_text(self) -> int:
        from .collect import current_season

        season = current_season(self.now.to_pydatetime())
        added = log_text(Store(self.root / "data_local"), self.espn, self.now, season)
        return sum(added.values())

    def espn_feeds(self) -> int:
        """ESPN's own current-only numbers: its power index, each of today's
        games' predictor, and futures. Saved when they change."""
        from .collect import current_season

        season = current_season(self.now.to_pydatetime())
        saved = 0
        name = self.now.strftime("%H%M%S")
        feeds = {
            "bpi": f"{ESPN_CORE}/seasons/{season}/powerindex?limit=50",
            "futures": f"{ESPN_CORE}/seasons/{season}/futures",
        }
        for ev in self.espn.scoreboard(self.today.strftime("%Y%m%d")):
            gid = ev.get("id")
            feeds[f"predictor/{gid}"] = (
                f"{ESPN_CORE}/events/{gid}/competitions/{gid}/predictor"
            )
        for feed, url in feeds.items():
            try:
                data = self.espn.get_url(url)
            except Exception as err:  # a missing predictor is normal
                log.info("%s: %s", feed, err)
                continue
            folder = self.out / "espn" / feed / str(self.today)
            text = json.dumps(data, sort_keys=True)
            saved += save_if_changed(folder, name, text, self.state, f"espn/{feed}")
        return saved

    # Driver --------------------------------------------------------------

    def run(self) -> dict:
        self.out.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(self.out).free < MIN_FREE_BYTES:
            log.error("under 5 GB free on the disk: nothing logged (run live-sweep)")
            return {"skipped": "low disk"}
        result = {}
        for task, minutes in CADENCE.items():
            if not due(self.state, task, self.now, minutes):
                continue
            try:
                result[task] = getattr(self, task)()
            except Refused as host:
                result[task] = f"refused by {host}"
            except Exception as err:  # one feed failing must not stop the rest
                log.warning("%s failed: %s", task, err)
                result[task] = f"failed: {err}"
            mark(self.state, task, self.now)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.rename(self.state_path)
        return result


def run(root: Path) -> dict:
    return Logger(root).run()
