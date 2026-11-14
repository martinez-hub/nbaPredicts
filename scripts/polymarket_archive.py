"""Archive Polymarket's NBA game prices (1-minute) and trades, kept local.

The question these answer: after a player's status changes, how long does
the market take to price it? That needs prices with timestamps, and
Polymarket's are free from 2024-25. Its CLOB still serves 1-minute history
for settled markets, but its newer API keeps 1-minute data for only about a
week, so the history is archived while it is served.

Research only: nothing here trades. Everything stays in data_local/ (never
the repo or the site), requests go one at a time with a plain client name,
and a refusal stops the run rather than being worked around.

    uv run python scripts/polymarket_archive.py events   # find the games
    uv run python scripts/polymarket_archive.py prices   # 1-minute prices
    uv run python scripts/polymarket_archive.py trades   # moneyline trades

Each step skips what is already on disk, so a stopped run resumes.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from nbapredicts.collect import Store

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "polymarket"
GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
DATA = "https://data-api.polymarket.com"
NBA_SERIES = 10345  # Polymarket's NBA game series; it starts with 2025-26
FIRST_DAY = "2024-04-01"  # the first NBA game markets (2024 playoffs)
GAME_SLUG = re.compile(r"^nba-([a-z]{3})-([a-z]{3})-(\d{4}-\d{2}-\d{2})$")
PAUSE = 0.25  # seconds between requests: about 4 a second, far under the limits
USER_AGENT = "research-client/1.0"
TRADE_FIELDS = ["timestamp", "side", "outcome_index", "price", "size"]

log = logging.getLogger("polymarket_archive")


class Refused(RuntimeError):
    """The server refused us; the run stops instead of working around it."""


def request(session: requests.Session, method: str, url: str, **kwargs) -> object:
    for attempt in range(6):
        time.sleep(PAUSE)
        try:
            resp = session.request(method, url, timeout=60, **kwargs)
        except requests.RequestException as err:
            log.warning("%s: %s (retry %d)", url, err, attempt + 1)
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code in (401, 403, 451):
            raise Refused(f"{resp.status_code} from {resp.url}")
        if resp.status_code in (429, 503):
            wait = float(resp.headers.get("Retry-After") or 10 * (attempt + 1))
            log.warning("%s: %d, waiting %.0f s", url, resp.status_code, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"{url} kept failing")


def get(session: requests.Session, url: str, **params) -> object:
    return request(session, "GET", url, params=params)


def post(session: requests.Session, url: str, body: dict) -> dict:
    return request(session, "POST", url, json=body)


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(obj, f)
    tmp.rename(path)


def read_json(path: Path) -> object:
    with gzip.open(path, "rt") as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Events: one per game
# --------------------------------------------------------------------------


def series_events(session: requests.Session) -> list[dict]:
    """Settled game events in Polymarket's NBA series (2025-26 on)."""
    out, offset = [], 0
    while True:
        page = get(
            session,
            f"{GAMMA}/events",
            series_id=NBA_SERIES,
            closed="true",
            order="startDate",
            ascending="true",
            limit=100,
            offset=offset,
        )
        if not page:
            return out
        out += [e for e in page if GAME_SLUG.match(e.get("slug", ""))]
        offset += len(page)


def team_codes(events: list[dict], teams: dict) -> dict[str, str]:
    """Our team id -> Polymarket's slug code, learned from titled events
    ("Knicks vs. Celtics" is nba-nyk-bos-...: away first)."""
    by_short = {t["short"]: tid for tid, t in teams.items()}
    codes: dict[str, str] = {}
    for e in events:
        m = GAME_SLUG.match(e["slug"])
        names = [s.strip() for s in e.get("title", "").split(" vs. ")]
        if m and len(names) == 2 and all(n in by_short for n in names):
            codes.setdefault(by_short[names[0]], m.group(1))
            codes.setdefault(by_short[names[1]], m.group(2))
    return codes


def slug_candidates(away: str, home: str, game: pd.Series) -> list[str]:
    utc_day = game["start"].strftime("%Y-%m-%d")
    days = dict.fromkeys([str(game["date"]), utc_day])
    return [f"nba-{a}-{h}-{d}" for d in days for a, h in ((away, home), (home, away))]


def find_events(session: requests.Session) -> None:
    teams = json.loads((ROOT / "data" / "teams.json").read_text())
    events = series_events(session)
    log.info("series %d: %d settled game events", NBA_SERIES, len(events))
    codes = team_codes(events, teams)
    if len(codes) < 30:
        log.warning("learned codes for %d teams only: %s", len(codes), codes)
    for e in events:
        path = OUT / "events" / f"{e['slug']}.json.gz"
        if not path.exists():
            write_json(path, e)
    # Before the series existed: look each game up by its slug.
    games = Store(ROOT / "data").load("games")
    games = games[(games["status"] == "final") & (games["start"] >= FIRST_DAY)]
    known = {p.name.removesuffix(".json.gz") for p in (OUT / "events").glob("*.gz")}
    missing = []
    for _, g in games.sort_values("start").iterrows():
        away, home = codes.get(g["away_id"]), codes.get(g["home_id"])
        if not away or not home:
            continue
        cands = slug_candidates(away, home, g)
        if any(c in known for c in cands):
            continue
        found = None
        for slug in cands:
            hits = get(session, f"{GAMMA}/events", slug=slug)
            if hits:
                found = hits[0]
                break
        if found:
            write_json(OUT / "events" / f"{found['slug']}.json.gz", found)
            known.add(found["slug"])
        else:
            missing.append(g["game_id"])
    log.info("%d events on disk; %d games with no event", len(known), len(missing))
    (OUT / "missing_games.json").write_text(json.dumps(missing))


def markets(kinds: set[str] | None) -> list[tuple[str, dict]]:
    out = []
    for path in sorted((OUT / "events").glob("*.json.gz")):
        event = read_json(path)
        for m in event.get("markets", []):
            kind = m.get("sportsMarketType") or "moneyline"
            if kinds is None or kind in kinds:
                out.append((event["slug"], m))
    # Moneylines first: they are what the timing test needs.
    out.sort(key=lambda sm: (sm[1].get("sportsMarketType") != "moneyline", sm[0]))
    return out


# --------------------------------------------------------------------------
# Prices and trades
# --------------------------------------------------------------------------


def _time(text: str | None) -> datetime | None:
    if not text:
        return None
    return pd.Timestamp(text).tz_convert(UTC).to_pydatetime()


BATCH = 20  # tokens per batch request (the endpoint's limit)


def batch_history(session: requests.Session, tokens: list[str], start, end) -> dict:
    """1-minute points for up to 20 tokens over one window, a day per
    request; the same points as ``price_history`` token by token."""
    points: dict[str, dict[int, float]] = {t: {} for t in tokens}
    t = start
    while t < end:
        stop = min(t + timedelta(days=1), end)
        page = post(
            session,
            f"{CLOB}/batch-prices-history",
            {
                "markets": tokens,
                "start_ts": int(t.timestamp()),
                "end_ts": int(stop.timestamp()),
                "fidelity": 1,
            },
        )
        for token, hist in (page.get("history") or {}).items():
            for pt in hist:
                points.setdefault(token, {})[int(pt["t"])] = float(pt["p"])
        t = stop
    return {
        tok: [{"t": k, "p": pts[k]} for k in sorted(pts)] for tok, pts in points.items()
    }


def archive_prices(session: requests.Session, kinds: set[str] | None) -> None:
    """Every market of every game, 20 tokens per request over the game's
    window. Both sides of a moneyline (``p`` may not be complementary), one
    side elsewhere."""
    jobs: dict[str, list[tuple[Path, str, dict]]] = {}
    for slug, m in markets(kinds):
        tokens = json.loads(m.get("clobTokenIds") or "[]")
        if (m.get("sportsMarketType") or "moneyline") != "moneyline":
            tokens = tokens[:1]
        for side, token in enumerate(tokens):
            path = OUT / "prices" / slug / f"{m['id']}_{side}.json.gz"
            if not path.exists():
                jobs.setdefault(slug, []).append((path, token, m))
    log.info("%d tokens to fetch in %d games", sum(map(len, jobs.values())), len(jobs))
    for i, todo in enumerate(jobs.values()):
        for k in range(0, len(todo), BATCH):
            group = todo[k : k + BATCH]
            starts = [
                _time(m.get("startDate")) or _time(m.get("createdAt"))
                for *_, m in group
            ]
            ends = [
                _time(m.get("closedTime")) or _time(m.get("endDate")) for *_, m in group
            ]
            if not any(starts) or not any(ends):
                continue
            start = min(s for s in starts if s)
            end = max(e for e in ends if e) + timedelta(hours=1)
            got = batch_history(session, [tok for _, tok, _ in group], start, end)
            for path, token, m in group:
                hist = got.get(token, [])
                write_json(path, {"market": m["id"], "token": token, "history": hist})
        if i % 50 == 0:
            log.info("prices: %d of %d games", i, len(jobs))
    log.info("prices done")


def archive_trades(session: requests.Session, kinds: set[str] | None) -> None:
    """Every fill (taker side once), newest first as served. Only time, side,
    outcome, price and size are kept: no wallets or names."""
    todo = markets(kinds)
    log.info("%d markets", len(todo))
    for i, (slug, m) in enumerate(todo):
        path = OUT / "trades" / slug / f"{m['id']}.csv.gz"
        if path.exists() or not m.get("conditionId"):
            continue
        rows, cursor = [], None
        while True:
            params = {"cursor": cursor} if cursor else {"limit": 1000}
            page = get(
                session, f"{DATA}/v2/trades", condition=m["conditionId"], **params
            )
            rows += [{k: r.get(k) for k in TRADE_FIELDS} for r in page.get("data", [])]
            cursor = (page.get("pagination") or {}).get("next_cursor")
            if not cursor:
                break
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda r: r["timestamp"] or 0))
        tmp.rename(path)
        if i % 50 == 0:
            log.info("trades: %d of %d markets", i, len(todo))
    log.info("trades done")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["events", "prices", "trades"])
    parser.add_argument(
        "--kinds",
        nargs="*",
        default=None,
        help="market types (moneyline, spreads, totals); default all for "
        "prices, moneyline for trades",
    )
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUT / "archive.log"),
        ],
    )
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    kinds = set(args.kinds) if args.kinds else None
    try:
        if args.step == "events":
            find_events(session)
        elif args.step == "prices":
            archive_prices(session, kinds)
        else:
            archive_trades(session, kinds or {"moneyline"})
    except Refused as err:
        log.error("stopped: %s", err)
        sys.exit(2)


if __name__ == "__main__":
    main()
