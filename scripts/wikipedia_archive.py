"""Wikipedia for every NBA player: daily page views since 2015-07 (an
attention signal) and the article summary (CC BY-SA text, kept local).

    uv run python scripts/wikipedia_archive.py

Players and their English articles come from the Wikidata table that
scripts/history_archive.py saves.

- A new player is two requests to Wikimedia's REST APIs: the views since
  2015 and the summary.
- A player already on disk gets the days since its last saved one from the
  Action API, which gives the last 60 days for 50 titles a request (answered
  about 11 at a time). A gap longer than that goes back to the REST API.

Wikimedia lets a client that gives no contact details in its User-Agent
make about 10 requests a minute, so requests go one every 6.5 s and a 429
waits for its Retry-After. Wikimedia revises past days' counts now and then;
saved days are kept as they were fetched.
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote, urlencode

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "history" / "wikipedia"
PLAYERS = ROOT / "data_local" / "history" / "wikidata" / "players.json"
VIEWS = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{title}/daily/{start}/{end}"
)
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
API = "https://en.wikipedia.org/w/api.php"
FIRST_DAY = "20150701"  # the first day Wikimedia's API has
USER_AGENT = "research-client/1.0"
PAUSE = 6.5  # seconds between requests: under 10 a minute
TRIES = 6
BATCH = 50  # titles per Action API request (its limit without an account)
MAX_GIVE_UPS = 5  # in a row: the API is not answering, so stop

log = logging.getLogger("wikipedia_archive")


class GaveUp(Exception):
    """No answer after every try (throttled, or the server failing)."""


def retry_after(resp: requests.Response, attempt: int) -> float:
    try:
        return float(resp.headers["Retry-After"]) + 1
    except (KeyError, ValueError):
        return 10.0 * (attempt + 1)


def fetch(s: requests.Session, url: str) -> dict | None:
    """The JSON at url, or None when the API has nothing for it (404).
    Raises GaveUp when every try is throttled or fails."""
    for attempt in range(TRIES):
        time.sleep(PAUSE)
        try:
            resp = s.get(url, timeout=60)
        except requests.RequestException as err:  # timeouts happen; retry
            log.warning("%s: %s (try %d)", url, err, attempt + 1)
            time.sleep(30 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        if resp.status_code in (401, 403):
            log.error("refused (%d) at %s: stopping", resp.status_code, url)
            sys.exit(2)
        wait = retry_after(resp, attempt)
        log.warning(
            "%d at %s: waiting %.0fs (try %d)", resp.status_code, url, wait, attempt + 1
        )
        time.sleep(wait)
    raise GaveUp(url)


def batch_views(s: requests.Session, titles: list[str]) -> dict[str, dict[str, int]]:
    """The last 60 days of views per title, keyed by the title as sent and
    the day as YYYYMMDD; days without a count yet are left out."""
    params = {
        "action": "query",
        "prop": "pageviews",
        "titles": "|".join(titles),
        "format": "json",
        "formatversion": "2",
    }
    out: dict[str, dict[str, int]] = {}
    for _ in range(2 * len(titles)):  # each answer covers at least one title
        data = fetch(s, f"{API}?{urlencode(params)}") or {}
        if "error" in data:
            log.warning("pageviews: %s", data["error"].get("info"))
            break
        query = data.get("query", {})
        sent = {n["to"]: n["from"] for n in query.get("normalized", [])}
        for page in query.get("pages", []):
            days = {
                d.replace("-", ""): n
                for d, n in (page.get("pageviews") or {}).items()
                if n is not None
            }
            if days:
                out.setdefault(sent.get(page["title"], page["title"]), {}).update(days)
        if "continue" not in data:
            break
        params.update(data["continue"])
    return out


def views_of(page: dict | None) -> list[dict]:
    return [
        {"day": v["timestamp"][:8], "views": v["views"]}
        for v in (page or {}).get("items", [])
    ]


def load(dest: Path) -> dict:
    with gzip.open(dest, "rt") as f:
        return json.load(f)


def save(dest: Path, record: dict) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as f:
        json.dump(record, f)
    tmp.rename(dest)


def last_day(dest: Path) -> str | None:
    return max((v["day"] for v in load(dest)["views"]), default=None)


def next_day(day: str) -> str:
    return (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime("%Y%m%d")


def add_days(dest: Path, views: dict[str, int]) -> int:
    """Append the days after the last saved one; returns how many."""
    record = load(dest)
    last = max((v["day"] for v in record["views"]), default="")
    new = [{"day": d, "views": n} for d, n in sorted(views.items()) if d > last]
    if new:
        record["views"] += new
        save(dest, record)
    return len(new)


def top_up(s: requests.Session, dest: Path, title: str, end: str) -> int:
    """The REST API's days since the last saved one (for a gap the Action
    API's 60 days don't reach); returns how many were added."""
    last = last_day(dest)
    start = next_day(last) if last else FIRST_DAY
    if start > end:
        return 0
    page = fetch(s, VIEWS.format(title=title, start=start, end=end))
    return add_days(dest, {v["day"]: v["views"] for v in views_of(page)})


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    players = pd.DataFrame(json.loads(PLAYERS.read_text()))
    players["qid"] = players["p"].str.rsplit("/", n=1).str[1]
    articles = players.dropna(subset=["article"]).drop_duplicates("p")
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y%m%d")
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT

    due, new_players = [], []
    for row in articles.itertuples(index=False):
        dest = OUT / f"{row.qid}.json.gz"
        tail = unquote(row.article.rsplit("/wiki/", 1)[1])
        player = (row, dest, tail.replace("_", " "), quote(tail, safe=""))
        if not dest.exists():
            new_players.append(player)
        elif (last_day(dest) or "") < end:
            due.append(player)
    log.info(
        "%d players with articles: %d to top up, %d new",
        len(articles),
        len(due),
        len(new_players),
    )

    added = new = gave_up = in_a_row = 0

    def gave_up_on(what: str, err: GaveUp) -> None:
        # Nothing is written, so the next run asks again.
        nonlocal gave_up, in_a_row
        gave_up += 1
        in_a_row += 1
        log.warning("gave up on %s for now: %s", what, err)
        if in_a_row >= MAX_GIVE_UPS:
            log.error("%d requests in a row got no answer: stopping", in_a_row)
            sys.exit(1)

    for i in range(0, len(due), BATCH):
        chunk = due[i : i + BATCH]
        if i % (10 * BATCH) == 0:
            log.info("top-ups: %d of %d", i, len(due))
        try:
            got = batch_views(s, [p[2] for p in chunk])
        except GaveUp as err:
            gave_up_on(f"{len(chunk)} top-ups", err)
            continue
        in_a_row = 0
        for row, dest, title, rest_title in chunk:
            views = got.get(title, {})
            last = last_day(dest)
            if last is None or (views and min(views) > next_day(last)):
                try:  # the 60 days don't reach back to the last saved one
                    added += top_up(s, dest, rest_title, end)
                except GaveUp as err:
                    gave_up_on(row.qid, err)
                    continue
                in_a_row = 0
            else:
                added += add_days(dest, views)

    for i, (row, dest, _, rest_title) in enumerate(new_players):
        if i % 100 == 0:
            log.info("new players: %d of %d", i, len(new_players))
        try:
            views = fetch(s, VIEWS.format(title=rest_title, start=FIRST_DAY, end=end))
            summary = fetch(s, SUMMARY.format(title=rest_title)) or {}
        except GaveUp as err:
            gave_up_on(row.qid, err)
            continue
        in_a_row = 0
        save(
            dest,
            {
                "wikidata": row.qid,
                "article": row.article,
                "views": views_of(views),
                "summary": summary.get("extract"),
                "description": summary.get("description"),
            },
        )
        new += 1

    log.info(
        "done: %d days of views added, %d new players, %d gave up for now",
        added,
        new,
        gave_up,
    )


if __name__ == "__main__":
    main()
