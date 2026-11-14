"""Wikipedia for every NBA player: daily page views since 2015-07 (an
attention signal) and the article summary (CC BY-SA text, kept local).

    uv run python scripts/wikipedia_archive.py

Players and their English articles come from the Wikidata table that
scripts/history_archive.py saves. Each player is two requests (Wikimedia's
REST APIs), a few a second; players already on disk are skipped.
"""

from __future__ import annotations

import gzip
import json
import logging
import sys
import time
from pathlib import Path
from urllib.parse import quote, unquote

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "history" / "wikipedia"
PLAYERS = ROOT / "data_local" / "history" / "wikidata" / "players.json"
VIEWS = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia/all-access/user/{title}/daily/20150701/{end}"
)
SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
USER_AGENT = "research-client/1.0"
PAUSE = 0.3

log = logging.getLogger("wikipedia_archive")


def fetch(s: requests.Session, url: str) -> dict | None:
    for attempt in range(4):
        time.sleep(PAUSE)
        try:
            resp = s.get(url, timeout=60)
        except requests.RequestException as err:  # timeouts happen; retry
            log.warning("%s: %s (retry %d)", url, err, attempt + 1)
            time.sleep(30 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        if resp.status_code in (401, 403):
            log.error("refused (%d) at %s: stopping", resp.status_code, url)
            sys.exit(2)
        time.sleep(float(resp.headers.get("Retry-After") or 10 * (attempt + 1)))
    return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    players = pd.DataFrame(json.loads(PLAYERS.read_text()))
    articles = players.dropna(subset=["article"]).drop_duplicates("p")
    end = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y%m%d")
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    done = 0
    for _, row in articles.iterrows():
        qid = row["p"].rsplit("/", 1)[1]
        dest = OUT / f"{qid}.json.gz"
        if dest.exists():
            continue
        title = quote(unquote(row["article"].rsplit("/wiki/", 1)[1]), safe="")
        views = fetch(s, VIEWS.format(title=title, end=end))
        summary = fetch(s, SUMMARY.format(title=title))
        record = {
            "wikidata": qid,
            "article": row["article"],
            "views": [
                {"day": v["timestamp"][:8], "views": v["views"]}
                for v in (views or {}).get("items", [])
            ],
            "summary": (summary or {}).get("extract"),
            "description": (summary or {}).get("description"),
        }
        OUT.mkdir(parents=True, exist_ok=True)
        with gzip.open(dest, "wt") as f:
            json.dump(record, f)
        done += 1
        if done % 250 == 0:
            log.info("%d of %d players", done, len(articles))
    log.info("done: %d players saved this run, %d with articles", done, len(articles))


if __name__ == "__main__":
    main()
