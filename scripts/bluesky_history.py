"""Every post the live logger's Bluesky accounts have made, from the day each
joined (the logger itself only sees new posts from 2026-11-14 on).

    uv run python scripts/bluesky_history.py

Pages each account's feed through Bluesky's public API (no replies; reposts
kept and marked) and writes data_local/history/bluesky/<handle>.csv.gz with
the logger's columns plus reposted_at (a repost's own time; its created_at and
indexed_at are the original post's). created_at is the time the posting
client gives; indexed_at is when Bluesky received the post. A run fetches
everything again and replaces the files, so it can be repeated to fill a gap
in the logger.
"""

from __future__ import annotations

import csv
import gzip
import logging
import sys
import time
from pathlib import Path

import requests

from nbapredicts.live import BLUESKY, BLUESKY_ACCOUNTS, USER_AGENT, bluesky_posts

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "history" / "bluesky"
PAUSE = 0.5  # seconds between requests; the public API allows far more
PAGE = 100  # the most posts a request returns

log = logging.getLogger("bluesky_history")


def get_page(s: requests.Session, handle: str, cursor: str | None) -> dict:
    params = {"actor": handle, "limit": PAGE, "filter": "posts_no_replies"}
    if cursor:
        params["cursor"] = cursor
    for attempt in range(5):
        time.sleep(PAUSE)
        try:
            resp = s.get(BLUESKY, params=params, timeout=60)
        except requests.RequestException as err:
            log.warning("%s: %s (try %d)", handle, err, attempt + 1)
            time.sleep(10 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in (401, 403):
            log.error("refused (%d) for %s: stopping", resp.status_code, handle)
            sys.exit(2)
        wait = float(resp.headers.get("Retry-After") or 10 * (attempt + 1))
        log.warning("%s: %d, waiting %.0fs", handle, resp.status_code, wait)
        time.sleep(wait)
    raise RuntimeError(f"{handle}: no answer after 5 tries")


def account_history(s: requests.Session, handle: str) -> list[dict]:
    posts: dict[str, dict] = {}
    cursor = None
    while True:
        page = get_page(s, handle, cursor)
        entries = page.get("feed", [])
        for entry, post in zip(entries, bluesky_posts(page, handle), strict=True):
            post["reposted_at"] = (entry.get("reason") or {}).get("indexedAt")
            posts.setdefault(post["key"], post)
        cursor = page.get("cursor")
        if not cursor or not page.get("feed"):
            return sorted(posts.values(), key=lambda p: p["indexed_at"] or "")


def save(dest: Path, rows: list[dict]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with gzip.open(tmp, "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.rename(dest)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    for handle in BLUESKY_ACCOUNTS:
        rows = account_history(s, handle)
        if not rows:
            log.warning("%s: no posts", handle)
            continue
        save(OUT / f"{handle}.csv.gz", rows)
        log.info(
            "%s: %d posts, %s to %s",
            handle,
            len(rows),
            rows[0]["indexed_at"][:10],
            rows[-1]["indexed_at"][:10],
        )


if __name__ == "__main__":
    main()
