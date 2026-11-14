"""Keep local copies of sportsdataverse's NBA data releases.

sportsdataverse publishes free parquet files of ESPN and NBA-stats data on
GitHub releases (play-by-play, shots, lineups, possessions, every version of
the official injury report since 2018-12, ...). The project reorganised its
repos in 2026, so the files the project relies on are pinned here. They stay
in data_local/ (NBA and ESPN data, not ours to republish).

    uv run python scripts/sportsdataverse_pin.py            # every NBA tag
    uv run python scripts/sportsdataverse_pin.py nba_stats_injury_report

A file is downloaded again only when the release's copy changed (size or
update time), and manifest.json records what is on disk.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "sportsdataverse"
RELEASES = "https://api.github.com/repos/sportsdataverse/sportsdataverse-data/releases"
USER_AGENT = "research-client/1.0"

log = logging.getLogger("sportsdataverse_pin")


def nba_releases(session: requests.Session) -> list[dict]:
    out, page = [], 1
    while True:
        resp = session.get(RELEASES, params={"per_page": 100, "page": page}, timeout=60)
        resp.raise_for_status()
        rows = resp.json()
        if not rows:
            return out
        out += [
            r
            for r in rows
            if "nba" in r["tag_name"] and not r["tag_name"].startswith("wnba")
        ]
        page += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tags", nargs="*", help="release tags (default: every NBA tag)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    manifest_path = OUT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    releases = nba_releases(session)
    if args.tags:
        releases = [r for r in releases if r["tag_name"] in args.tags]
    fetched = 0
    for rel in releases:
        tag = rel["tag_name"]
        for asset in rel["assets"]:
            name = asset["name"]
            if not name.endswith((".parquet", ".json", ".txt")):
                continue  # the same tables also come as csv and rds
            key = f"{tag}/{name}"
            stamp = {"size": asset["size"], "updated_at": asset["updated_at"]}
            path = OUT / tag / name
            if path.exists() and manifest.get(key) == stamp:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with session.get(
                asset["browser_download_url"], stream=True, timeout=300
            ) as r:
                r.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".tmp")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.rename(path)
            manifest[key] = stamp
            fetched += 1
            time.sleep(0.2)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
        log.info(
            "%s: %d files on disk", tag, sum(k.startswith(tag + "/") for k in manifest)
        )
    log.info("fetched %d files from %d releases", fetched, len(releases))


if __name__ == "__main__":
    main()
