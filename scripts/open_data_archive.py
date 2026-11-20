"""Openly licensed NBA datasets from Hugging Face and GitHub, kept local.

    uv run python scripts/open_data_archive.py

Run again (the weekly refresh does), a Hugging Face dataset is downloaded
again only when its revision changed, and each repository pulls its latest
commit.

Hugging Face datasets (licence checked on each card, files from /resolve):
- minmaxxxer: closing lines and results 2024-25, 2025-26 and 2026-27 odds
  (CC BY 4.0)
- cbratkovics/nba-game-logs: player game logs 2021-22 to 2025-26 (CC0)

GitHub repositories (shallow clones):
- shufinskiy/nba_data: shot detail and play-by-play from 1996-97, pbpstats
  possessions 2000-01 to 2024-25, the NBA's live play-by-play with x/y from
  2020-21, matchups (Apache-2.0 repository over NBA data)
- nsandholtz/nba_replay: one 2012-13 game of STATS tracking, released with
  STATS's consent (MIT code)
- dcervone/EPVDemo: one game of STATS tracking from the expected-possession-
  value paper (no licence file; local research only)

Skipped on purpose: a Polymarket order-book capture whose licence states no
terms, and a 22.6 GB Hugging Face archive that repeats shufinskiy's files.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "open"
USER_AGENT = "research-client/1.0"
HF_DATASETS = {
    "minmaxxxer/nba-closing-lines-ats-2024-25": "cc-by-4.0",
    "minmaxxxer/nba-closing-lines-ats-2025-26": "cc-by-4.0",
    "minmaxxxer/nba-odds-2026-27": "cc-by-4.0",
    "cbratkovics/nba-game-logs": "cc0-1.0",
}
REPOS = ("shufinskiy/nba_data", "nsandholtz/nba_replay", "dcervone/EPVDemo")

log = logging.getLogger("open_data_archive")


def hf_dataset(s: requests.Session, name: str, licence: str) -> None:
    info = s.get(f"https://huggingface.co/api/datasets/{name}", timeout=60).json()
    card = (info.get("cardData") or {}).get("license")
    if card != licence:
        log.warning("%s: licence is now %r, not %r: skipped", name, card, licence)
        return
    files = [x["rfilename"] for x in info.get("siblings", [])]
    folder = OUT / "huggingface" / name.replace("/", "__")
    revision = info["sha"]
    stamp = folder / ".revision"
    if stamp.exists() and stamp.read_text().strip() == revision:
        log.info("%s: unchanged", name)
        return
    for f in files:
        if f == ".gitattributes":
            continue
        dest = folder / f
        resp = s.get(
            f"https://huggingface.co/datasets/{name}/resolve/{revision}/{f}",
            timeout=300,
        )
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.rename(dest)
        time.sleep(0.2)
    stamp.write_text(revision)
    log.info("%s: %d files at revision %s", name, len(files), revision[:8])


def repo(name: str) -> None:
    dest = OUT / "github" / name.replace("/", "__")
    if dest.exists():
        done = subprocess.run(
            ["git", "-C", str(dest), "pull", "-q", "--ff-only"],
            capture_output=True,
            text=True,
        )
        if done.returncode:
            log.warning("%s: pull failed: %s", name, done.stderr.strip())
        else:
            log.info("%s: up to date", name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{name}.git"
    subprocess.run(["git", "clone", "-q", "--depth", "1", url, str(dest)], check=True)
    log.info("%s: cloned", name)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    for name, licence in HF_DATASETS.items():
        hf_dataset(s, name, licence)
    for name in REPOS:
        repo(name)


if __name__ == "__main__":
    main()
