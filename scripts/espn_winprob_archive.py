"""ESPN's per-play win probability, pre-game predictor and pick-center odds
for every final game since 2017-18 (the first season ESPN publishes win
probability for), kept local: ESPN's own model outputs are not republished.

    uv run python scripts/espn_winprob_archive.py [--first 2018]

The project's summary cache drops these parts to stay small, so each game's
summary is fetched once more, uncached; one file per season, resumable.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path

from nbapredicts.collect import Store
from nbapredicts.espn import ESPNClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data_local" / "espn_winprob"

log = logging.getLogger("espn_winprob_archive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=int, default=2018)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    games = Store(ROOT / "data").load("games")
    games = games[(games["status"] == "final") & (games["season"] >= args.first)]
    client = ESPNClient(cache_dir=None, min_interval=0.25)
    OUT.mkdir(parents=True, exist_ok=True)
    for season, rows in games.groupby("season"):
        path = OUT / f"{season}.jsonl.gz"
        done = set()
        if path.exists():
            with gzip.open(path, "rt") as f:
                done = {json.loads(line)["game_id"] for line in f}
        todo = [g for g in rows["game_id"] if g not in done]
        with gzip.open(path, "at") as f:
            for i, gid in enumerate(todo):
                try:
                    s = client.get("summary", event=gid)
                except Exception as err:
                    log.warning("%s: %s", gid, err)
                    continue
                record = {
                    "game_id": gid,
                    "winprobability": [
                        {
                            "play_id": w.get("playId"),
                            "home": w.get("homeWinPercentage"),
                            "tie": w.get("tiePercentage"),
                        }
                        for w in s.get("winprobability") or []
                    ],
                    "predictor": s.get("predictor"),
                    "pickcenter": s.get("pickcenter"),
                }
                f.write(json.dumps(record) + "\n")
                if (i + 1) % 250 == 0:
                    f.flush()
                    log.info("%s: %d of %d games", season, i + 1, len(todo))
        log.info("%s: done (%d new)", season, len(todo))


if __name__ == "__main__":
    main()
