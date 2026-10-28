"""Rebuild data/odds for every finished game from ESPN's core odds API."""

import logging
from pathlib import Path

from nbapredicts.collect import Store
from nbapredicts.espn import ESPNClient
from nbapredicts.market import fetch_odds

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    store = Store(ROOT / "data")
    games = store.load("games")
    finals = games[games["status"] == "final"]
    odds = fetch_odds(ESPNClient(), finals["game_id"].tolist(), workers=8)
    odds["season"] = odds["game_id"].map(finals.set_index("game_id")["season"])
    store.save("odds", odds)
    print(
        odds.groupby("season")
        .agg(
            n=("game_id", "size"),
            books=("books", "median"),
            open=("p_open", lambda x: x.notna().mean()),
            models=("p_models", lambda x: x.notna().mean()),
        )
        .round(2)
        .to_string()
    )
