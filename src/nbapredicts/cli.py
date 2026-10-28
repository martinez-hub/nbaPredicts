"""Command line entry point: ``nbapredicts <command>``."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .collect import FIRST_SEASON, Store, current_season, update
from .espn import ESPNClient

ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="nbapredicts")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=6)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("collect", help="fetch schedules and missing box scores")
    p.add_argument("--first", type=int, default=FIRST_SEASON)
    p.add_argument("--last", type=int, default=None)
    p.add_argument(
        "--recent", action="store_true", help="only the current season (daily job)"
    )

    p = sub.add_parser("daily", help="collect, retrain, forecast and publish")
    p.add_argument("--no-collect", action="store_true")
    p.add_argument("--no-finetune", action="store_true", help="use TimesFM as released")

    p = sub.add_parser("backtest", help="walk-forward evaluation of both models")
    p.add_argument("--seasons", type=int, nargs="+", default=None)
    p.add_argument("--player-sample", type=int, default=3000)
    p.add_argument(
        "--winner-only",
        action="store_true",
        help="rerun the winner backtest, keep the published TimesFM results",
    )

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root: Path = args.root
    store = Store(root / "data")
    client = ESPNClient(cache_dir=args.cache or root / "cache.nosync")

    if args.command == "collect":
        last = args.last or current_season()
        first = last if args.recent else args.first
        result = update(store, client, range(first, last + 1), workers=args.workers)
        print(json.dumps(result))

    elif args.command == "daily":
        from .pipeline import daily

        result = daily(
            root,
            store,
            client,
            collect=not args.no_collect,
            finetune=not args.no_finetune,
        )
        print(json.dumps(result, default=str))

    elif args.command == "backtest":
        import pandas as pd

        from .backtest import backtest_forecasts, backtest_winner
        from .features import (
            apply_pregame_availability,
            build_features,
            player_values,
            pregame_availability,
            team_state,
        )
        from .finetune import finetune_point_head, training_series
        from .forecast import TimesFM, player_frame, team_frame
        from .pipeline import write_backtest

        games = store.load("games")
        team_box, player_box = store.load("team_box"), store.load("player_box")
        finished = sorted(games.loc[games["status"] == "final", "season"].unique())
        # The last three complete seasons by default (the current one is partial).
        seasons = args.seasons or [int(s) for s in finished[-4:-1]]
        pv = player_values(player_box, games)
        state = team_state(games, team_box, pv, [])
        feats = build_features(games, state, [])
        teams = json.loads((root / "data" / "teams.json").read_text())
        feats = apply_pregame_availability(
            feats,
            pregame_availability(
                pv,
                games,
                store.load("injury_reports"),
                {t["name"]: tid for tid, t in teams.items()},
            ),
        )
        winner = backtest_winner(feats, seasons, odds=store.load("odds"))
        if args.winner_only:
            previous = json.loads((root / "site/data/backtest.json").read_text())
            kept = {
                k: v for k, v in previous.items() if k not in ("winner", "generated_at")
            }
            write_backtest(root, {"winner": winner, **kept})
            print(json.dumps({"winner": winner["seasons"]}))
            return
        write_backtest(root, {"winner": winner})  # keep it if TimesFM fails
        tfm = TimesFM(quantile_head=False)
        tf = team_frame(state)
        pf = player_frame(pv, tf)
        season = seasons[-1]
        zero_shot = backtest_forecasts(tfm, tf, pf, season, sample=args.player_sample)
        # Fine-tune once on everything before the test season, then forecast
        # the whole season with that head: stricter than the daily job, which
        # refreshes the head every morning.
        start = tf.loc[tf["season"] == season, "start"].min()
        series, ends, limits = training_series(
            tf, pf, start, start - pd.Timedelta(days=3 * 365)
        )
        tune = finetune_point_head(tfm.model.model, series, ends, limits)
        tuned = backtest_forecasts(tfm, tf, pf, season, sample=args.player_sample)
        write_backtest(
            root,
            {
                "winner": winner,
                "forecast": tuned,
                "forecast_zero_shot": zero_shot,
                "fine_tune": tune,
            },
        )
        print(json.dumps({"winner": winner["seasons"], "forecast": tuned}))


if __name__ == "__main__":
    main()
