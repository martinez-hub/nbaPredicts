# nbaPredicts

Daily NBA predictions for the [NBA tab](https://martinez-hub.github.io/nba.html)
of martinez-hub.github.io:

- **Who wins**: an MLP embedding with EQUINE's Gaussian-process head
  (`EquineGP`) gives a calibrated home-win probability and a measure of how
  unfamiliar the matchup is to the model.
- **By how much, and who scores**: Google's TimesFM 2.5 forecasts each team's
  points and each rotation player's points, rebounds, assists and threes, with
  an in-context regression on home/away, opponent defense and rest (XReg).

A GitHub Actions job runs twice a day, collects the night's box scores,
retrains, forecasts the next three days of games, commits the data and
redeploys `site/` to GitHub Pages, where the website reads
`/nbaPredicts/data/latest.json`.

## EQUINE version

The package is built from the tip of the stack of open pull requests on
[mit-ll-responsible-ai/equine](https://github.com/mit-ll-responsible-ai/equine),
not from the last release. The open PRs #236 → #263 form a single linear
stack on `main`, so the tip (`maint/python-311-syntax`) contains all of them.
`equine_stack.json` records the PRs and commit; `pyproject.toml` pins the
commit by SHA, because the stack's branches are force-pushed when rebased.
To move to the newest tip:

```bash
GITHUB_TOKEN=$(gh auth token) uv run python scripts/equine_stack_tip.py --write
uv lock
```

The script refuses to pin if a branch in the stack is behind its base.

## TimesFM version

`google/timesfm-2.5-200m-pytorch`. TimesFM 3.0 is newer, but its weights are
licensed for non-commercial, non-production use, which excludes "direct or
indirect interactions with end users"; the 2.5 weights are Apache-2.0.

## Data

ESPN's public JSON API, every regular-season, play-in and playoff game since
2012-13 (stats.nba.com blocks cloud runners). Tables are one CSV per season
under `data/`, so a daily commit is a small text diff:

| Table | Rows |
|---|---|
| `data/games/` | one per game, final or scheduled |
| `data/team_box/` | one per team per final game |
| `data/player_box/` | one per player per final game, DNPs included |
| `data/predictions/` | the last prediction published before each tip-off |
| `data/player_predictions/` | the same for player stat lines |

About 5% of 2012-16 games have a final score but no box score on ESPN; their
team ratings use the score and that season's median pace, and their player
rows are skipped.

## Leakage

Every feature for a game is computed from games that started before it:
rolling statistics are shifted one game, Elo is read before the game updates
it, and player values come from earlier appearances. The one exception is who
plays, read from the box score for past games and from ESPN's injury report
for upcoming ones; both are known at tip-off. `tests/test_features.py` checks
that rewriting a game and everything after it leaves that game's features
unchanged.

## Running it

```bash
uv sync
uv run nbapredicts collect            # backfill 2012-13 to today (~20 min)
uv run nbapredicts daily              # collect, retrain, forecast, publish
uv run nbapredicts backtest           # walk-forward evaluation -> site/data/backtest.json
uv run pytest
```

On a Mac whose Documents folder syncs to iCloud, keep the environment out of
iCloud (it evicts files when the disk fills up, and imports then hang):
`mkdir .venv.nosync && ln -s .venv.nosync .venv` before `uv sync`. The ESPN
cache already lives in `cache.nosync/`.

Predictions are for fun, not betting advice.
