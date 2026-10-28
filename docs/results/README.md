# Can the model beat the betting market? Results so far

All numbers are walk-forward: each test season is predicted by models trained
only on earlier seasons, and every comparison is on the same games. Brier
differences are model minus market (negative = better than the market), with
paired bootstrap 95% intervals. Lines are the median over sportsbooks with the
bookmaker's margin removed (power method); prediction sites (numberfire,
teamrankings, accuscore) and ESPN's own consensus entry are excluded.

## The market

| Season | Opening line Brier | Closing line Brier | Close minus open |
|---|---|---|---|
| 2023-24 | 0.2029 | 0.1993 | −0.0037 (CI excludes 0) |
| 2024-25 | 0.2065 | 0.2012 | −0.0052 (CI excludes 0) |
| 2025-26 | 0.2017 | 0.1974 | −0.0043 (CI excludes 0) |

The close is better than the open in every season: news that arrives during
the day (injuries, rest) is worth about 0.004-0.005 Brier. Recalibrating the
close (favorite-longshot bias) gains nothing.

## Our models against the market

`scripts/market_eval.py`. "Stacked" = a head over two inputs, the line's
log-odds and an out-of-fold logistic model of the 25 game features.

| Pre-game availability (honest) | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| EQUINE stack vs close | +0.0006 | +0.0000 | +0.0003 |
| Logistic stack vs close | +0.0001 | −0.0004 | −0.0001 |
| EQUINE stack vs open | +0.0012 | −0.0002 | +0.0005 |
| Logistic stack vs open | +0.0002 | −0.0005 | −0.0003 |

No interval excludes zero: **the stacked model matches the closing line and
does not beat either line.** The features' weight in the stack is about −0.08
next to the close and +0.09..+0.13 next to the open: they hold a little
information the opening line has not priced, not enough to show up as a
measurable gain.

With *hindsight* availability (who actually played) the logistic stack beat
the open in 2024-25 (−0.0012, CI excludes 0) and the features' weight next to
the open was +0.22..+0.28. That edge came from late scratches nobody knew at
prediction time, so the open-line model was not deployed.

## What changed EQUINE's numbers

- Given the 25 features and the line together, EQUINE's GP drowned the line
  (Brier 0.212 vs the close's 0.199 in 2023-24). Stacking fixed it.
- EquineGP LayerNorms its embedding before the random-feature GP; for a
  one- or two-dimensional input like a stacked log-odds that discards the
  magnitude. With it off and the mean-field factor at pi/8 (SNGP's value),
  the EQUINE stack went from not beating the (hindsight-era) open at all to
  beating it by 0.00065 (`scripts/stack_sweep.py`).

## Injury-report history

139,733 rows from 1,618 game days (2018-19 on, Wayback Machine and the NBA
CDN). Joined to box scores, matched players with each status played: Out
0.1%, Doubtful 1.4%, Questionable 51%, Probable 82%, Available 85%.

## Next

Turning "player X is out" into a team-strength change is where the remaining
information is: the availability feature is a sum of box-score Game Scores.
Play-by-play player impact (RAPM-style) and projected minutes are the next
experiment (docs/research_plan.md, E4).
