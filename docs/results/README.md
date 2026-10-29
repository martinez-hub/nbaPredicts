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

## Player impact (minutes-weighted plus-minus)

`scripts/impact_eval.py`. Game-level adjusted plus-minus ratings (ridge 5,
two-year half-life, refit monthly on earlier games only), projected minutes
and the injury report give each team's expected strength in points of
margin. Rating quality on the next month's games with actual minutes: margin
MAE 11.16, correlation 0.475, 68.2% winners.

| Brier | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| Logistic, 25 features | 0.2096 | 0.2078 | 0.2048 |
| Logistic, + impact | 0.2082 | 0.2063 | 0.2040 |
| Stack + impact vs close | +0.0001 | −0.0003 | −0.0001 |
| Stack + impact vs open | −0.0003 | −0.0003 | −0.0001 |

The impact features improve the stand-alone model every season; next to
either line they add nothing measurable (feature weight next to the open
0.05-0.10). The opening line already prices roster strength.

## Features-only EQUINE

`scripts/equine_features_sweep.py`. With the impact features, EquineGP's
defaults (LayerNorm on GP features, mean-field factor 25) gave Brier 0.2139;
without the LayerNorm and with pi/8, 0.2070 (logistic 0.2061). Those are now
the defaults.

## Final backtest (published, 2023-24..2025-26)

| Season | EQUINE with the line | Closing line | EQUINE, no line | Logistic | Elo |
|---|---|---|---|---|---|
| 2023-24 | 68.7% / 0.1999 | 68.9% / 0.1993 | 66.3% / 0.2096 | 65.7% / 0.2089 | 64.4% / 0.2182 |
| 2024-25 | 69.6% / 0.2013 | 69.3% / 0.2012 | 67.6% / 0.2067 | 68.0% / 0.2066 | 64.7% / 0.2185 |
| 2025-26 | 68.5% / 0.1977 | 68.5% / 0.1974 | 68.7% / 0.2052 | 68.2% / 0.2042 | 67.4% / 0.2110 |

(accuracy / Brier). By EQUINE's GP-variance tercile the no-line model picked
62.9%, 66.9% and 72.8% correctly (most ordinary to rarest), calibration
error 0.033, 0.030, 0.028. With the current settings, high variance marks
rare, mostly lopsided matchups, which are easier to call; with EquineGP's
defaults (LayerNorm, mean-field 25) the order had been reversed (75/65/60%).

## TimesFM's score as a winner pick

`scripts/timesfm_margin.py`. TimesFM forecasts each team's points from that
team's own series (plus home/away, the opponent's defensive rating and rest),
so the margin it implies is a separate opinion about the winner. One-step,
zero-shot forecasts for every game of 2018-19..2025-26; test seasons scored on
games with a closing line, everything fit on earlier seasons only.

| Season | TimesFM picks right | EQUINE (with the line) | Disagree | EQUINE right when they disagree |
|---|---|---|---|---|
| 2021-22 | 59.5% | 68.1% | 25% | 221 of 328 |
| 2022-23 | 58.3% | 65.8% | 36% | 285 of 471 |
| 2023-24 | 60.7% | 68.7% | 27% | 232 of 358 |
| 2024-25 | 64.4% | 69.6% | 22% | 183 of 297 |
| 2025-26 | 64.4% | 68.7% | 21% | 166 of 276 |

As a third input to the stack, the TimesFM margin gets a weight of about
-0.01 per point and changes the logistic stack's Brier by -0.0002..+0.0005
(only 2021-22's -0.0002 excludes zero); the EQUINE stack with it lands
-0.0005..+0.0013 from the published one, no interval excluding zero. Margin MAE: TimesFM
11.0-11.9 points, EQUINE's win probability read as points 10.0-11.1, closing
spread 9.8-11.0.

Keeping TimesFM's total and splitting it by EQUINE's pick (margin = about 6.3 x
the pick's log-odds, the market's own spread-to-odds slope) lowered the
team-points MAE every season: 9.74 to 9.43, 9.44 to 9.10, 9.78 to 9.46, 9.49
to 9.29, 9.79 to 9.56. That is what the page now shows, so its projected
score always agrees with the pick.

## Training on recent seasons only

`scripts/train_window.py`. The game features are team-level and live
predictions use ESPN's current rosters, so retired players never enter a
prediction; dropping them from history would break the plus-minus fit (every
player on the floor has to be in it) and, in a backtest, use knowledge of who
was still playing years later. The real question is whether old seasons teach
relationships that no longer hold. Brier of the published stack trained on the
last N seasons, minus training on everything since 2013-14:

| Window | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| Last 8 seasons | -0.0003 | +0.0000 | +0.0002 |
| Last 5 seasons | -0.0000 | +0.0005 | +0.0005 |
| Last 3 seasons | +0.0009 (worse) | +0.0004 | +0.0014 (worse) |

No window helps; the last three seasons alone are worse in two of three
(intervals exclude zero). The feature model alone is mixed (last five seasons:
-0.0005, -0.0016, +0.0011). Training stays on every season.

## Injury reports published after tip-off

The history keeps each day's report from 6:30 PM ET, which comes after the
tip-off of about one game in ten (9.7% of 9,405 games on report days:
afternoon weekend and holiday games). Until 2026-10-28 those games' pre-game
availability read that report anyway; now a game only sees report rows
published before it starts (`features.pregame_rosters`), which on such days
usually leaves the previous day's report. Numbers in the sections above were
computed before this change; the final backtest above was rerun with it
(EQUINE with the line moved by at most 0.0001 Brier).

## 2026-27 games from before the page went live

The season began on Oct 20; the daily job's first slate was Oct 27.
`nbapredicts backfill` predicted the 52 games in between after the fact, each
day with both EQUINE models refit on the games before it, that day's injury
report as of 6:30 PM ET (only rows published before each tip-off), the
closing line, and zero-shot TimesFM. They are flagged `backfilled` in the
prediction log and marked on the page. Through Oct 27: 37 of 52 right
(the closing line also 37), plus 3 of 4 published before tip-off.

## NVIDIA Kumo-Forecast against TimesFM

`scripts/kumo_compare.py` and `scripts/kumo_infer.py`. Kumo-Forecast
(nvidia/Kumo-Forecast, Apache-2.0; formerly NV-Tesseract) is a MOMENT-1-large
encoder with a forecasting head trained on the Monash archive. It reads a
512-point window and takes no covariates known before the game. Scored on
2025-26, one step ahead, on the same rows as the published TimesFM backtest:
every team's points (2,636) and the 3,000-appearance player sample (seed 0).
Cross-channel checkpoint, direct mode, Apple GPU. Mean absolute error; the
last column is Kumo minus TimesFM alone with its bootstrap 95% interval.

| Forecast | TimesFM + XReg (page) | TimesFM alone | Recent average | Kumo | Kumo vs TimesFM alone |
|---|---|---|---|---|---|
| Team points | 9.79 | 10.37 | 10.35 | 10.73 | +0.35 (+0.22, +0.48) |
| Game total | 15.38 | 16.08 | 16.10 | 16.48 | +0.40 (+0.11, +0.70) |
| Player points | 4.56 | 4.57 | 4.58 | 4.66 | +0.10 (+0.04, +0.15) |
| Player rebounds | 1.87 | 1.87 | 1.91 | 1.93 | +0.06 (+0.03, +0.08) |
| Player assists | 1.33 | 1.34 | 1.37 | 1.39 | +0.05 (+0.03, +0.07) |
| Player threes | 0.87 | 0.87 | 0.90 | 0.91 | +0.04 (+0.03, +0.06) |

Kumo is worse than TimesFM on every row, with or without TimesFM's
covariates, and worse than the recent average too. Picking the side it gives
more points gets 56.1% of winners (TimesFM + XReg 64.4%). Player rows use
Kumo with each window z-scored by its own mean and spread; with the released
standardizer (one mean and spread, -5.67 and 8.69) player errors are far
larger (points 5.43), team errors identical. Half the player histories are
shorter than 256 games and padded (masked); restricted to the 1,512 with at
least 256, Kumo still trails TimesFM alone by 0.04-0.13. Its head was trained
to forecast 72 steps of smooth sensor and energy series from about 3 million
points, and per-game box-score counts are neither. TimesFM stays.

As an extra input next to TimesFM (`kumo_compare.py stack`): least squares
of the actual on TimesFM and Kumo, fit on 2025-26 games before Feb 1 and
scored on the rest, against the same fit on TimesFM alone. Kumo changed the
mean absolute error by +0.010 for team points (worse; interval +0.001 to
+0.021), -0.005 for player points, +0.004 rebounds, +0.002 assists (none
excluding zero) and -0.006 threes (-0.009 to -0.002). It reads the same
history TimesFM does, so it has little to add, and a forecast made further
ahead from that history has less; it is not wired in.

Two refinements, so the comparison is not of an untuned Kumo:

- Weights that follow each model's error (`kumo_compare.py adaptive`): each
  game day the combination is re-fit on the previous 600 team-game (300
  player-game) forecasts, either by least squares or by inverse squared
  recent error. Against TimesFM alone, team points: -0.012 (interval -0.07
  to +0.05) and +0.20 (worse); player stats: least squares +0.03 to +0.06
  (worse, the recent-window refit itself hurts), inverse error -0.005 to
  +0.013 (assists and threes worse, intervals excluding zero).
- Kumo retrained on NBA data (`kumo_finetune.py`): its MOMENT encoder stays
  frozen and its next-game output is retrained on the same windows as
  TimesFM's fine-tune (8,000 from team and player series in the three
  seasons before 2025-26, 15% held out, best epoch kept: held-out error 4.054
  to 4.035). On 2025-26 it improves on released Kumo but still trails:
  team points 10.51 (TimesFM alone 10.37, with covariates 9.79), player
  points 4.67 (4.57), rebounds 1.89 (1.87), assists 1.36 (1.34), threes 0.89
  (0.87); every gap to TimesFM alone excludes zero except rebounds (+0.018,
  -0.004 to +0.038). TimesFM's own fine-tuned head, on the same rows: 9.75,
  4.57, 1.85, 1.33, 0.87.

## Where this leaves "beating the market"

Matching the closing line is achieved; beating it, or the opening line, is
not, with every input tried so far. Remaining candidates from
docs/research_plan.md: stint-level RAPM from play-by-play, a text model over
the injury reports' reasons and logged news, and live grading against the
line at prediction time, which the prediction log now records.
