# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/azkia13/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/azkia13/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecast, for every one of Grand Theft Auto VI's 55 PlayStation 5 trophies, the percent of the players Exophase tracks for the game who will have earned it at 2026-11-26T00:00:00Z, a week after launch, as five quantiles (5%, 25%, 50%, 75%, 95%). The program reads one case file (`/input/<id>.in`) and writes `/output/<id>.out`. Python 3.13 standard library only, no network, deterministic.

## Model

A trophy's earn rate is a ratio -- earners over players tracked -- and Exophase's page goes stale between refreshes and then jumps. On launch night the page sat unchanged for hours (638 players tracked) and one refresh took it to 3,868 players, pushing almost every rate *down* even though nobody un-earned a trophy. A forecaster that only extrapolates the rate cannot see that coming; one that forecasts the numerator and the denominator separately can. So each trophy is forecast two ways and the two are averaged on the logit scale (weights 0.35 / 0.65):

- **Counts leg.** Log(earners) and log(players tracked) are each fit against log(1 + hours since launch) and the two forecasts are divided. The earners fit is a concave quadratic in log-log space once there are enough distinct reads, so it bends over as a trophy saturates instead of extrapolating its early sprint forever; with few reads it falls back to a straight line. Forecast earners never fall below the last count seen. This is the dilution-aware leg: when the tracked base grows faster than a trophy's earners, the rate falls, and the leg says so.
- **Rate-trend leg.** Logit(rate) fit against log(1 + hours since launch), weighted by each read's earner count -- a launch-night rate of 0.63% is four players, and one player more or fewer moves it a quarter, so those reads count for little next to later reads with thousands of earners behind them. This leg saturates properly as a rate approaches 100%.

Both fits use only *distinct* reads: consecutive reads that reprint the same stale page are one observation, not many.

A light prior keeps trophies with no usable history honest: a log-logistic curve per trophy group (opening activity, story chapters 1-4, finale, side strands, easy one-offs, grinds, platinum) with a week-one ceiling and a half-time taken from how Rockstar launches actually play out -- the opening activity is near-universal within hours, chapter 1 takes most players the best part of a day, later chapters and the finale take days, and grind trophies (250 stunts, 100% completion, gold medals everywhere) barely move in week one. The prior's weight shrinks as the case's own history grows, so with two days of real reads the data, not the prior, decides.

**Spread.** The logit spread combines a base term, an extrapolation term that grows with the log-time gap to the target, and a refresh term that is wide when the history has seen few refreshes (a flat history can hide a big jump, as launch night proved) and narrows as refreshes accumulate. Trophies nobody has earned yet get extra spread. The whole spread is scaled by 0.85, the factor backtests put the honest spread at.

## Backtest

Scored with the contest's exact pinball loss on the logit scale. The practice answers are not published, so practice is scored against the refresh the rules describe (players tracked 638 -> 3,868 at 04:30Z, every rate but the 100% one falling), under two assumptions about how much earners grew in the meantime. The final-like cases are synthetic worlds built from the real trophy list: tracked players growing as a power law, each trophy's rate a saturating curve, Exophase reprinting a stale page for hours between refreshes, rates rounded to 2 decimals -- across three different curve families (log-logistic, exponential approach, logit-linear) and six random draws each.

| model | practice, earners frozen | practice, earners +25% | synthetic final, mean of 18 | synthetic final, mean of 8 |
|---|---:|---:|---:|---:|
| house persistence baseline | 1.688 | 1.476 | 0.374 | 0.374 |
| house curve baseline | 1.688 | 1.476 | 0.257 | 0.257 |
| this entry | **0.644** | **0.551** | **0.182** | **0.158** |

The practice gap comes mostly from the dilution leg: a flat history still expects the tracked base to grow, so the median falls toward the refreshed rates instead of sitting on the stale ones. The final gap comes from the saturating earners fit plus the precision-weighted trend: the persistence baseline's median is the day-2 rate (far too low for story trophies still climbing) and the curve baseline's last-24-hours line cannot bend.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Runs in well under a second per case on one CPU; two runs on the same case write byte-identical output.
