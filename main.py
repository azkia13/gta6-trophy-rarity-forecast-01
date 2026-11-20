"""Forecast Grand Theft Auto VI trophy earn rates on Exophase.

For every trophy, forecast the percent of the players Exophase tracks for
the game who will have earned it at the case's target, as five quantiles
(5%, 25%, 50%, 75%, 95%).  Reads one case file /input/<id>.in, writes
/output/<id>.out.  Python 3.13 standard library only, no network, fully
deterministic.

Why the forecast is built from counts, not just from rates
------------------------------------------------------------
A rate is a ratio: earners / players tracked.  Exophase's page goes stale
between refreshes and then jumps: on launch night the page sat unchanged
for hours (638 players tracked) and one refresh took it to 3,868 players,
which pushed almost every rate *down* even though nobody un-earned a
trophy.  A forecaster that only extrapolates the rate itself cannot see
that coming; one that forecasts the numerator and the denominator
separately can.  So each trophy is forecast two ways and the two are
averaged on the logit scale:

* counts: fit log(earners) against log(1 + hours since launch) and
  log(players tracked) against the same clock, then divide the two
  forecasts.  This is the dilution-aware leg: when the tracked base grows
  faster than a trophy's earners, the rate falls, and the leg says so.
  The earners fit is a concave quadratic in log-log space once there
  are enough distinct reads, so it bends over as a trophy saturates
  instead of extrapolating its early sprint forever; with few reads it
  falls back to a straight line.  Earners never forecast below the last
  count seen, since a cumulative count cannot fall.
* rate trend: fit logit(rate) against log(1 + hours since launch),
  weighted by each read's earner count, so the noisy launch-night
  reads -- a rate of 0.63% is four players, and one player more or
  fewer moves it a quarter -- count for little next to the later reads
  with thousands of earners behind them.  This leg saturates properly
  as a rate approaches 100% and is the steadier one for trophies whose
  earners track the player base closely.  The two legs are averaged on
  the logit scale with weights 0.35 / 0.65 in the trend leg's favour.

Both fits use only *distinct* reads (consecutive reads with an identical
page are one observation, not many), because Exophase reprints the same
stale page for hours and the repeats carry no information.

A light prior keeps trophies with no usable history honest.  The prior
is a log-logistic curve per trophy group (story chapter, side strand,
grind, easy one-off, platinum) with a week-one ceiling and a half-time
taken from how Rockstar launches actually play out: the opening activity
is near-universal within hours, chapter 1 takes most players the best
part of a day, later chapters and the finale take days, and grind
trophies (250 stunts, 100% completion, gold medals everywhere) barely
move in week one.  The prior's weight shrinks as the case's own history
grows, so with two days of real reads the data, not the prior, decides.

Spread: the logit spread combines a base term, an extrapolation term
that grows with the log-time gap to the target, and a refresh term that
is wide when the history has seen few refreshes (a flat history can hide
a big jump, as launch night proved) and narrows as refreshes accumulate.
Trophies nobody has earned yet get extra spread.  Quantiles are the
median shifted by the normal quantiles times that spread, scaled by
0.85 -- backtests over three different launch-shape families put the
honest spread a little inside the raw formula -- and rounded to four
decimals half-up like the house baseline.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Group priors: (week-one ceiling in percent, half-time in hours, shape).
# The curve is cap * h**a / (h**a + half**a) at h hours after launch.
GROUP_PRIOR = {
    "intro": (98.0, 3.0, 1.0),      # the opening activity: nearly everyone
    "chapter1": (45.0, 40.0, 1.2),  # chapter 1 takes ~a day of play
    "chapter2": (30.0, 80.0, 1.2),
    "chapter3": (20.0, 130.0, 1.2),
    "chapter4": (14.0, 180.0, 1.2),
    "finale": (8.0, 220.0, 1.2),    # final mission
    "side": (12.0, 120.0, 1.1),     # side strands / named missions
    "easy": (18.0, 90.0, 1.0),      # one-off activities and exploration
    "grind": (4.0, 250.0, 1.0),     # collections, counts, medals
    "platinum": (0.5, 300.0, 1.0),  # needs 100% completion
}

GRIND_WORDS = (
    "250 stunts", "100% completion", "gold medal", "100 locations",
    "collectible card", "50 challenges", "discover 100",
    "purchase all garage", "earn $250,000", "win 20 different races",
    "spend $100,000", "register 10 vehicles", "drone race at every",
    "meditate at every spot", "8 mission strands", "rob 5 bank vaults",
    "illegal operation belonging to each gang", "explore all of leonida",
)


def parse_time(text):
    """ISO 8601 timestamp (with Z or an offset) to an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def ols(xs, ys):
    """Least-squares line through (xs, ys); returns (intercept, slope)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def solve3(matrix, vector):
    """Solve a 3x3 linear system by Gauss elimination with pivoting."""
    rows = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda r: abs(rows[r][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        if abs(rows[col][col]) < 1e-12:
            return None
        for r in range(3):
            if r == col:
                continue
            factor = rows[r][col] / rows[col][col]
            for c in range(col, 4):
                rows[r][c] -= factor * rows[col][c]
    return [rows[i][3] / rows[i][i] for i in range(3)]


def fit_earners(points, x_target):
    """Forecast a trophy's earner count at x_target from (x, earners).

    A concave quadratic in log-log space when there are enough points
    (it bends as growth saturates), else a straight line.  Returns the
    forecast count, never below the last count seen.
    """
    xs = [x for x, _ in points]
    ys = [math.log(e) for _, e in points]
    last = points[-1][1]
    if len(points) >= 4 and len({e for _, e in points}) > 1:
        centre = sum(xs) / len(xs)
        design = [[1.0, x - centre, (x - centre) ** 2] for x in xs]
        normal = [[sum(d[i] * d[j] for d in design) for j in range(3)]
                  for i in range(3)]
        rhs = [sum(d[i] * y for d, y in zip(design, ys)) for i in range(3)]
        normal[2][2] += 0.5  # ridge on the curvature only
        coef = solve3(normal, rhs)
        if coef is not None:
            curve = min(coef[2], 0.0)  # concave only: growth slows, never re-accelerates
            shift = x_target - centre
            value = coef[0] + coef[1] * shift + curve * shift * shift
            return max(math.exp(value), last)
    intercept, slope = ols(xs, ys)
    slope = min(max(slope, -0.5), 4.0)
    return max(math.exp(intercept + slope * x_target), last)


def weighted_ols(xs, ys, weights):
    """Weighted least-squares line; returns (intercept, slope)."""
    total = sum(weights)
    mx = sum(w * x for w, x in zip(weights, xs)) / total
    my = sum(w * y for w, y in zip(weights, ys)) / total
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(weights, xs))
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum(w * (x - mx) * (y - my)
              for w, x, y in zip(weights, xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def group_of(trophy):
    """Classify a trophy from its tier, name and description."""
    if trophy.get("tier") == "platinum":
        return "platinum"
    desc = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    if "complete chapter 1" in desc:
        return "chapter1"
    if "complete chapter 2" in desc:
        return "chapter2"
    if "complete chapter 3" in desc:
        return "chapter3"
    if "complete chapter 4" in desc:
        return "chapter4"
    if "complete the final mission" in desc:
        return "finale"
    if "100% completion" in desc:
        return "platinum"
    if "gone fishing" in desc or "just like paradise" in name:
        return "intro"
    if any(word in desc for word in GRIND_WORDS):
        return "grind"
    if desc.startswith("complete "):
        return "side"
    return "easy"


def prior_rate(group, hours):
    """The group's log-logistic prior curve at `hours` after launch."""
    cap, half, shape = GROUP_PRIOR[group]
    h = max(hours, 0.05)
    grow = h ** shape
    return cap * grow / (grow + half ** shape)


def distinct_reads(history):
    """Collapse consecutive reads that reprint the same page into one.

    Returns a list of (slot datetime, players tracked, rates tuple).
    The first occurrence's time stands for the whole stale run.
    """
    out = []
    prev = None
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if key == prev:
            continue
        prev = key
        out.append((parse_time(read["slot"]), read.get("players_tracked"),
                    read.get("rates") or ()))
    return out


def forecast_trophy(trophy, reads, hours_target, x_target):
    """One trophy's five quantiles at the target, on the logit scale."""
    idx = trophy["_index"]
    group = trophy["_group"]

    # The trophy's own series over the distinct reads: (x, players, rate).
    series = []
    for slot, players, rates in reads:
        if idx >= len(rates) or rates[idx] is None or players is None:
            continue
        hours = max((slot - trophy["_release"]).total_seconds() / 3600.0, 0.05)
        series.append((math.log1p(hours), players, float(rates[idx])))
    if not series:
        # No usable history at all: the prior, with a wide spread.
        med = logit(prior_rate(group, hours_target))
        return quantiles(med, 1.3)

    x_last = series[-1][0]
    last_rate = series[-1][2]
    last_n = series[-1][1]
    n_distinct = len(series)

    # --- Leg 1: counts.  Forecast earners and the tracked base apart. ---
    xs_n = [x for x, _, _ in series]
    ys_n = [math.log(max(p, 1)) for _, p, _ in series]
    if len(set(ys_n)) > 1 and len(series) >= 2:
        a_n, b_n = ols(xs_n, ys_n)
        b_n = min(max(b_n, 0.0), 2.0)
    else:
        # No observed growth in the tracked base: fall back to a launch
        # prior of roughly one power of log-time, so a flat history still
        # expects dilution instead of pretending the base stands still.
        b_n, a_n = 0.8, math.log(max(last_n, 1)) - 0.8 * x_last
    n_target = math.exp(a_n + b_n * x_target)
    n_target = min(max(n_target, last_n), last_n * 300.0)

    earners = [(x, r * p / 100.0) for x, p, r in series]
    positive = [(x, e) for x, e in earners if e > 0]
    if len(positive) >= 2 and len({e for _, e in positive}) > 1:
        e_target = fit_earners(positive, x_target)
    else:
        e_target = positive[-1][1] if positive else 0.0
    e_target = min(max(e_target, 0.0), 0.995 * n_target)
    if e_target > 0:
        rate_counts = clamp_rate(100.0 * e_target / n_target)
    else:
        rate_counts = None

    # --- Leg 2: the rate's own logit trend on the log-time clock. ---
    usable = [(x, r) for x, _, r in series if 0.0 < r < 100.0]
    if len(usable) >= 2 and len({r for _, r in usable}) > 1:
        weights = [max(r * p / 100.0, 0.5)
                   for x, p, r in series if 0.0 < r < 100.0]
        a_r, b_r = weighted_ols([x for x, _ in usable],
                                [logit(r) for _, r in usable], weights)
        b_r = min(max(b_r, -2.0), 4.0)
        rate_trend = clamp_rate(inv_logit(a_r + b_r * x_target))
    elif last_rate > 0:
        rate_trend = clamp_rate(last_rate)
    else:
        rate_trend = None

    # --- Combine on the logit scale, pulled toward the group prior. ---
    legs = []
    if rate_counts is not None:
        legs.append((0.35, logit(rate_counts)))
    if rate_trend is not None:
        legs.append((0.65, logit(rate_trend)))
    if not legs:
        # Never earned, never seen moving: the prior is all there is.
        return quantiles(logit(prior_rate(group, hours_target)), 1.3)
    centre = sum(w * v for w, v in legs) / sum(w for w, _ in legs)

    if n_distinct <= 1:
        w_prior = 0.30
    elif n_distinct <= 3:
        w_prior = 0.18
    elif n_distinct <= 6:
        w_prior = 0.10
    else:
        w_prior = 0.05
    if last_rate == 0.0:
        # A trophy nobody holds yet: the flat trend leg is meaningless
        # (it would anchor the median at zero), so trust the prior more.
        w_prior = 0.60
    median_logit = (1.0 - w_prior) * centre + w_prior * logit(
        prior_rate(group, hours_target))

    # --- Spread: base + extrapolation + refresh surprise. ---
    gap = max(0.0, x_target - x_last)
    refreshes = max(0, n_distinct - 1)
    var = 0.35 ** 2
    var += (0.28 * gap * (1.0 + 1.0 / math.sqrt(max(n_distinct, 1)))) ** 2
    var += (0.85 / math.sqrt(1.0 + refreshes)) ** 2
    if last_rate == 0.0:
        var += 0.30 ** 2
    elif positive and positive[-1][1] <= 2.0:
        var += 0.15 ** 2
    sigma = min(max(math.sqrt(var) * 0.85, 0.40), 1.50)
    return quantiles(median_logit, sigma)


def quantiles(median_logit, sigma):
    return [round4(clamp_rate(inv_logit(median_logit + z * sigma)))
            for z in Z]


def forecast_case(case):
    if case.get("levels") != LEVELS:
        raise ValueError("levels are not the contest's five")
    history = case.get("history") or []
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if case.get("game", {}).get("release_date") else None
    target = parse_time(case["target"])
    reads = distinct_reads(history)
    if release is None and reads:
        release = reads[0][0]
    if release is None:
        raise ValueError("no release date and no history")
    hours_target = max((target - release).total_seconds() / 3600.0, 0.05)
    x_target = math.log1p(hours_target)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        trophy = dict(trophy)
        trophy["_index"] = i
        trophy["_group"] = group_of(trophy)
        trophy["_release"] = release
        rows.append({"id": trophy["id"],
                     "q": forecast_trophy(trophy, reads, hours_target,
                                          x_target)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
