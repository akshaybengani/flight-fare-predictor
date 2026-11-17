"""One definition of "may drop" and "big save", shared by the India and US markets.

Pure Python, no model or data imports, so it can be tested on its own (see tests_drop_logic.py).

The rule, in plain terms
  * Look only at the future: curve points with fewer days before departure than today, plus the model's own
    price for today (P0).
  * Do not trust a single odd point. The "sustained expected low" is the lowest level the curve holds for two
    checkpoints in a row: min over adjacent future pairs of max(price_i, price_i+1). A one-point dip cannot
    produce it.
  * A flight "may drop" only if that low is at least `flag_pct` below BOTH the listed fare and the model's own
    price for today. (Comparing only with the listed fare mixes a scraped last-minute fare with model output.)
  * drop_pct and the saving are measured against the listed fare, the number the user sees.
  * For a tagged flight the card shows the expected low as its FareSight figure, so the number, the arrow and
    the tag always agree. Untagged flights keep the horizon-based FareSight figure.
This is a model estimate, not a promise.
"""


def sustained_low(series):
    """series: [(days_left, price), ...] in time order, series[0] being today's model price (P0).

    The expected low is the lowest level held at two adjacent future points: min over adjacent pairs of the dearer
    price of the pair. Returns (days_left, price, (days_left_a, days_left_b)), or None when there are fewer than two
    future points. days_left is that of the LATER point of the pair, the date by which the price has been at or below
    that level twice in a row. This is also the lowest point of the "held level" line drawn on the chart (the
    running two-point maximum), so the number on the card and the chart agree. Ties go to the earlier pair.
    """
    pts = series[1:]
    best = None
    for a, b in zip(pts, pts[1:]):
        level = max(a[1], b[1])
        if best is None or level < best[1]:
            best = (b[0], level, (a[0], b[0]))
    return best


def future_series(today_days_left, p0, curve_points, extra_points=()):
    """Build the time-ordered series: today's P0, then curve points strictly before today (fewer days left).

    curve_points / extra_points: iterables of (days_left, price). Duplicate days keep the first value given.
    """
    seen, fut = set(), []
    for d, p in list(curve_points) + list(extra_points):
        if d is None or p is None or d >= today_days_left or d in seen:
            continue
        seen.add(d)
        fut.append((d, p))
    fut.sort(key=lambda t: -t[0])
    return [(today_days_left, p0)] + fut


def big_save_rule(pct, save, big_pct, big_min):
    """A big save is a May drop whose fall is large in percent, or a large amount that is ALSO at least a 10% fall
    (an amount alone must not qualify, or a small fall on a very dear fare would count)."""
    return bool(pct >= big_pct or (big_min > 0 and save >= big_min and pct >= 10))


def drop_info(listed, series, flag_pct, big_pct, big_min, headline=None):
    """Decide may_drop / big_save for one flight. Money fields are in the market's own currency.

    headline: the price shown in the card's big slot (FareSight at the chosen horizon). When given, a flight is only
    tagged if that price is also at least 1% below the listed fare, so a tagged card never shows a rise or "no change"
    next to its tag.

    Returns a dict: may_drop, drop_pct (0 when not a drop), drop_best_days_left, drop_price, save, big_save,
    low_pair (the two checkpoints that hold the low), p0.
    """
    out = {"may_drop": False, "drop_pct": 0, "drop_best_days_left": None, "drop_price": None, "save": 0,
           "big_save": False, "low_pair": None, "p0": series[0][1] if series else None}
    if not series or not listed or listed <= 0:
        return out
    low = sustained_low(series)
    if low is None:
        return out
    d, price, pair = low
    k = 1 - flag_pct / 100.0
    if headline is not None and headline > listed * 0.99:
        return out
    if price <= series[0][1] * k + 1e-9 and price <= listed * k + 1e-9:
        pct = round((listed - price) / listed * 100, 1)
        save = max(0.0, listed - price)
        out.update(may_drop=True, drop_pct=pct, drop_best_days_left=d, drop_price=price, save=save, low_pair=list(pair))
        out["big_save"] = big_save_rule(pct, save, big_pct, big_min)
    return out
