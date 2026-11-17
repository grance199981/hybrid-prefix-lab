"""Small statistics helpers shared by the judge and summary scripts. Standard library only."""
import json
import statistics


def median(xs, ndigits=None):
    """Median of the values that are not None; None if there are none.
    ndigits: round the result (None keeps full precision)."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    m = statistics.median(xs)
    return m if ndigits is None else round(m, ndigits)


def percentile(sorted_vals, q):
    """Nearest-rank percentile of an already sorted list (q in [0, 1])."""
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def rnd(x, ndigits=3):
    return None if x is None else round(x, ndigits)


def describe(xs):
    """Summary of paired differences: count, median, share of positive values."""
    if not xs:
        return {"n": 0, "median": None, "share_positive": None}
    return {"n": len(xs), "median": round(statistics.median(xs), 2),
            "share_positive": round(sum(x > 0 for x in xs) / len(xs), 3)}


def line_fit(xs, ys):
    """Least-squares line y = intercept + slope * x. None with fewer than 3 points
    or when all x are equal."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return {"intercept": my - slope * mx, "slope": slope, "n": n}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
