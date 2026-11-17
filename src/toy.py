"""Python mirror of the Walkthrough toy (templates/walkthrough.html): the same one-question trees (stumps) and boosting.

Rules (identical in the browser):
  * Questions tried, in tie-break order: "Is it Business?", "Under N days before departure?" for N = 10, 20, 30, ...,
    "Is it 1 stop?", "Is it 0 stops (non-stop)?".
  * Two questions that split the flights into the same two groups count once (the first in the order above is kept).
  * A question that puts every flight on one side is skipped. A question already used in an earlier round is not asked again.
  * The best question is the one with the smallest total squared leftover error after each side is given its own average.
    Equal errors go to the earlier question in the order above."""
DEFAULT_ROWS = [("Economy", 30, 0, 3000), ("Economy", 20, 0, 3200), ("Economy", 5, 0, 4600), ("Economy", 2, 0, 6000),
                ("Economy", 25, 1, 2600), ("Economy", 3, 1, 5000), ("Business", 20, 0, 9000), ("Business", 3, 0, 13000)]


def questions(rows):
    qs = [("class", "Business", lambda r: r[0] == "Business")]
    top = max(r[1] for r in rows)
    n = 10
    while n <= top:
        qs.append(("days", n, (lambda n: lambda r: r[1] < n)(n)))
        n += 10
    qs += [("stops", 1, lambda r: r[2] == 1), ("stops", 0, lambda r: r[2] == 0)]
    seen, out = set(), []
    for k, a, f in qs:
        side = tuple(f(r) for r in rows)
        if all(side) or not any(side) or side in seen:
            continue
        seen.add(side); out.append((k, a, f))
    return out


def fit(rows, rate=0.5, trees=3):
    guess = [sum(r[3] for r in rows) / len(rows)] * len(rows)
    start = guess[0]; rounds = []; used = set()
    for _ in range(trees):
        res = [r[3] - g for r, g in zip(rows, guess)]
        best = None
        for kind, arg, f in questions(rows):
            if (kind, arg) in used:
                continue
            yes = [i for i, r in enumerate(rows) if f(r)]; no = [i for i in range(len(rows)) if i not in yes]
            my = sum(res[i] for i in yes) / len(yes); mn = sum(res[i] for i in no) / len(no)
            err = sum((res[i] - my) ** 2 for i in yes) + sum((res[i] - mn) ** 2 for i in no)
            if best is None or err < best["err"] - 1e-9:
                best = dict(kind=kind, arg=arg, f=f, err=err, my=my, mn=mn)
        if best is None:
            break
        used.add((best["kind"], best["arg"]))
        for i, r in enumerate(rows):
            guess[i] += rate * (best["my"] if best["f"](r) else best["mn"])
        rounds.append(best)
    return start, rounds, guess


def predict(start, rounds, row, rate=0.5):
    p = start; path = [p]
    for b in rounds:
        p += rate * (b["my"] if b["f"](row) else b["mn"]); path.append(p)
    return p, path
