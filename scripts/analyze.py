"""Analyze moe-trace CSVs.

Usage:
    py scripts/analyze.py results/traces/*.csv --experts 128 --out results/summary

Answers three questions:
  1. Skew: do a few experts get most of the work? (share of picks covered by the top 10/25/50% experts)
  2. Domains: do code, chat, math and Arabic use different experts? (overlap of each domain's top experts)
  3. Prediction: can layer L's choices predict layer L+1's choices? (hit rate of a simple co-occurrence predictor)
"""
import argparse, csv, json, math, os, sys
from collections import Counter, defaultdict


def load(path):
    rows = defaultdict(dict)  # pos -> layer -> list of experts
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r)
        k = sum(1 for h in head if h.startswith("e"))
        for row in r:
            pos, layer = int(row[0]), int(row[1])
            rows[pos][layer] = [int(x) for x in row[2:2 + k]]
    return rows, k


def coverage(counts, n_exp, frac):
    vals = sorted((counts.get(e, 0) for e in range(n_exp)), reverse=True)
    total = sum(vals) or 1
    top = max(1, round(n_exp * frac))
    return sum(vals[:top]) / total


def entropy_ratio(counts, n_exp):
    total = sum(counts.values()) or 1
    h = -sum(c / total * math.log(c / total) for c in counts.values() if c)
    return h / math.log(n_exp)  # 1.0 = perfectly even, lower = more skewed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--top", type=int, default=32, help="experts per layer counted as a domain's favourites")
    ap.add_argument("--out", default="results/summary")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    domains = {}
    for p in a.traces:
        name = os.path.splitext(os.path.basename(p))[0]
        domains[name], k = load(p)
    layers = sorted({l for d in domains.values() for pos in d.values() for l in pos})

    # 1. skew per layer, all domains together
    per_layer = {l: Counter() for l in layers}
    per_dom_layer = {d: {l: Counter() for l in layers} for d in domains}
    for d, rows in domains.items():
        for pos, ls in rows.items():
            for l, ex in ls.items():
                per_layer[l].update(ex)
                per_dom_layer[d][l].update(ex)
    skew = {l: {"top10": coverage(c, a.experts, .10), "top25": coverage(c, a.experts, .25),
                "top50": coverage(c, a.experts, .50), "evenness": entropy_ratio(c, a.experts),
                "unused": sum(1 for e in range(a.experts) if c.get(e, 0) == 0)} for l, c in per_layer.items()}

    # 2. domain overlap: Jaccard of each domain's top experts, averaged over layers
    names = list(domains)
    overlap = {}
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            js = []
            for l in layers:
                tx = {e for e, _ in per_dom_layer[x][l].most_common(a.top)}
                ty = {e for e, _ in per_dom_layer[y][l].most_common(a.top)}
                if tx and ty:
                    js.append(len(tx & ty) / len(tx | ty))
            overlap[f"{x}~{y}"] = sum(js) / len(js) if js else None

    # 3. prediction: co-occurrence table from even positions, tested on odd positions
    co = defaultdict(Counter)  # (layer, expert) -> Counter of next-layer experts
    hits = tested = 0
    base_hits = 0
    for phase in ("train", "test"):
        for d, rows in domains.items():
            for pos, ls in rows.items():
                if (pos % 2 == 0) != (phase == "train"):
                    continue
                for i in range(len(layers) - 1):
                    l, nl = layers[i], layers[i + 1]
                    if l not in ls or nl not in ls:
                        continue
                    cur, nxt = ls[l], ls[nl]
                    if phase == "train":
                        for e in cur:
                            co[(l, e)].update(nxt)
                    else:
                        votes = Counter()
                        for e in cur:
                            votes.update(co[(l, e)])
                        guess = {e for e, _ in votes.most_common(len(nxt))}
                        popular = {e for e, _ in per_layer[nl].most_common(len(nxt))}
                        hits += len(guess & set(nxt))
                        base_hits += len(popular & set(nxt))
                        tested += len(nxt)

    summary = {
        "tokens": {d: len(r) for d, r in domains.items()},
        "layers": len(layers), "experts_per_token": k, "experts": a.experts,
        "skew_mean": {m: sum(s[m] for s in skew.values()) / len(skew) for m in ("top10", "top25", "top50", "evenness", "unused")},
        "domain_overlap_jaccard_top%d" % a.top: overlap,
        "next_layer_prediction": {"cooccurrence_hit_rate": hits / tested if tested else None,
                                  "most_popular_hit_rate": base_hits / tested if tested else None,
                                  "random_hit_rate": k / a.experts},
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump({"summary": summary, "per_layer_skew": skew}, f, indent=2)
    with open(os.path.join(a.out, "usage_per_layer.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer"] + [f"x{e}" for e in range(a.experts)])
        for l in layers:
            w.writerow([l] + [per_layer[l].get(e, 0) for e in range(a.experts)])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main())
