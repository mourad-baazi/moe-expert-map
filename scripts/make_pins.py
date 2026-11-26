"""Pick the hottest experts of the CPU-side layers from the expert map, for the pinned MoE cache.

Usage: py scripts/make_pins.py --cpu-layers 0-9,18-23 --n 300 --out results/pins.txt [--eval]

Each trace row is one pass (0-47); passes map to physical layers (6-17 run 3 times), so an expert of a looped
layer collects the uses of all 3 passes. The N experts with the most uses overall are written as "layer expert".
--eval: pick from even token positions, report the share of CPU expert work the pins cover on odd positions.
"""
import argparse, collections, csv, gzip

CORPORA = ["chat", "code", "math", "arabic"]


def physical(p):
    return p if p < 6 else (6 + (p - 6) % 12 if p < 42 else p - 24)


def parse(s):
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-layers", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--out")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    cpu = parse(a.cpu_layers)
    rows = []
    for c in CORPORA:
        for r in csv.DictReader(gzip.open(f"results/traces_v2/{c}.csv.gz", "rt")):
            l = physical(int(r["layer"]))
            if l in cpu:
                rows.append((int(r["pos"]), l, [int(r[f"e{i}"]) for i in range(8)]))
    pick = [r for r in rows if r[0] % 2 == 0] if a.eval else rows
    cnt = collections.Counter((l, e) for _, l, es in pick for e in es)
    pins = [k for k, _ in cnt.most_common(a.n)]
    if a.eval:
        s = set(pins)
        test = [r for r in rows if r[0] % 2 == 1]
        hit = sum((l, e) in s for _, l, es in test for e in es)
        print(f"{a.n} pins cover {hit / (8 * len(test)):.1%} of CPU expert work on held-out tokens")
    if a.out:
        with open(a.out, "w") as f:
            for l, e in sorted(pins):
                f.write(f"{l} {e}\n")
        print("wrote", a.out, len(pins), "pins, per layer:", dict(sorted(collections.Counter(l for l, _ in pins).items())))


if __name__ == "__main__":
    main()
