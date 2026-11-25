"""Draws the figures for the README from moe-trace CSVs.

Usage: py scripts/plot.py results/traces_v2 --out docs
Figures:
  heatmap_all.png      expert usage per layer pass (rows) and expert id (columns), all domains
  heatmap_domains.png  the same for each domain side by side, for one looped layer
  coverage.png         share of expert picks covered by the hottest N% of experts, per domain
  loop_reuse.png       experts shared between loop passes of the same token, per looped layer
"""
import argparse, csv, glob, os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

N_EXP, N_PHYS, CS, CE = 128, 24, 6, 18


def phys(il):
    return il if il < N_PHYS else CS + (il - N_PHYS) % (CE - CS)


def npass(il):
    return 0 if il < N_PHYS else 1 + (il - N_PHYS) // (CE - CS)


def load(path):
    rows = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if any(w.lower() == "nan" for w in row[10:18]):
                continue  # broken trace row (NaN router weights)
            rows[int(row[0])][int(row[1])] = [int(x) for x in row[2:10]]
    return rows


def pass_order():
    # logical layer indices in the order a token visits them
    order = list(range(CS))
    for p in range(3):
        for l in range(CS, CE):
            order.append(l if p == 0 else N_PHYS + (p - 1) * (CE - CS) + (l - CS))
    order += list(range(CE, N_PHYS))
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tracedir")
    ap.add_argument("--out", default="docs")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    doms = {os.path.basename(p)[:-4]: load(p) for p in sorted(glob.glob(os.path.join(a.tracedir, "*.csv")))
            if not p.endswith(".tokens.csv")}
    order = pass_order()
    label = [f"L{phys(il)}" + (f" p{npass(il) + 1}" if CS <= phys(il) < CE else "") for il in order]

    counts = {d: np.zeros((len(order), N_EXP)) for d in doms}
    for d, rows in doms.items():
        idx = {il: i for i, il in enumerate(order)}
        for ls in rows.values():
            for il, ex in ls.items():
                for e in ex:
                    counts[d][idx[il], e] += 1

    # 1. heatmap, all domains, each row normalised so 1.0 = even usage
    tot = sum(counts.values())
    norm = tot / tot.sum(axis=1, keepdims=True) * N_EXP
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(norm, aspect="auto", cmap="magma", vmin=0, vmax=4)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(label, fontsize=6)
    ax.set_xlabel("expert id")
    ax.set_title("Inkling 2 Mini: expert usage per layer pass (1.0 = even share)")
    fig.colorbar(im, ax=ax, label="usage relative to an even split")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "heatmap_all.png"), dpi=130)
    plt.close(fig)

    # 2. one looped layer, pass 1, per domain
    row = order.index(12)
    fig, axs = plt.subplots(len(doms), 1, figsize=(12, 1.4 * len(doms) + 1), sharex=True)
    for ax, (d, c) in zip(np.atleast_1d(axs), counts.items()):
        v = c[row] / max(1, c[row].sum()) * N_EXP
        ax.bar(range(N_EXP), v, width=1.0, color="#c2410c")
        ax.set_ylabel(d, rotation=0, ha="right", va="center")
        ax.set_ylim(0, max(4, v.max()))
    np.atleast_1d(axs)[-1].set_xlabel("expert id (layer 12, first pass)")
    fig.suptitle("Different domains light up different experts")
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "heatmap_domains.png"), dpi=130)
    plt.close(fig)

    # 3. coverage curve
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    xs = np.arange(1, N_EXP + 1) / N_EXP * 100
    for d, c in counts.items():
        cov = []
        for r in c:
            s = np.sort(r)[::-1]
            cov.append(np.cumsum(s) / max(1, s.sum()))
        ax.plot(xs, np.mean(cov, axis=0) * 100, label=d)
    ax.plot(xs, xs, "k--", lw=1, label="even usage")
    ax.set_xlabel("hottest experts kept (% of 128, per layer)")
    ax.set_ylabel("expert picks covered (%)")
    ax.set_title("How much work the hottest experts do")
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "coverage.png"), dpi=130)
    plt.close(fig)

    # 4. loop reuse per looped layer
    reuse12, reuse23 = Counter(), Counter()
    n = Counter()
    for rows in doms.values():
        for ls in rows.values():
            for l in range(CS, CE):
                a1, a2, a3 = l, N_PHYS + (l - CS), N_PHYS + (CE - CS) + (l - CS)
                if a1 in ls and a2 in ls and a3 in ls:
                    reuse12[l] += len(set(ls[a1]) & set(ls[a2]))
                    reuse23[l] += len(set(ls[a2]) & set(ls[a3]))
                    n[l] += 1
    ls_ = list(range(CS, CE))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([x - .2 for x in ls_], [reuse12[l] / n[l] for l in ls_], .4, label="pass 1 → 2")
    ax.bar([x + .2 for x in ls_], [reuse23[l] / n[l] for l in ls_], .4, label="pass 2 → 3")
    ax.axhline(8 * 8 / N_EXP, color="k", ls="--", lw=1, label="random (0.5)")
    ax.set_xlabel("looped layer")
    ax.set_ylabel("experts reused (of 8)")
    ax.set_ylim(0, 8)
    ax.set_title("Looped layers reuse most of their experts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(a.out, "loop_reuse.png"), dpi=130)
    plt.close(fig)
    print("wrote figures to", a.out)


if __name__ == "__main__":
    main()
