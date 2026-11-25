"""Usage-aware expert pruning: write a copy of the GGUF that keeps only the K most-used experts of every layer.

Usage:
  py scripts/prune_experts.py models/in.gguf models/out.gguf --keep 96 --traces results/traces_v2/*.csv.gz

How experts are ranked: for each physical layer, sum the routing weights each expert received over all tokens
and all loop passes of that layer in the traces (looped layers 6-17 share their weights across 3 passes).
The top K experts are kept; their slices are copied from every per-expert tensor
(ffn_gate_up_exps, ffn_down_exps, their .scale, ffn_gate_inp, exp_probs_b) and expert_count is set to K.
The router then simply never sees the dropped experts, so tokens that would have used one pick the next best.
"""
import argparse, csv, glob, gzip, io, json, os, sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "gguf-py"))
import gguf  # noqa: E402

N_PHYS, CS, CE = 24, 6, 18


def phys(il):
    return il if il < N_PHYS else CS + (il - N_PHYS) % (CE - CS)


def expert_scores(paths, n_exp):
    score = defaultdict(lambda: np.zeros(n_exp))
    for p in paths:
        opener = gzip.open if p.endswith(".gz") else open
        with opener(p, "rt", encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                ws = row[10:18]
                if any(w.lower() == "nan" for w in ws):
                    continue
                s = score[phys(int(row[1]))]
                for e, w in zip(row[2:10], ws):
                    s[int(e)] += float(w)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--keep", type=int, required=True)
    ap.add_argument("--traces", nargs="+", required=True)
    a = ap.parse_args()

    traces = [p for t in a.traces for p in glob.glob(t) if ".tokens." not in p]
    reader = gguf.GGUFReader(a.inp)
    arch = bytes(reader.fields[gguf.Keys.General.ARCHITECTURE].parts[-1]).decode()
    n_exp = int(reader.fields[f"{arch}.expert_count"].parts[-1][0])
    print(f"arch {arch}, {n_exp} experts per layer, keeping {a.keep}; ranking from {len(traces)} trace files")

    score = expert_scores(traces, n_exp)
    keep = {l: np.sort(np.argsort(-score[l])[:a.keep]) for l in range(N_PHYS)}
    kept_share = {l: float(score[l][keep[l]].sum() / max(score[l].sum(), 1e-9)) for l in range(N_PHYS)}
    print("routing weight kept per layer (training traces):",
          " ".join(f"{l}:{kept_share[l] * 100:.1f}%" for l in range(N_PHYS)))

    writer = gguf.GGUFWriter(a.out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        val = a.keep if field.name == f"{arch}.expert_count" else field.contents()
        writer.add_key_value(field.name, val, vt, sub_type=st)

    per_expert = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight", "ffn_gate_up_exps.scale",
                  "ffn_down_exps.scale", "ffn_gate_inp.weight", "exp_probs_b.bias")
    datas = []
    for t in reader.tensors:
        data = t.data
        if t.name.startswith("blk.") and t.name.split(".", 2)[2] in per_expert:
            l = int(t.name.split(".")[1])
            assert data.shape[0] == n_exp, (t.name, data.shape)
            data = np.ascontiguousarray(data[keep[l]])
        datas.append(data)
        writer.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for data in datas:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()

    with open(os.path.splitext(a.out)[0] + ".keep.json", "w") as f:
        json.dump({"keep": a.keep, "experts": {l: keep[l].tolist() for l in keep}, "kept_weight_share": kept_share}, f)
    print("wrote", a.out, f"{os.path.getsize(a.out) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
