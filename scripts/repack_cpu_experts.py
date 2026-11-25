"""Repack the experts of the CPU-side layers into a format the CPU unpacks faster (no experts removed).

Usage:
  py scripts/repack_cpu_experts.py models/in.gguf models/out.gguf --layers 0-7,18-23 --to Q4_0 [--tensors gate_up]

Why: with loop-aware placement the GPU holds the experts of layers 8-17 and the CPU computes the rest. Measured
speed keeps rising with more CPU threads, and the RAM is only ~10% busy, so the CPU is limited by *unpacking*
the 2-bit IQ2_XXS codebook format, not by memory. Q4_0 has 4-bit values (finer than the 2-bit source, so the
values carry over almost exactly) and one of the fastest CPU kernels in llama.cpp. The file grows; RAM has room.
"""
import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "llama.cpp", "gguf-py"))
import gguf  # noqa: E402
from gguf import quants  # noqa: E402


def parse_layers(s):
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--layers", required=True)
    ap.add_argument("--to", default="Q4_0", choices=["Q4_0", "Q8_0", "Q5_0"])
    ap.add_argument("--tensors", default="gate_up", help="comma list: gate_up,down")
    a = ap.parse_args()
    layers = parse_layers(a.layers)
    names = {f"ffn_{t}_exps.weight" for t in a.tensors.split(",")}
    target = getattr(gguf.GGMLQuantizationType, a.to)

    reader = gguf.GGUFReader(a.inp)
    arch = bytes(reader.fields[gguf.Keys.General.ARCHITECTURE].parts[-1]).decode()
    writer = gguf.GGUFWriter(a.out, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)

    plan = []
    for t in reader.tensors:
        parts = t.name.split(".")
        convert = (t.name.startswith("blk.") and int(parts[1]) in layers and ".".join(parts[2:]) in names)
        if convert:
            ne = [int(x) for x in t.shape]  # ggml order: ne0 (row length), ne1, ne2 (experts)
            row_bytes = quants.quant_shape_to_byte_shape((ne[0],), target)[-1]
            shape = (ne[2], ne[1], row_bytes)
            writer.add_tensor_info(t.name, shape, np.dtype(np.uint8), int(np.prod(shape)), target)
            plan.append((t, True, ne))
        else:
            writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
            plan.append((t, False, None))

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t, convert, ne in plan:
        if not convert:
            writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
            continue
        out = []
        for e in range(ne[2]):  # one expert at a time keeps memory small
            w = quants.dequantize(t.data[e], t.tensor_type).reshape(ne[1], ne[0]).astype(np.float32)
            out.append(quants.quantize(w, target))
        writer.write_tensor_data(np.stack(out), tensor_endianess=reader.endianess)
        print(f"repacked {t.name} {t.tensor_type.name} -> {a.to}", flush=True)
    writer.close()
    print("wrote", a.out, f"{os.path.getsize(a.out) / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
