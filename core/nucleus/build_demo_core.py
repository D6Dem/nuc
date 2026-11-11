"""Random-init core для smoke-теста. Не для реального использования."""
import argparse, numpy as np
from pathlib import Path
from .nblk import write_block


def make_core(path, d_model=64, n_layers=2, n_heads=4,
              vocab=512, d_ff=256):
    rng = np.random.default_rng(42)
    def r(*shape, scale=0.02):
        return (rng.standard_normal(shape) * scale).astype(np.float32)
    t = {"embed": r(vocab, d_model),
         "lm_head": r(vocab, d_model),
         "norm_out": np.ones(d_model, np.float32)}
    for L in range(n_layers):
        t[f"L{L}.wq"] = r(d_model, d_model)
        t[f"L{L}.wk"] = r(d_model, d_model)
        t[f"L{L}.wv"] = r(d_model, d_model)
        t[f"L{L}.wo"] = r(d_model, d_model)
        t[f"L{L}.ff1"] = r(d_ff, d_model)
        t[f"L{L}.ff2"] = r(d_model, d_ff)
        t[f"L{L}.norm1"] = np.ones(d_model, np.float32)
        t[f"L{L}.norm2"] = np.ones(d_model, np.float32)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    write_block(path, name="core-demo", kind="core",
                arch={"n_layers": n_layers, "n_heads": n_heads,
                      "d_model": d_model, "d_ff": d_ff,
                      "vocab_size": vocab},
                meta={"desc": "demo random-init", "tags": []},
                tensors=t)
    print(f"wrote {path}  ({sum(a.nbytes for a in t.values()) / 1024:.1f} KB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./blocks/core.nblk")
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    a = ap.parse_args()
    make_core(a.out, d_model=a.d, n_layers=a.layers)
