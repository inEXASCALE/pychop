"""Native CPU tensor benchmark with synchronized timings and exact result checks.

Use --backend torch/jax/tensorflow. --compiled enables torch.compile (Inductor),
jax.jit or tf.function. Compilation and warm-up are excluded for both candidates.
TensorFlow has no gfloat baseline because gfloat's array API does not support it.
"""
import argparse
import json
from importlib.metadata import version
import platform
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np
from pychop import P3109


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["torch", "jax", "tensorflow"], required=True)
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.size < 1 or args.repeat < 1:
        parser.error("size and repeat must be positive")
    data = np.random.default_rng(2026).normal(size=args.size).astype(np.float32)
    q = P3109()
    if args.backend == "torch":
        import torch as lib
        x = lib.tensor(data)
        compile_fn = lambda fn: lib.compile(fn, fullgraph=True)
        sync = lambda a: a.detach().cpu().numpy()
    elif args.backend == "tensorflow":
        import tensorflow as lib
        lib.config.set_visible_devices([], "GPU")
        x = lib.constant(data)
        compile_fn = lambda fn: lib.function(fn, autograph=False)
        sync = lambda a: a.numpy()
    else:
        import jax as lib
        x = lib.device_put(data, lib.devices("cpu")[0])
        compile_fn = lib.jit
        sync = lambda a: np.asarray(a.block_until_ready())
    candidates = {"pychop": q}
    if args.backend != "tensorflow":
        import gfloat
        from gfloat.formats import format_info_p3109
        fi = format_info_p3109(8, 4)
        candidates["gfloat"] = lambda a: gfloat.round_ndarray(fi, a)
    compiled = {}
    for name, fn in candidates.items():
        compiled[name] = compile_fn(fn) if args.compiled else fn
        np.testing.assert_array_equal(sync(compiled[name](x)), q(data))
        sync(compiled[name](x))
    timings = {name: [] for name in candidates}
    names = list(candidates)
    for i in range(args.repeat):
        for name in names[::1 if i % 2 == 0 else -1]:
            start = perf_counter()
            sync(compiled[name](x))
            timings[name].append((perf_counter() - start) * 1000)
    medians = {name: median(samples) for name, samples in timings.items()}
    result = {"backend": args.backend, "version": lib.__version__, "platform": platform.platform(),
              "gfloat_version": version("gfloat") if "gfloat" in candidates else None,
              "compiled": args.compiled, "device": "CPU", "dtype": "float32", "size": args.size,
              "rounding": "nearest_even", "format": "p3109_k8p4se", "samples_ms": timings,
              "median_ms": medians, "speedup": medians["gfloat"] / medians["pychop"] if "gfloat" in medians else None}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
