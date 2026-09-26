"""Correctness-gated CPU benchmark: python benchmarks/benchmark_p3109.py.

Install gfloat separately. No JIT, GPU, scalar-loop baseline, or setup costs are
hidden in one side only. Inputs are float64 and both sides return float64.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import gfloat
from gfloat.formats import format_info_p3109
from pychop import P3109, P3109Format


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 1024, 65536, 1000000])
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeat < 1 or any(n < 1 for n in args.sizes):
        parser.error("sizes and repeat must be positive")
    rng = np.random.default_rng(2026)
    rows = []
    for precision in (3, 4):
        fmt = P3109Format(8, precision)
        q = P3109(fmt)
        fi = format_info_p3109(8, precision)
        for distribution in ("normal", "wide"):
            for size in args.sizes:
                x = rng.normal(size=size)
                if distribution == "wide":
                    x *= np.exp2(rng.uniform(-25, 20, size=size))
                expected = gfloat.round_ndarray(fi, x)
                np.testing.assert_array_equal(q(x), expected)
                funcs = [lambda: q(x), lambda: gfloat.round_ndarray(fi, x)]
                for fn in funcs:
                    fn()
                loops = max(1, min(500, 100000 // size))
                times = [[], []]
                for repeat in range(args.repeat):
                    for index in ([0, 1] if repeat % 2 == 0 else [1, 0]):
                        start = time.perf_counter_ns()
                        for _ in range(loops):
                            funcs[index]()
                        times[index].append((time.perf_counter_ns() - start) / loops / 1e6)
                medians = [statistics.median(t) for t in times]
                row = dict(format=fmt.name, distribution=distribution, size=size,
                           pychop_ms=medians[0], gfloat_ms=medians[1],
                           speedup=medians[1] / medians[0], samples_ms=times)
                rows.append(row)
                print(f"{fmt.name} {distribution:6s} n={size:8d}: {row['speedup']:.2f}x "
                      f"({medians[0]:.4f} vs {medians[1]:.4f} ms)")
    report = dict(environment=dict(platform=platform.platform(), machine=platform.machine(),
                                   python=platform.python_version(), numpy=np.__version__,
                                   gfloat=version("gfloat")), seed=2026, repeats=args.repeat,
                  rounding="nearest_even", saturation=False, dtype="float64", results=rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    main()
