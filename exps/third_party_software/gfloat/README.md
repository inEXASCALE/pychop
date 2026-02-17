<!-- Copyright (c) 2024 Graphcore Ltd. All rights reserved. -->

# gfloat: Generic floating-point types in Python

An implementation of generic floating point encode/decode logic,
handling various current and proposed floating point types:

 - [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754): Binary16, Binary32
 - [OCP Float8](https://www.opencompute.org/documents/ocp-8-bit-floating-point-specification-ofp8-revision-1-0-2023-06-20-pdf): E5M2, E4M3
 - [IEEE WG P3109](https://github.com/P3109/Public/blob/main/IEEE%20WG%20P3109%20Interim%20Report%20(latest).pdf): P3109_{K}p{P} for K > 2, and 1 <= P < K.
 - [OCP MX Formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf): E2M1, M2M3, E3M2, E8M0, INT8, and the MX block formats.

The library favours readability and extensibility over speed (although the *_ndarray functions are reasonably fast for large arrays, see the [benchmarking  notebook](docs/source/04-benchmark.ipynb)).
For other implementations of these datatypes more focused on speed see, for example, [ml_dtypes](https://github.com/jax-ml/ml_dtypes),
[bitstring](https://github.com/scott-griffiths/bitstring),
[MX PyTorch Emulation Library](https://github.com/microsoft/microxcaling).

See https://gfloat.readthedocs.io for documentation, or dive into the notebooks to explore the formats.

For example, here's a table from the [02-value-stats](docs/source/02-value-stats.ipynb) notebook:

|name|B: Bits in the format|P: Precision in bits|E: Exponent field width in bits|0<x<1|1<x<Inf|minSubnormal|maxSubnormal|minNormal|maxNormal|Exact in float16?|Exact in float32?|
|--------------|-----|-----|-----|-------|-------|----------------|----------------|-------------|---------------|--------|--------|
| p3109_k3p2sf |   3 |   2 |   1 |     1 |     1 | 0.5            | 0.5            | 1           | 1.5           | True   | True   |
| ocp_e2m1     |   4 |   2 |   2 |     1 |     5 | 0.5            | 0.5            | 1           | 6             | True   | True   |
| p3109_k4p2sf |   4 |   2 |   2 |     3 |     3 | 0.25           | 0.25           | 0.5         | 3             | True   | True   |
| ocp_e2m3     |   6 |   4 |   2 |     7 |    23 | 0.125          | 0.875          | 1           | 7.5           | True   | True   |
| ocp_e3m2     |   6 |   3 |   3 |    11 |    19 | 0.0625         | 0.1875         | 0.25        | 28            | True   | True   |
| p3109_k6p3sf |   6 |   3 |   3 |    15 |    15 | 0.03125        | 0.09375        | 0.125       | 14            | True   | True   |
| p3109_k6p4sf |   6 |   4 |   2 |    15 |    15 | 0.0625         | 0.4375         | 0.5         | 3.75          | True   | True   |
| ocp_e4m3     |   8 |   4 |   4 |    55 |    70 | 2^-9           | 7/4*2^-7       | 0.015625    | 448           | True   | True   |
| ocp_e5m2     |   8 |   3 |   5 |    59 |    63 | 2^-16          | 3/2*2^-15      | 2^-14       | 57344         | True   | True   |
| p3109_k8p1se |   8 |   1 |   7 |    63 |    62 | n/a            | n/a            | 2^-63       | 2^62          | False  | True   |
| p3109_k8p1ue |   8 |   1 |   8 |   127 |   125 | n/a            | n/a            | 2^-127      | 2^125         | False  | True   |
| p3109_k8p3se |   8 |   3 |   5 |    63 |    62 | 2^-17          | 3/2*2^-16      | 2^-15       | 49152         | True   | True   |
| p3109_k8p3sf |   8 |   3 |   5 |    63 |    63 | 2^-17          | 3/2*2^-16      | 2^-15       | 57344         | True   | True   |
| p3109_k8p3ue |   8 |   3 |   6 |   127 |   125 | 2^-33          | 3/2*2^-32      | 2^-31       | 5/4*2^31      | False  | True   |
| p3109_k8p3uf |   8 |   3 |   6 |   127 |   126 | 2^-33          | 3/2*2^-32      | 2^-31       | 3/2*2^31      | False  | True   |
| p3109_k8p4se |   8 |   4 |   4 |    63 |    62 | 2^-10          | 7/4*2^-8       | 0.0078125   | 224           | True   | True   |
| p3109_k8p4sf |   8 |   4 |   4 |    63 |    63 | 2^-10          | 7/4*2^-8       | 0.0078125   | 240           | True   | True   |
| p3109_k8p4ue |   8 |   4 |   5 |   127 |   125 | 2^-18          | 7/4*2^-16      | 2^-15       | 53248         | True   | True   |
| p3109_k8p4uf |   8 |   4 |   5 |   127 |   126 | 2^-18          | 7/4*2^-16      | 2^-15       | 57344         | True   | True   |
| p3109_k8p7sf |   8 |   7 |   1 |    63 |    63 | 0.015625       | 63/32*2^-1     | 1           | 127/64*2^0    | True   | True   |
| p3109_k8p8uf |   8 |   8 |   1 |   127 |   126 | 0.0078125      | 127/64*2^-1    | 1           | 127/64*2^0    | True   | True   |
| binary16     |  16 |  11 |   5 | 15359 | 16383 | 2^-24          | 1023/512*2^-15 | 2^-14       | 65504         | True   | True   |
| bfloat16     |  16 |   8 |   8 | 16255 | 16383 | 2^-133         | 127/64*2^-127  | 2^-126      | 255/128*2^127 | False  | True   |
| ocp_e8m0     |   8 |   1 |   8 |   127 |   127 | n/a            | n/a            | 2^-127      | 2^127         | False  | True   |
| ocp_int8     |   8 |   8 |   0 |    63 |    63 | 0.015625       | 127/64*2^0     | n/a         | n/a           | True   | True   |

#### Notes

All NaNs are the same, with no distinction between signalling or quiet,
or between differently encoded NaNs.
