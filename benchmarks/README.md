# Reproducing the benchmarks

See [Validation and performance](../docs/source/validation.rst) for the benchmark
commands, input distributions, timing method and scope of the recorded results.
The JSON files in this directory record the measured environment and samples.

For the recorded macOS Torch Inductor run, the benchmark process used Apple
`clang++`, macOS SDK C++ headers, and an `OMP_PREFIX` pointing to the OpenMP
library already loaded by Torch. If Torch compilation fails on macOS, check the
C++ toolchain and OpenMP library paths before interpreting the failure as a
quantization error. Eager tensor quantization does not require a C++ compiler.
