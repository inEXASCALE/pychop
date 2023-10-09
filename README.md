# pychop

### A python package for simulaing low precision floating point arithmetic in scientific computing

Using low precesion can gain extra speedup while resulting in less storage and energy cost.  The intention of ``pychop``, following the same function of ``chop`` in Matlab provided by Nick higham, is to simulate the low precision formats based on single and double precisions, which is pravalent on modern machine. 

This package provides consistent APIs to the original chop software as much as possible.   

## Install

``pychop`` has the following dependencies:

- numpy >=1.7.3

## References

Nicholas J. Higham and Srikara Pranesh, Simulating Low Precision Floating-Point Arithmetic, SIAM J. Sci. Comput., 41(4):A2536-A2551, 2019.
