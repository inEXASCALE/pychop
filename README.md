# pychop

[![!pypi](https://img.shields.io/pypi/v/pychop?color=yellowgreen)](https://pypi.org/project/pychop/)
[![Download Status](https://static.pepy.tech/badge/pychop)](https://pypi.org/project/pychop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

### A python package for simulaing low precision floating point arithmetic in scientific computing

Using low precesion can gain extra speedup while resulting in less storage and energy cost.  The intention of ``pychop``, following the same function of ``chop`` in Matlab provided by Nick higham, is to simulate the low precision formats based on single and double precisions, which is pravalent on modern machine. 


The supported rounding modes include:

1. Round to nearest using round to even last bit to break ties
  (the default).

2. Round towards plus infinity (round up).

3. Round towards minus infinity (round down).

4. Round towards zero.

5. Stochastic rounding - round to the next larger or next smaller
  floating-point number with probability proportional to
  the distance to those floating-point numbers.

6. Stochastic rounding - round to the next larger or next smaller 
  floating-point number with equal probability.

Subnormal numbers is supported, they are flushed to zero if it not considered (by setting `subnormal` to 0).

The supported floating point arithmetic formats include:

| format | description |
| ------------- | ------------- |
| 'q43', 'fp8-e4m3'         | NVIDIA quarter precision (4 exponent bits, 3 significand (mantissa) bits) |
| 'q52', 'fp8-e5m2'         | NVIDIA quarter precision (5 exponent bits, 2 significand bits) |
|  'b', 'bfloat16'          | bfloat16 |
|  'h', 'half', 'fp16'      | IEEE half precision (the default) |
|  's', 'single', 'fp32'    | IEEE single precision |
|  'd', 'double', 'fp64'    | IEEE double precision |
|  'c', 'custom'            | custom format |



This package provides consistent APIs to the chop software by Nick higham as much as possible.  For the first four rounding mode,  with the same user-specific parameters, ``pychop`` generates exactly same result as that of the chop software. For stochastic rounding (``rmode`` as 5 and 6), both output same results if random numbers is given the same. 

## Install

``pychop`` has the only following dependency:

- numpy >=1.7.3

To install the current ``pychop`` release via PIP use:

```pip install pychop```

To check the ``pychop`` installation use:

```python -m pip show pychop```


## Usage

After installing ``pychop``, load the pacakge by ```import pychop``` (of course you can use ```from pychop import chop``` for saving a lot code).

We use an simple example to illustrate its usage:


```Python
x = np.array([0.07630829, 0.77991879, 0.43840923, 0.72346518, 0.97798951]); 

cp = pychop.chop(prec='h', rmode=3) # use chop(prec='h', rmode=3) if you load with from pychop import chop
c = cp.chop(x)
print(c)
```

Output [0.07629395 0.77978516 0.43823242 0.72314453 0.97753906].


## References

Nicholas J. Higham and Srikara Pranesh, Simulating Low Precision Floating-Point Arithmetic, SIAM J. Sci. Comput., 41(4):A2536-A2551, 2019.
