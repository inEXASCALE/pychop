<div align="center">
<img src="docs/imgs/pychop_logo.png" width="330">

# Pychop: efficient reduced-precision quantization library 

[![All platforms](https://dev.azure.com/conda-forge/feedstock-builds/_apis/build/status/pychop-feedstock?branchName=main)]([https://anaconda.org/conda-forge/pychop](https://dev.azure.com/conda-forge/feedstock-builds/_build/latest?definitionId=26671&branchName=main))
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/pychop/badges/license.svg)](https://anaconda.org/conda-forge/pychop)
[![Codecov](https://github.com/inEXASCALE/pychop/actions/workflows/codecov.yml/badge.svg)](https://github.com/inEXASCALE/pychop/actions/workflows/codecov.yml) [![!pypi](https://img.shields.io/pypi/v/pychop?color=greem)](https://pypi.org/project/pychop/) [![Conda Version](https://img.shields.io/conda/vn/conda-forge/pychop.svg)](https://anaconda.org/conda-forge/pychop)
[![Download Status](https://static.pepy.tech/badge/pychop)](https://pypi.python.org/pypi/pychop/)
[![Download Status](https://img.shields.io/pypi/dm/pychop.svg?label=PyPI%20downloads)](https://pypi.org/project/pychop)
[![Documentation Status](https://readthedocs.org/projects/pychop/badge/?version=latest)](https://pychop.readthedocs.io/en/latest/?badge=latest)
[![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/pychop.svg)](https://anaconda.org/conda-forge/pychop)
</div>
      
Lower-precision floating-point arithmetic is becoming more common, moving beyond the usual IEEE 64-bit double-precision and 32-bit single-precision formats. Today, hardware accelerators and software simulations often use reduced-precision formats, such as 16-bit half-precision, which are popular in scientific computing and machine learning. These formats boost computational speed, reduce data transfer between memory and processors, and use less energy. These benefits are most important with large datasets or real-time applications.

``Pychop``brings these features to Python, inspired by MATLAB’s well-known chop function by Nick Higham. This Python library lets you quickly and reliably convert single- or double-precision numbers into any low-bitwidth format. It is flexible, so you can set up custom floating-point formats by choosing the number of exponent and significand bits, or pick fixed-point or integer quantization. This gives you control to match numerical precision and range to your algorithm, simulation, or hardware needs. It combines advanced features with ease of use. It includes many rounding modes, both deterministic and stochastic, and can handle denormal numbers as soft errors for accurate hardware emulation. The library is built for speed using vectorized operations for emulation. It also integrates directly with NumPy arrays, PyTorch tensors, and JAX arrays, so you can quantize data within your current workflow without extra conversions or performance loss.

``Pychop``lets you emulate low-precision arithmetic in a regular high-precision environment, so you do not need special hardware. This makes it easy to study how quantization affects stability, convergence, accuracy, and efficiency on your laptop or server. ``Pychop``works well for academic research needing careful control over numbers, and for software development where you want to quickly test different bit-widths to find the best balance between speed, memory use, and model quality. ``Pychop``offers a comprehensive solution.



## Install

The proper running environment of ``Pychop``  should by Python 3, which relies on the following dependencies: python > 3.8, numpy >=1.7.3, pandas >=2.0, torch, jax. 

To install the current current release via PIP manager use:

```Python
pip install pychop
```

Besides, one can install `pychop` from the `conda-forge` channel can be achieved by adding `conda-forge` to your channels with:

```
conda config --add channels conda-forge
conda config --set channel_priority strict
```

Once the `conda-forge` channel has been enabled, `pychop` can be installed with `conda`:

```
conda install pychop
```

or with `mamba`:

```
mamba install pychop
```

It is possible to list all of the versions of `pychop` available on your platform with `conda`:

```
conda search pychop --channel conda-forge
```

or with `mamba`:

```
mamba search pychop --channel conda-forge
```

## Features
The ``Pychop`` class offers several key advantages that make it a powerful tool for developers, researchers, and engineers working with numerical computations:

* Customizable Precision
* Multiple Rounding Modes
* Hardware-Independent Simulation
* Support for Denormal Numbers
* GPU Acceleration
* Reproducible Stochastic Rounding
* Ease of Integration
* Error Detection
* Soft error simulation

### The supported floating point formats

The supported floating point arithmetic formats include:

| format | description | bits |
| ------------- | ------------- | ------------- |
| 'q43', 'fp8-e4m3'         | NVIDIA quarter precision | 4 exponent bits, 3 significand  bits |
| 'q52', 'fp8-e5m2'         | NVIDIA quarter precision | 5 exponent bits, 2 significand bits |
|  'b', 'bfloat16'          | bfloat16 | 8 exponent bits, 7 significand bits  |
|  't', 'tf32'              | TensorFloat-32 | 8 exponent bits, 10 significand bits |
|  'h', 'half', 'fp16'      | IEEE half precision | 5 exponent bits, 10 significand bits  |
|  's', 'single', 'fp32'    | IEEE single precision |  8 exponent bits, 23 significand bits  |
|  'd', 'double', 'fp64'    | IEEE double precision | 11 exponent bits, 52 significand bits |
|  'c', 'custom'            | custom format | - - |



``Pychop`` support arbitrarily built-in reduced-precision types for scalar, array, and tensor. See here for detail [doc](https://pychop.readthedocs.io/en/latest/builtin.html). A simple example for scalar is as follows:

```python
from pychop import Chop
from pychop.builtin import CPFloat

half = Chop(exp_bits=5, sig_bits=10, subnormal=True, rmode=1)

a = CPFloat(1.234567, half)
b = CPFloat(0.987654, half)

print(a)                     # CPFloat(1.23438, prec=half)
c = a + b                    # stays a CPFloat, chopped
print(c)                     # CPFloat(2.22203, prec=half)
d = a * b / 2.0
print(d)                     # CPFloat(0.609863, prec=half)

# mixed with a normal Python float
e = a + 3.14
print(e)                     # CPFloat(4.37438, prec=half)
```

### Supported Microscaling (MX) formats

[Microscaling formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) use **block-level shared exponents** for extreme compression (2-4x vs FP16).

| format | description | element bits | block structure |
| ------------- | ------------- | ------------- | ------------- |
| 'mxfp8_e5m2'  | OCP MX FP8 E5M2 | 8 (5 exp + 2 sig) | 32 elements + E8M0 scale |
| 'mxfp8_e4m3'  | OCP MX FP8 E4M3 | 8 (4 exp + 3 sig) | 32 elements + E8M0 scale |
| 'mxfp6_e3m2'  | OCP MX FP6 E3M2 | 6 (3 exp + 2 sig) | 32 elements + E8M0 scale |
| 'mxfp6_e2m3'  | OCP MX FP6 E2M3 | 6 (2 exp + 3 sig) | 32 elements + E8M0 scale |
| 'mxfp4_e2m1'  | OCP MX FP4 E2M1 | 4 (2 exp + 1 sig) | 32 elements + E8M0 scale |
| custom MX     | user-defined    | any E/M combinati


**Key Features of MX Formats:**
- 🚀 **2-4x compression** vs FP16 while maintaining accuracy
- 🎯 **Block-level shared scale factor** (typically 32 elements per block)
- 🔧 **Fully customizable**: any `(exp_bits, sig_bits)` combination supported
- 📦 **Flexible block sizes**: 8, 16, 32, 64, 128, or custom
- ⚙️ **Adjustable scale precision**: E6M0, E8M0, E10M0, or custom


```python
from pychop.mx_formats import MXTensor, mx_quantize

# Predefined MX format
X_mx = mx_quantize(X, format='mxfp8_e4m3', block_size=32)

# Custom MX format (E5M4 elements)
mx_tensor = MXTensor(X, format=(5, 4), block_size=64)

# Custom with larger scale range
mx_tensor = MXTensor(X, format=(4, 3), scale_exp_bits=10, block_size=32)

# Ultra-low precision (3-bit elements!)
mx_tensor = MXTensor(X, format=(1, 1), block_size=16)
```


### Examples
We will go through the main functionality of ``Pychop``; for details refer to the documentation. 

#### (I). Floating point quantization
Users can specify the number of exponent (exp_bits) and significand (sig_bits) bits, enabling precise control over the trade-off between range and precision. 
For example, setting exp_bits=5 and sig_bits=4 creates a compact 10-bit format (1 sign, 5 exponent, 4 significand), ideal for testing minimal precision scenarios.

Rounding the values with specified precision format:

``Pychop`` supports faster low-precision floating point quantization and also enables GPU emulation (simply move the input to GPU device), with different rounding functions:

```Python
import pychop
from pychop import Chop
import numpy as np
np.random.seed(0)

X = np.random.randn(5000, 5000) 
pychop.backend('numpy', 1) # Specify different backends, e.g., jax and torch
# One can also specify 'auto', the pychop will automatically detect the types,
# but speed will be degraded. 
 
ch = Chop(exp_bits=5, sig_bits=10, rmode=3) # half precision
X_q = ch(X)
print(X_q[:10, 0])
```

If one is not seeking optimized performance and more emulation supports, one can use the following example. 

``Pychop`` also provides same functionalities just like Higham's chop [1] that supports soft error simulation (by setting ``flip=True``), but with relatively degraded speed:

```Python
from pychop import FaultChop

ch = FaultChop('h') # Standard IEEE 754 half precision
X_q = ch(X) # Rounding values
```

One can also customize the precision via:
```Python
from pychop import Customs
from pychop import FaultChop

pychop.backend('numpy', 1)
ct1 = Customs(exp_bits=5, sig_bits=10) # half precision (5 exponent bits, 10+(1) significand bits, (1) is implicit bits)

ch = FaultChop(customs=ct1, rmode=3) # Round towards minus infinity 
X_q = ch(X)
print(X_q[:10, 0])

ct2 = Customs(emax=15, t=11)
ch = FaultChop(customs=ct2, rmode=3)
X_q = ch(X)
print(X_q[:10, 0])
```


To enable quantization aware training, a sequential neural network can be built with derived quantied layer (seamlessly integrated with Straight-Through Estimator):

```Python
import torch.nn as nn
from pychop.layers import *

class MLP(nn.Module):
    def __init__(self, chop=None):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = QuantizedLinear(256, 256, chop=chop)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = QuantizedLinear(256, 10, chop=chop)
        # 5 exponent bits, 10 explicit significant bits , round to nearest ties to even

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

To enable quantization-aware training, one need to pass floating-point chopper ``ChopSTE`` or fixed-point chopper ``ChopfSTE`` to the parameter ``chop``, for details of example. we refer to [example_CNN_ft.py](examples/example_CNN_ft.py) and [example_CNN_fp.py](examples/example_CNN_fp.py)


For integer quantization, please see [example_CNN_int.py](examples/example_CNN_int.py).

#### (II). Fixed point quantization

Similar to floating point quantization, one can set the corresponding backend. The dominant parameters are ibits and fbits, which are the bitwidths of the integer part and the fractional part, respectively. 

```Python
pychop.backend('numpy')
from pychop import Chopf

ch = Chopf(ibits=4, fbits=4)
X_q = ch(X)
```


The code example can be found on the [guidance1](https://github.com/chenxinye/pychop/example/guidance1.ipynb) and [guidance2](https://github.com/chenxinye/pychop/example/guidance2.ipynb).

#### (III). Integer quantization

Integer quantization is another important feature of pychop. It intention is to convert the floating point number into a low bit-width integer, which speeds up the computations in certain computing hardware. It performs quantization with user-defined bitwidths. The following example illustrates the usage of the method.

The integer arithmetic emulation of ``Pychop`` is implemented by the interface Chopi. It can be used in many circumstances, and offers flexible options for users, such as symmetric or unsymmetric quantization and the number of bits to use. The usage is illustrated as below:


```Python
import numpy as np
from pychop import Chopi 
pychop.backend('numpy')

X = np.array([[0.1, -0.2], [0.3, 0.4]])
ch = Chopi(bits=8, symmetric=False)
X_q = ch.quantize(X) # Convert to integers
X_dq = ch.dequantize(X_q) # Convert back to floating points
```


### Call in MATlAB

If you use Python virtual environments in MATLAB, ensure MATLAB detects it:

```MATLAB
pe = pyenv('Version', 'your_env\python.exe'); % or simply pe = pyenv();
```

To use ``Pychop`` in your MATLAB environment, similarly, simply load the ``Pychop``module:

```MATLAB
pc = py.importlib.import_module('pychop');
ch = pc.Chop(exp_bits=5, sig_bits=10, rmode=1)
X = rand(100, 100);
X_q = ch(X);
```

Or more specifically, use
```MATLAB
np = py.importlib.import_module('numpy');
pc = py.importlib.import_module('pychop');
ch = pc.Chop(exp_bits=5, sig_bits=10, rmode=1)
X = np.random.randn(int32(100), int32(100));
X_q = ch(X);
```


### Use Cases
 
 * Machine Learning: Test the impact of low-precision arithmetic on model accuracy and training stability, especially for resource-constrained environments like edge devices.

 * Hardware Design: Simulate custom floating-point units before hardware implementation, optimizing bit allocations for specific applications.

 * Numerical Analysis: Investigate quantization errors and numerical stability in scientific computations.

 * Education: Teach concepts of floating-point representation, rounding, and denormal numbers with a hands-on, customizable tool.





## Contributing
Our software is licensed under License MIT. We welcome contributions in any form! Assistance with documentation is always welcome. To contribute, feel free to open an issue or please fork the project make your changes and submit a pull request. We will do our best to work through any issues and requests.


## Acknowledgement
This project is supported by the European Union (ERC, [InEXASCALE](https://www.karlin.mff.cuni.cz/~carson/inexascale), 101075632). Views and opinions
expressed are those of the authors only and do not necessarily reflect those of the European
 Union or the European Research Council. Neither the European Union nor the granting
 authority can be held responsible for them.

## Citations

If you use ``Pychop`` in your research or simulations, cite:

```bibtex
@misc{carson2025,
      title={Pychop: Emulating Low-Precision Arithmetic in Numerical Methods and Neural Networks}, 
      author={Erin Carson and Xinye Chen},
      year={2025},
      eprint={2504.07835},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.07835}, 
}
```


### References

[1] Nicholas J. Higham and Srikara Pranesh, Simulating Low Precision Floating-Point Arithmetic, SIAM J. Sci. Comput., 2019.

[2] IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2019 (revision of IEEE Std 754-2008), IEEE, 2019.

[3] Intel Corporation, BFLOAT16---hardware numerics definition,  2018

[4] Muller, Jean-Michel et al., Handbook of Floating-Point Arithmetic, Springer, 2018



[jax_link]: https://github.com/google/jax
[jax_badge_link]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq

































