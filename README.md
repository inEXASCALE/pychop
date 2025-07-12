# pychop: efficient low precision quantization library

[![Workflow for Codecov](https://github.com/inEXASCALE/pychop/actions/workflows/codecov.yml/badge.svg)](https://github.com/inEXASCALE/pychop/actions/workflows/codecov.yml)
[![!pypi](https://img.shields.io/pypi/v/pychop?color=greem)](https://pypi.org/project/pychop/)
[![Download Status](https://static.pepy.tech/badge/pychop)](https://pypi.python.org/pypi/pychop/)
[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://readthedocs.org/projects/pychop/badge/?version=latest)](https://pychop.readthedocs.io/en/latest/?badge=latest)


With the increasing availability of lower-precision floating-point arithmetic beyond IEEE 64-bit/32-bit precision, both in hardware and software simulation, reduced-precision formats such as 16-bit half precision have gained significant attention in scientific computing and machine learning domains. These formats offer higher computational throughput, reduced data transfer overhead, and lower energy consumption.

Inspired by MATLAB’s chop function by Nick Higham,  ``pychop`` is a Python library designed for efficient quantization, enabling the conversion of single- or double-precision numbers into low-bitwidth representations. It allows users to define custom floating-point formats with a specified number of exponent and significand bits as well as fixed-point and integer quantization, offering fine-grained control over precision and range. 
``pychop`` stands out for its versatility, efficiency, and ease of integration with NumPy, PyTorch, and JAX. Its key strengths—customizability, hardware independence, GPU support, and comprehensive rounding options—make it a valuable tool for both practical applications and theoretical exploration in numerical computing. The library supports multiple rounding modes, optional denormal number handling, and runs efficiently on both CPU and GPU devices. This makes it particularly useful for research, experimentation, and optimization in areas like machine learning, numerical analysis, and hardware design, where reduced precision can provide computational advantages. It also includes PyTorch and JAX backends, enabling low-precision neural network training simulations. By emulating low-precision formats within a high-precision environment (single or double), ``pychop`` allows users to analyze quantization effects without requiring specialized hardware. The library supports both deterministic and stochastic rounding strategies and is optimized for vectorized operations with NumPy arrays, PyTorch tensors, and JAX arrays.


## Install

The proper running environment of ``pychop``  should by Python 3, which relies on the following dependencies:
- python > 3.8  ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
- numpy >=1.7.3  ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
- pandas >=2.0 ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
- torch ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
- jax ![jax_badge][jax_badge_link]




To install the current current release via PIP use:

`pip install pychop`


## Features
The ``pychop`` class offers several key advantages that make it a powerful tool for developers, researchers, and engineers working with numerical computations:

* Customizable Precision
* Multiple Rounding Modes
* Hardware-Independent Simulation
* Support for Denormal Numbers
* GPU Acceleration
* Reproducible Stochastic Rounding
* Ease of Integration
* Error Detection

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


### Examples
We will go through the main functionality of ``pychop``; for details refer to the documentation. 

#### (I). Floating point quantization
Users can specify the number of exponent (exp_bits) and significand (sig_bits) bits, enabling precise control over the trade-off between range and precision. 
For example, setting exp_bits=5 and sig_bits=4 creates a compact 10-bit format (1 sign, 5 exponent, 4 significand), ideal for testing minimal precision scenarios.

Rounding the values with specified precision format:

``pychop`` supports faster low-precision floating point quantization and also enables GPU emulation (simply move the input to GPU device), with different rounding functions:

```Python
import pychop
from pychop import LightChop
import numpy as np
np.random.seed(0)

arr = np.random.randn(5000, 5000) 
pychop.backend('numpy', 1) # Specify different backends, e.g., jax and torch
 
ch = LightChop(exp_bits=5, sig_bits=10, rmode=3) # half precision
arr_q = ch(arr)
print(X_q[:10, 0])
```

If one is not seeking optimized performance and more emulation supports, one can use the following example. 

``pychop`` also provides same functionalities just like Higham's chop [1], but with relatively faster rounding:

```Python
from pychop import Chop

ch = Chop('h') # Standard IEEE 754 half precision
arr_q = ch(arr) # Rounding values
```

One can also customize the precision via:
```Python
from pychop import Customs
pychop.backend('numpy', 1)
ct1 = Customs(exp_bits=5, sig_bits=10) # half precision (5 exponent bits, 10+(1) significand bits, (1) is implicit bits)

ch = Chop(customs=ct1, rmode=3) # Round towards minus infinity 
arr_q = ch(arr)
print(arr_q[:10, 0])

ct2 = Customs(emax=15, t=11)
ch = Chop(customs=ct2, rmode=3)
arr_q = ch(arr)
print(arr_q[:10, 0])
```



Set quantized layer (seamlessly integrated with Straight-Through Estimator):

```Python
import torch
from pychop.layers import QuantizedLayer, 
layer = QuantizedLayer(exp_bits=5, sig_bits=10, rmode=1) # half precision, round to nearest ties to even
input_tensor = torch.randn(3, 4)
output = layer(input_tensor)
```

A sequential neural network can be built with:
```Python
import torch.nn as nn
from pychop.layers import *

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__(exp_bits=5, sig_bits=10, rmode=1)
        self.flatten = nn.Flatten()
        self.fc1 = QuantizedLinear(256, 256, exp_bits, sig_bits, rmode=rmode)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = QuantizedLinear(256, 10, exp_bits, sig_bits, rmode=rmode)
        # 5 exponent bits, 10 explicit significant bits , round to nearest ties to even

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```


Alternatively, one can seek a less strict simulation:

```Python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256, 256)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 10)
        self.quant = QuantizedLayer(exp_bits=5, sig_bits=10, rmode=1) 

    def forward(self, x):
        x = self.flatten(x)
        x = self.quant(self.fc1(x))
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.quant(self.fc2(x))
        return x
```


#### (II). Fixed point quantization

Similar to floating point quantization, one can set the corresponding backend. The dominant parameters are ibits and fbits, which are the bitwidths of the integer part and the fractional part, respectively. 

```Python
pychop.backend('numpy')
from pychop import Chopf

ch = Chopf(ibits=4, fbits=4)
arr_q = ch(arr)
```


The code example can be found on the [guidance1](https://github.com/chenxinye/pychop/example/guidance1.ipynb) and [guidance2](https://github.com/chenxinye/pychop/example/guidance2.ipynb).

#### (III). Integer quantization

Integer quantization is another important feature of pychop. It intention is to convert the floating point number into a low bit-width integer, which speeds up the computations in certain computing hardware. It performs quantization with user-defined bitwidths. The following example illustrates the usage of the method.

The integer arithmetic emulation of ``pychop`` is implemented by the interface Chopi. It can be used in many circumstances, and offers flexible options for users, such as symmetric or unsymmetric quantization and the number of bits to use. The usage is illustrated as below:


```Python
import numpy as np
from pychop import Chopi 
pychop.backend('numpy')

arr = np.array([[0.1, -0.2], [0.3, 0.4]])
ch = Chopi(num_bits=8, symmetric=False)
arr_q = ch.quantize(arr) # Convert to integers
arr_dq = ch.dequantize(arr_q) # Convert back to floating points
```


### Call in MATlAB

If you use Python virtual environments in MATLAB, ensure MATLAB detects it:

```MATLAB
pe = pyenv('Version', 'your_env\python.exe'); % or simply pe = pyenv();
```

To use ``Pychop`` in your MATLAB environment, similarly, simply load the Pychop module:

```MATLAB
pc = py.importlib.import_module('pychop');
ch = pc.LightChop(exp_bits=5, sig_bits=10, rmode=1)
arr = rand(100, 100);
arr_q = ch(arr);
```

Or more specifically, use
```MATLAB
np = py.importlib.import_module('numpy');
pc = py.importlib.import_module('pychop');
ch = pc.LightChop(exp_bits=5, sig_bits=10, rmode=1)
arr = np.random.randn(int32(100), int32(100));
arr_q = ch(arr);
```


### Use Cases
 
 * Machine Learning: Test the impact of low-precision arithmetic on model accuracy and training stability, especially for resource-constrained environments like edge devices.

 * Hardware Design: Simulate custom floating-point units before hardware implementation, optimizing bit allocations for specific applications.

 * Numerical Analysis: Investigate quantization errors and numerical stability in scientific computations.

 * Education: Teach concepts of floating-point representation, rounding, and denormal numbers with a hands-on, customizable tool.








## Contributing
We welcome contributions in any form! Assistance with documentation is always welcome. To contribute, feel free to open an issue or please fork the project make your changes and submit a pull request. We will do our best to work through any issues and requests.


## Acknowledgement
This project is supported by the European Union (ERC, [InEXASCALE](https://www.karlin.mff.cuni.cz/~carson/inexascale), 101075632). Views and opinions
 expressed are those of the authors only and do not necessarily reflect those of the European
 Union or the European Research Council. Neither the European Union nor the granting
 authority can be held responsible for them.


### References

[1] Nicholas J. Higham and Srikara Pranesh, Simulating Low Precision Floating-Point Arithmetic, SIAM J. Sci. Comput., 2019.

[2] IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2019 (revision of IEEE Std 754-2008), IEEE, 2019.

[3] Intel Corporation, BFLOAT16---hardware numerics definition,  2018

[4] Muller, Jean-Michel et al., Handbook of Floating-Point Arithmetic, Springer, 2018



[jax_link]: https://github.com/google/jax
[jax_badge_link]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq
