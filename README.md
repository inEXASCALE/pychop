<div align="center">
<img src="docs/imgs/pychop_logo.png" width="330">

# pychop: efficient reduced-precision quantization library 

[![PyPI](https://img.shields.io/pypi/v/pychop?color=3776AB&logo=pypi&logoColor=white)](https://pypi.org/project/pychop/)
[![Python](https://img.shields.io/pypi/pyversions/pychop?logo=python&logoColor=white)](https://pypi.org/project/pychop/)
[![Tests](https://github.com/inEXASCALE/pychop/actions/workflows/numerical-core.yml/badge.svg)](https://github.com/inEXASCALE/pychop/actions/workflows/numerical-core.yml)
[![Docs](https://readthedocs.org/projects/pychop/badge/?version=latest)](https://pychop.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22A06B.svg)](LICENSE)
[![Conda](https://img.shields.io/conda/vn/conda-forge/pychop?logo=anaconda)](https://anaconda.org/conda-forge/pychop)
[![P3109](https://img.shields.io/badge/P3109-draft%20emulation-6F42C1)](docs/source/p3109.rst)
</div>
      
Lower-precision floating-point arithmetic is becoming more often used in recently growing hardware, moving beyond the usual IEEE 64-bit double-precision and 32-bit single-precision formats. Today, hardware accelerators and software simulations often, in one way or another, use reduced-precision formats, such as 16-bit half-precision (e.g., brain floating point), which are widely used in scientific computing and deep learning applications. These formats, if used properly, improve speed performance, reduce data transfer between memory and processors, and use less energy, while retaining the target accuracy. These benefits are most important with large datasets or real-time applications. 
However, one never realizes how much reduced precision is needed in their applications or certain computational steps, unless via rigorous error analysis, a program precision tuning tool, or trial and error. 

Inspired by MATLAB’s well-known chop function by Nick Higham, to support both practical and theoretical floating-point analysis, ``pychop`` features efficient low-precision emulation for Python, with an easy extension to MATLAB, without relying on hardware. This library lets you quickly and reliably convert single- or double-precision numbers into any low-bitwidth format. It is flexible, so you can set up custom floating-point formats by choosing the number of exponent and significand bits, or pick fixed-point or integer quantization. This gives you control to match numerical precision and range to your algorithm, simulation, or hardware needs. It combines advanced features with ease of use. It includes many rounding modes, both deterministic and stochastic, and supports subnormal numbers and, separately, optional bit-flip fault injection. The library is built for speed using vectorized operations for emulation. It also integrates directly with NumPy arrays, PyTorch tensors, and JAX arrays, so you can quantize data within your current workflow through backend-specific implementations.

``pychop`` enables one to emulate low-precision arithmetic in a regular high-precision environment, so you do not need special hardware. This makes it easy to study how quantization affects stability, convergence, accuracy, and efficiency on your laptop or server. ``pychop``works well for academic research needing careful control over numbers, and for software development where you want to quickly test different bit-widths to find the best balance between speed, memory use, and model quality. ``pychop``offers a comprehensive solution.







## Install

``pychop`` requires Python >= 3.8. Core dependencies include NumPy >= 1.17.3, pandas, SciPy >= 1.0, scikit-learn >= 0.20, and ``dask[array]``. PyTorch, JAX, and TensorFlow are optional backends and can be installed separately when needed. 

To install the current release with pip, use:

```Python
pip install pychop
```

Alternatively, to install `pychop` from the `conda-forge` channel, first add `conda-forge` to your channels with:

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

## P3109 emulation and parameter export

The local source includes NumPy, PyTorch, TensorFlow and JAX emulation of the **P3109 v4.1 public draft**,
with validated formats, nine rounding modes, three saturation policies, integer
code export and reproducible explicit-bit stochastic rounding. P3109 is an
unapproved draft; this is not a formal IEEE conformance claim. Install the checkout
with `python -m pip install -e .` to use these additions.

```python
import numpy as np
from pychop import P3109, P3109Format

q = P3109(P3109Format(k=8, precision=4), saturate=True)
weights = np.array([[1.1, -0.3], [0.25, 0.9]])
x = np.array([0.7, -0.2])
codes = q.encode(weights)
restored = P3109.from_dict(q.to_dict())
# Host matrix accumulation, followed by explicit output rounding.
prediction = restored(restored.decode(codes) @ restored(x))
print("prediction:", prediction)
print("weight codes:", codes)
print("exportable policy:", q.to_dict())
```

- [P3109 guide](docs/source/p3109.rst): format semantics, encode/decode, rounding and parameter export.
- [NumPy quickstart](examples/p3109/quickstart.py): rounding, code round trips, policy export and a quantized linear model.
- [Native tensor training](examples/p3109/tensor_training.py): Torch, TensorFlow and JAX STE training with portable parameter export.
- [Architecture](docs/source/architecture.rst) and [validation](docs/source/validation.rst): numerical contracts, test coverage and reproducible gfloat benchmarks.

```bash
python examples/p3109/quickstart.py
python -m pip install gfloat
python benchmarks/benchmark_p3109.py --output /tmp/p3109-benchmark.json
```

The [recorded CPU benchmark](benchmarks/p3109_macos_arm64.json) compares identical
float64 results against `gfloat.round_ndarray`, with every measured case faster.
See the validation guide for scope and timing methodology. Emulation normally
retains host storage and host arithmetic; it does not automatically accelerate
model inference. Explicit `encode` produces integer storage codes.

## Features
The ``pychop`` library offers several key features for developers, researchers, and engineers working with numerical computations:

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



``pychop`` supports built-in and customizable reduced-precision types for scalars, arrays, and tensors. See the [documentation](https://pychop.readthedocs.io/en/latest/builtin.html) for details. A simple scalar example is as follows:

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

### Supported microscaling (MX) formats

[Microscaling formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) use a **block-level shared scale** together with low-bit-width element formats. For the OCP-defined formats below, 32 elements share one E8M0 scale. When physically encoded and packed, this representation can reduce storage relative to FP16; PyChop itself emulates the numerical behavior rather than storing tensors in a packed MX representation.

| format | description | element bits | block structure |
| ------------- | ------------- | ------------- | ------------- |
| 'mxfp8_e5m2'  | OCP MX FP8 E5M2 | 8 (1 sign + 5 exp + 2 sig) | 32 elements + E8M0 scale |
| 'mxfp8_e4m3'  | OCP MX FP8 E4M3 | 8 (1 sign + 4 exp + 3 sig) | 32 elements + E8M0 scale |
| 'mxfp6_e3m2'  | OCP MX FP6 E3M2 | 6 (1 sign + 3 exp + 2 sig) | 32 elements + E8M0 scale |
| 'mxfp6_e2m3'  | OCP MX FP6 E2M3 | 6 (1 sign + 2 exp + 3 sig) | 32 elements + E8M0 scale |
| 'mxfp4_e2m1'  | OCP MX FP4 E2M1 | 4 (1 sign + 2 exp + 1 sig) | 32 elements + E8M0 scale |
| 'mxint8'       | OCP MX INT8 | 8 (two's-complement integer) | 32 elements + E8M0 scale |
| custom MX     | user-defined MX-style emulation | 1 + E + M | configurable block and scale widths |


**Key Features of MX Formats:**
- 🎯 **Block-level shared scale factor**: OCP-defined formats use 32 elements per block with an E8M0 scale
- 🔧 **Custom MX-style emulation**: user-defined `(exp_bits, sig_bits)` combinations are supported, but such combinations are not necessarily OCP-standard formats
- 📦 **Configurable block size** for custom emulation; use 32 for the OCP-defined formats above
- ⚙️ **Configurable scale exponent width** via `scale_exp_bits`; OCP-defined formats use E8M0


```python
from pychop.mx_formats import MXTensor, mx_quantize

# Predefined MX format
X_mx = mx_quantize(X, format='mxfp8_e4m3', block_size=32)

# Custom MX-style format (E5M4 elements; non-standard)
mx_tensor = MXTensor(X, format=(5, 4), block_size=64)

# Custom format with a larger shared-scale exponent range (non-standard)
mx_tensor = MXTensor(X, format=(4, 3), scale_exp_bits=10, block_size=32)

# Ultra-low-precision custom format: 3-bit elements (non-standard)
mx_tensor = MXTensor(X, format=(1, 1), block_size=16)
```

`mx_quantize` returns quantized-dequantized values in the same backend family as the input. `MXTensor.statistics()` reports format-level encoded-size/compression estimates; these are not measurements of the in-memory size of the emulation tensor.

### Examples
We will go through the main functionality of ``pychop``; for details refer to the documentation. 

#### (I). Floating point quantization
Users can specify the number of exponent (exp_bits) and significand (sig_bits) bits, enabling precise control over the trade-off between range and precision. 
For example, setting exp_bits=5 and sig_bits=4 emulates a 10-bit format (1 sign, 5 exponent, 4 significand), which is useful for testing minimal-precision scenarios.

Rounding the values with specified precision format:

``pychop`` supports efficient low-precision floating-point quantization with multiple rounding modes and can use GPU-backed PyTorch or JAX arrays/tensors for emulation:

```Python
import pychop
from pychop import Chop
import numpy as np
np.random.seed(0)

X = np.random.randn(5000, 5000) 
pychop.backend('numpy', 1) # Specify different backends, e.g., jax and torch
backend = pychop.get_backend() # you can also get current backend via .get_backend()
# One can also specify 'auto', the pychop will automatically detect the types,
# and reuse the selected implementation on repeated calls.
# For other backends, e.g., the ``torch`` backend, the input must be consistent array type, e.g., X = torch.from_numpy(X) # torch array
 
ch = Chop(exp_bits=5, sig_bits=10, rmode=3) # half precision
X_q = ch(X)
print(X_q[:10, 0])
```

For a more feature-complete research interface, including soft-error simulation, use ``FaultChop``. 

``FaultChop`` follows the style of Higham's chop [1] and supports soft-error simulation by setting ``flip=True``; this interface may be slower than the optimized quantization path:

```Python
from pychop import FaultChop

ch = FaultChop('h', flip=True) # IEEE 754 half precision with soft-error simulation enabled
X_q = ch(X) # Rounding values
```

One can also customize the precision via:
```Python
from pychop import Customs
from pychop import FaultChop

pychop.backend('numpy', 1)
ct1 = Customs(exp_bits=5, sig_bits=10) # half precision: 5 exponent bits and 10 stored significand bits (plus the implicit leading bit)

ch = FaultChop(customs=ct1, rmode=3) # Round towards minus infinity 
X_q = ch(X)
print(X_q[:10, 0])

ct2 = Customs(emax=15, t=11)
ch = FaultChop(customs=ct2, rmode=3)
X_q = ch(X)
print(X_q[:10, 0])
```


To enable quantization-aware training, a sequential neural network can be built with quantized layers that integrate a Straight-Through Estimator (STE):

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

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
```

To enable quantization-aware training, pass the floating-point chopper ``ChopSTE`` or fixed-point chopper ``ChopfSTE`` to the ``chop`` parameter. For examples, see [example_CNN_ft.py](examples/example_CNN_ft.py) and [example_CNN_fp.py](examples/example_CNN_fp.py).


For integer quantization, please see [example_CNN_int.py](examples/example_CNN_int.py).

#### (II). Fixed point quantization

Similar to floating point quantization, one can set the corresponding backend. The dominant parameters are ibits and fbits, which are the bitwidths of the integer part and the fractional part, respectively. 

```Python
pychop.backend('numpy')
from pychop import Chopf

ch = Chopf(ibits=4, fbits=4)
X_q = ch(X)
```


See the [fixed-point documentation](docs/source/fix_point.rst) for additional details.

#### (III). Integer quantization

Integer quantization is another important feature of ``pychop``. Its purpose is to quantize floating-point values to low-bit-width integers, which can enable faster computation on hardware with suitable integer support. It supports user-defined bit-widths. The following example illustrates its use.

Integer quantization in ``pychop`` is provided by the ``Chopi`` interface. It supports options such as symmetric or asymmetric quantization and a user-defined bit-width. Its usage is illustrated below:


```Python
import numpy as np
from pychop import Chopi 
pychop.backend('numpy')

X = np.array([[0.1, -0.2], [0.3, 0.4]])
ch = Chopi(bits=8, symmetric=False)
X_q = ch.quantize(X) # Convert to integers
X_dq = ch.dequantize(X_q) # Convert back to floating points
```


### Call in MATLAB

If you use Python virtual environments in MATLAB, ensure MATLAB detects it:

```MATLAB
pe = pyenv('Version', 'your_env\python.exe'); % or simply pe = pyenv();
```

To use ``pychop`` from MATLAB, import the ``pychop`` Python module:

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


### Use cases
 
 * Machine Learning: Test the impact of low-precision arithmetic on model accuracy and training stability, especially for resource-constrained environments like edge devices.

 * Hardware Design: Simulate custom floating-point units before hardware implementation, optimizing bit allocations for specific applications.

 * Numerical Analysis: Investigate quantization errors and numerical stability in scientific computations.

 * Education: Teach concepts of floating-point representation, rounding, and denormal numbers with a hands-on, customizable tool.





## Contributing
Our software is licensed under the MIT License. We welcome contributions in any form! Assistance with documentation is always welcome. To contribute, feel free to open an issue or fork the project, make your changes, and submit a pull request. We will do our best to work through any issues and requests.


## Acknowledgement
This project is supported by the European Union (ERC, [InEXASCALE](https://www.karlin.mff.cuni.cz/~carson/inexascale), 101075632). Views and opinions
expressed are those of the authors only and do not necessarily reflect those of the European
 Union or the European Research Council. Neither the European Union nor the granting
 authority can be held responsible for them.

## Citations

If you use ``pychop`` in your research or simulations, cite:

```bibtex
@article{carson2025pychop,
  title   = {{pychop}: Emulating Low-Precision Arithmetic in Numerical Methods and Neural Networks},
  author  = {Carson, Erin and Chen, Xinye},
  journal = {ACM Transactions on Mathematical Software},
  year    = {2025},
  note    = {Accepted for publication},
  eprint  = {2504.07835},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2504.07835}
}
```


### References

[1] Nicholas J. Higham and Srikara Pranesh, Simulating Low Precision Floating-Point Arithmetic, SIAM J. Sci. Comput., 2019.

[2] IEEE Standard for Floating-Point Arithmetic, IEEE Std 754-2019 (revision of IEEE Std 754-2008), IEEE, 2019.

[3] Intel Corporation, BFLOAT16---hardware numerics definition,  2018

[4] Muller, Jean-Michel et al., Handbook of Floating-Point Arithmetic, Springer, 2018



[jax_link]: https://github.com/google/jax
[jax_badge_link]: https://img.shields.io/badge/JAX-Accelerated-9cf.svg?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC0AAAAaCAYAAAAjZdWPAAAIx0lEQVR42rWWBVQbWxOAkefur%2B7u3les7u7F3ZIQ3N2tbng8aXFC0uAuKf2hmlJ3AapIgobMv7t0w%2Ba50JzzJdlhlvNldubeq%2FY%2BXrTS1z%2B6sttrKfQOOY4ns13ecFImb47pVvIkukNe4y3Junr1kSZ%2Bb3Na248tx7rKiHlPo6Ryse%2F11NKQuk%2FV3tfL52yHtXm8TGYS1wk4J093wrPQPngRJH9HH1x2fAjMhcIeIaXKQCmd2Gn7IqSvG83BueT0CMkTyESUqm3vRRggTdOBIb1HFDaNl8Gdg91AFGkO7QXe8gJInpoDjEXC9gbhtWH3rjZ%2F9yK6t42Y9zyiC1iLhZA8JQe4eqKXklrJF0MqfPv2bc2wzPZjpnEyMEVlEZCKQzYCJhE8QEtIL1RaXEVFEGmEaTn96VuLDzWflLFbgvqUec3BPVBmeBnNwUiakq1I31UcPaTSR8%2B1LnditsscaB2A48K6D9SoZDD2O6bELvA0JGhl4zIYZzcWtD%2BMfdvdHNsDOHciXwBPN18lj7sy79qQCTNK3nxBZXakqbZFO2jHskA7zBs%2BJhmDmr0RhoadIZjYxKIVHpCZngPMZUKoQKrfEoz1PfZZdKAe2CvP4XnYE8k2LLMdMumwrLaNlomyVqK0UdwN%2BD7AAz73dYBpPg6gPiCN8TXFHCI2s7AWYesJgTabD%2FS5uXDTuwVaAvvghncTdk1DYGkL0daAs%2BsLiutLrn0%2BRMNXpunC7mgkCpshfbw4OhrUvMkYo%2F0c4XtHS1waY4mlG6To8oG1TKjs78xV5fAkSgqcZSL0GoszfxEAW0fUludRNWlIhGsljzVjctr8rJOkCpskKaDYIlgkVoCmF0kp%2FbW%2FU%2F%2B8QNdXPztbAc4kFxIEmNGwKuI9y5gnBMH%2BakiZxlfGaLP48kyj4qPFkeIPh0Q6lt861zZF%2BgBpDcAxT3gEOjGxMDLQRSn9XaDzPWdOstkEN7uez6jmgLOYilR7NkFwLh%2B4G0SQMnMwRp8jaCrwEs8eEmFW2VsNd07HQdP4TgWxNTYcFcKHPhRYFOWLfJJBE5FefTQsWiKRaOw6FBr6ob1RP3EoqdbHsWFDwAYvaVI28DaK8AHs51tU%2BA3Z8CUXvZ1jnSR7SRS2SnwKw4O8B1rCjwrjgt1gSrjXnWhBxjD0Hidm4vfj3e3riUP5PcUCYlZxsYFDK41XnLlUANwVeeILFde%2BGKLhk3zgyZNeQjcSHPMEKSyPPQKfIcKfIqCf8yN95MGZZ1bj98WJ%2BOorQzxsPqcYdX9orw8420jBQNfJVVmTOStEUqFz5dq%2F2tHUY3LbjMh0qYxCwCGxRep8%2FK4ZnldzuUkjJLPDhkzrUFBoHYBjk3odtNMYoJVGx9BG2JTNVehksmRaGUwMbYQITk3Xw9gOxbNoGaA8RWjwuQdsXdGvpdty7Su2%2Fqn0qbzWsXYp0nqVpet0O6zzugva1MZHUdwHk9G8aH7raHua9AIxzzjxDaw4w4cpvEQlM84kwdI0hkpsPpcOtUeaVM8hQT2Qtb4ckUbaYw4fXzGAqSVEd8CGpqamj%2F9Q2pPX7miW0NlHlDE81AxLSI2wyK6xf6vfrcgEwb0PAtPaHM1%2BNXzGXAlMRcUIrMpiE6%2Bxv0cyxSrC6FmjzvkWJE3OxpY%2BzmpsANFBxK6RuIJvXe7bUHNd4zfCwvPPh9unSO%2BbIL2JY53QDqvdbsEi2%2BuwEEHPsfFRdOqjHcjTaCLmWdBewtKzHEwKZynSGgtTaSqx7dwMeBLRhR1LETDhu76vgTFfMLi8zc8F7hoRPpAYjAWCp0Jy5dzfSEfltGU6M9oVCIATnPoGKImDUJNfK0JS37QTc9yY7eDKzIX5wR4wN8RTya4jETAvZDCmFeEPwhNXoOlQt5JnRzqhxLZBpY%2BT5mZD3M4MfLnDW6U%2Fy6jkaDXtysDm8vjxY%2FXYnLebkelXaQtSSge2IhBj9kjMLF41duDUNRiDLHEzfaigsoxRzWG6B0kZ2%2BoRA3dD2lRa44ZrM%2FBW5ANziVApGLaKCYucXOCEdhoew5Y%2Btu65VwJqxUC1j4lav6UwpIJfnRswQUIMawPSr2LGp6WwLDYJ2TwoMNbf6Tdni%2FEuNvAdEvuUZAwFERLVXg7pg9xt1djZgqV7DmuHFGQI9Sje2A9dR%2FFDd0osztIRYnln1hdW1dff%2B1gtNLN1u0ViZy9BBlu%2BzBNUK%2BrIaP9Nla2TG%2BETHwq2kXzmS4XxXmSVan9KMYUprrbgFJqCndyIw9fgdh8dMvzIiW0sngbxoGlniN6LffruTEIGE9khBw5T2FDmWlTYqrnEPa7aF%2FYYcPYiUE48Ul5jhP82tj%2FiESyJilCeLdQRpod6No3xJNNHeZBpOBsiAzm5rg2dBZYSyH9Hob0EOFqqh3vWOuHbFR5eXcORp4OzwTUA4rUzVfJ4q%2FIa1GzCrzjOMxQr5uqLAWUOwgaHOphrgF0r2epYh%2FytdjBmUAurfM6CxruT3Ee%2BDv2%2FHAwK4RUIPskqK%2Fw4%2FR1F1bWfHjbNiXcYl6RwGJcMOMdXZaEVxCutSN1SGLMx3JfzCdlU8THZFFC%2BJJuB2964wSGdmq3I2FEcpWYVfHm4jmXd%2BRn7agFn9oFaWGYhBmJs5v5a0LZUjc3Sr4Ep%2FmFYlX8OdLlFYidM%2B731v7Ly4lfu85l3SSMTAcd5Bg2Sl%2FIHBm3RuacVx%2BrHpFcWjxztavOcOBcTnUhwekkGlsfWEt2%2FkHflB7WqKomGvs9F62l7a%2BRKQQQtRBD9VIlZiLEfRBRfQEmDb32cFQcSjznUP3um%2FkcbV%2BjmNEvqhOQuonjoQh7QF%2BbK811rduN5G6ICLD%2BnmPbi0ur2hrDLKhQYiwRdQrvKjcp%2F%2BL%2BnTz%2Fa4FgvmakvluPMMxbL15Dq

































