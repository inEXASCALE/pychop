"""PyTorch LightChop backend with optional STE support for training."""

import torch
import torch.nn as nn
from ..cadna_random import CADNARandomGenerator


def _check_torch_tensor(x):
    """
    Check if the input is a torch.Tensor.
    Issues a clear warning if a numpy.ndarray or other type is passed to the Torch backend.
    
    Parameters
    ----------
    x : Any
        Input to be checked.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Input to must be a torch.Tensor, "
            f"but got {type(x).__name__}. "
            "The Torch backend (LightChop_) only supports torch.Tensor inputs. "
            "Please use Chop with backend='numpy' for numpy.ndarray, "
            "or convert your data using torch.from_numpy(x) or torch.tensor(x)."
            )
        return 0

    return 1



class LightChop_:
    """
    A class to simulate different floating-point precisions and rounding modes
    for PyTorch tensors. This code implements a custom floating-point precision simulator
    that mimics IEEE 754 floating-point representation with configurable exponent bits (exp_bits),
    significand bits (sig_bits), and various rounding modes (rmode). 
    It uses PyTorch tensors for efficient computation and handles special cases like zeros,
    infinities, NaNs, and subnormal numbers. The code follows IEEE 754 conventions for sign, 
    exponent bias, implicit leading 1 (for normal numbers), and subnormal number handling.

    Initialize with specific format parameters.
    Convert to custom float representation with proper IEEE 754 handling
    
    Parameters
    ----------
    exp_bits: int 
        Number of bits for exponent.

    sig_bits : int
        Number of bits for significand (significant digits)

    rmode : int
        rounding modes.

        Rounding mode to use when quantizing the significand. Options are:
        - 1 : Round to nearest value, ties to even (IEEE 754 default).
        - 2 : Round towards plus infinity (round up).
        - 3 : Round towards minus infinity (round down).
        - 4 : Truncate toward zero (no rounding up).
        - 5 : Stochastic rounding proportional to the fractional part.
        - 6 : Stochastic rounding with 50% probability.
        - 7 : Round to nearest value, ties to zero.
        - 8 : Round to nearest value, ties to away.
        - 9 : Round to odd.
        - 10 : CADNA-style random directed rounding.
        
    random_state : int, default=42
        random seed for stochastic rounding.
    """
    def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, subnormal: bool = True, chunk_size: int = 1000, random_state: int = 42):
        """Initialize float precision simulator with custom format, rounding mode, and subnormal support."""
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = rmode
        self.subnormal = subnormal
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        # Precompute constants
        self.sig_steps = 2 ** sig_bits
        self.min_exp_power = 2.0 ** self.min_exp
        self.exp_min = 0
        self.exp_max = 2 ** exp_bits - 1
        self.inv_sig_steps = 1.0 / self.sig_steps
        self.inv_min_exp_power = 1.0 / self.min_exp_power  # Precompute for subnormal case
        # Use per-instance RNGs for stochastic rounding modes.
        # This avoids modifying PyTorch's global RNG state via torch.manual_seed(...).
        self.random_state = random_state
        self._torch_gens = {}
        self._bit_quantizers = {}
        if rmode == 10:
            self._cadna_gen = CADNARandomGenerator(seed=random_state, backend="torch")
        else:
            self._cadna_gen = None

    def _get_torch_generator(self, device: torch.device) -> torch.Generator:
        """Return a per-instance torch.Generator for the tensor device."""
        device = torch.device(device)
        key = (device.type, device.index)
        if key not in self._torch_gens:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.random_state)
            self._torch_gens[key] = gen
        return self._torch_gens[key]

    def _rand_like(self, x: torch.Tensor) -> torch.Tensor:
        """Per-instance replacement for torch.rand_like used by stochastic rounding."""
        return torch.rand(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=self._get_torch_generator(x.device),
        )

    def _to_custom_float(self, x):
        """Return exact source components without a logarithm or magnitude clamp."""
        sign = torch.sign(x)
        abs_x = torch.abs(x)
        zero_mask = abs_x == 0
        inf_mask = torch.isinf(x)
        nan_mask = torch.isnan(x)
        active = torch.isfinite(x) & ~zero_mask
        fraction, power = torch.frexp(torch.where(active, abs_x, torch.ones_like(x)))
        exponent = power - 1
        significand = 2 * fraction
        sub = active & (exponent < self.min_exp)
        if self.subnormal:
            sub_input = torch.where(sub, abs_x, torch.zeros_like(x))
            sub_scale = torch.full_like(power, -self.min_exp)
            significand = torch.where(sub, torch.ldexp(sub_input, sub_scale), significand)
        else:
            significand = torch.where(sub, torch.zeros_like(x), significand)
        exponent = torch.where(sub, self.min_exp - 1, exponent)
        return sign, exponent + self.bias, significand, zero_mask, inf_mask, nan_mask

    def _cadna_directed_round(self, val: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """
        CADNA-style random directed rounding for positive scaled significands.

        random bit = 0: upward rounding
            positive number -> ceil(val)
            negative number -> floor(val)

        random bit = 1: downward rounding
            positive number -> floor(val)
            negative number -> ceil(val)
        """
        if self._cadna_gen is None:
            self._cadna_gen = CADNARandomGenerator(backend="torch")

        bits_np = self._cadna_gen.random_bits(tuple(val.shape))

        bits = torch.as_tensor(bits_np, device=val.device, dtype=torch.bool)

        floor_val = torch.floor(val)
        ceil_val = torch.ceil(val)

        # bit=0: upward   => sign>0 chooses ceil, sign<=0 chooses floor
        # bit=1: downward => sign>0 chooses floor, sign<=0 chooses ceil
        choose_floor = bits == (sign > 0)

        return torch.where(choose_floor, floor_val, ceil_val)
    
    def _quantize_components(self, x, sign, exponent, significand,
                             zero_mask, inf_mask, nan_mask):
        """Compatibility entry point for the native bit quantizer."""
        return self.quantize(x)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified precision using the initialized rounding mode."""
        from .bitchop import Bitchop

        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        source_dtype = x.dtype
        if source_dtype not in (torch.float32, torch.float64):
            x = x.to(torch.float32)
        if x.dtype == torch.float32 and self.exp_bits > 8:
            x = x.to(torch.float64)
        key = x.device
        quantizer = self._bit_quantizers.get(key)
        if quantizer is None:
            quantizer = Bitchop(
                self.exp_bits, self.sig_bits, rmode=self.rmode,
                subnormal=self.subnormal, random_state=self.random_state,
                device=key,
            )
            self._bit_quantizers[key] = quantizer
        result = quantizer(x)
        if source_dtype in (torch.float16, torch.bfloat16, torch.float32):
            return result.to(source_dtype)
        return result



    # Trigonometric Functions
    def sin(self, x):
        x = self.quantize(x)
        result = torch.sin(x)
        return self.quantize(result)

    def cos(self, x):
        x = self.quantize(x)
        result = torch.cos(x)
        return self.quantize(result)

    def tan(self, x):
        x = self.quantize(x)
        result = torch.tan(x)
        return self.quantize(result)

    def arcsin(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) <= 1):
            raise ValueError("arcsin input must be in [-1, 1]")
        result = torch.asin(x)
        return self.quantize(result)

    def arccos(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) <= 1):
            raise ValueError("arccos input must be in [-1, 1]")
        result = torch.acos(x)
        return self.quantize(result)

    def arctan(self, x):
        x = self.quantize(x)
        result = torch.atan(x)
        return self.quantize(result)

    # Hyperbolic Functions
    def sinh(self, x):
        x = self.quantize(x)
        result = torch.sinh(x)
        return self.quantize(result)

    def cosh(self, x):
        x = self.quantize(x)
        result = torch.cosh(x)
        return self.quantize(result)

    def tanh(self, x):
        x = self.quantize(x)
        result = torch.tanh(x)
        return self.quantize(result)

    def arcsinh(self, x):
        x = self.quantize(x)
        result = torch.asinh(x)
        return self.quantize(result)

    def arccosh(self, x):
        x = self.quantize(x)
        if not torch.all(x >= 1):
            raise ValueError("arccosh input must be >= 1")
        result = torch.acosh(x)
        return self.quantize(result)

    def arctanh(self, x):
        x = self.quantize(x)
        if not torch.all(torch.abs(x) < 1):
            raise ValueError("arctanh input must be in (-1, 1)")
        result = torch.atanh(x)
        return self.quantize(result)

    # Exponential and Logarithmic Functions
    def exp(self, x):
        x = self.quantize(x)
        result = torch.exp(x)
        return self.quantize(result)

    def expm1(self, x):
        x = self.quantize(x)
        result = torch.expm1(x)
        return self.quantize(result)

    def log(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log input must be positive")
        result = torch.log(x)
        return self.quantize(result)

    def log10(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log10 input must be positive")
        result = torch.log10(x)
        return self.quantize(result)

    def log2(self, x):
        x = self.quantize(x)
        if not torch.all(x > 0):
            raise ValueError("log2 input must be positive")
        result = torch.log2(x)
        return self.quantize(result)

    def log1p(self, x):
        x = self.quantize(x)
        if not torch.all(x > -1):
            raise ValueError("log1p input must be > -1")
        result = torch.log1p(x)
        return self.quantize(result)

    # Power and Root Functions
    def sqrt(self, x):
        x = self.quantize(x)
        if not torch.all(x >= 0):
            raise ValueError("sqrt input must be non-negative")
        result = torch.sqrt(x)
        return self.quantize(result)

    def cbrt(self, x):
        x = self.quantize(x)
        result = torch.pow(x, 1/3)
        return self.quantize(result)

    # Miscellaneous Functions
    def abs(self, x):
        x = self.quantize(x)
        result = torch.abs(x)
        return self.quantize(result)

    def reciprocal(self, x):
        x = self.quantize(x)
        if not torch.all(x != 0):
            raise ValueError("reciprocal input must not be zero")
        result = torch.reciprocal(x)
        return self.quantize(result)

    def square(self, x):
        x = self.quantize(x)
        result = torch.square(x)
        return self.quantize(result)

    # Additional Mathematical Functions
    def frexp(self, x):
        x = self.quantize(x)
        mantissa, exponent = torch.frexp(x)
        return self.quantize(mantissa), exponent  # Exponent typically not chopped

    def hypot(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.hypot(x, y)
        return self.quantize(result)

    def diff(self, x, n=1):
        x = self.quantize(x)
        for _ in range(n):
            x = torch.diff(x)
        return self.quantize(x)

    def power(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.pow(x, y)
        return self.quantize(result)

    def modf(self, x):
        x = self.quantize(x)
        fractional, integer = torch.modf(x)
        return self.quantize(fractional), self.quantize(integer)

    def ldexp(self, x, i):
        x = self.quantize(x)
        i = torch.tensor(i, dtype=torch.int32, device=x.device)  # Exponent not chopped
        result = x * torch.pow(2.0, i)
        return self.quantize(result)

    def angle(self, x):
        if torch.is_complex(x):
            x = self.quantize(x)
            result = torch.angle(x)
        else:
            x = self.quantize(x)
            result = torch.atan2(x, torch.zeros_like(x))
        return self.quantize(result)

    def real(self, x):
        x = self.quantize(x)
        result = torch.real(x) if torch.is_complex(x) else x
        return self.quantize(result)

    def imag(self, x):
        x = self.quantize(x)
        result = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(x)
        return self.quantize(result)

    def conj(self, x):
        x = self.quantize(x)
        result = torch.conj(x) if torch.is_complex(x) else x
        return self.quantize(result)

    def maximum(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.maximum(x, y)
        return self.quantize(result)

    def minimum(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.minimum(x, y)
        return self.quantize(result)

    # Binary Arithmetic Functions
    def multiply(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.mul(x, y)
        return self.quantize(result)

    def mod(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("mod divisor must not be zero")
        result = torch.fmod(x, y)
        return self.quantize(result)

    def divide(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("divide divisor must not be zero")
        result = torch.div(x, y)
        return self.quantize(result)

    def add(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.add(x, y)
        return self.quantize(result)

    def subtract(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.sub(x, y)
        return self.quantize(result)

    def floor_divide(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        if not torch.all(y != 0):
            raise ValueError("floor_divide divisor must not be zero")
        result = torch.div(x, y, rounding_mode='floor')
        return self.quantize(result)

    def bitwise_and(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_and(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    def bitwise_or(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_or(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    def bitwise_xor(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.bitwise_xor(x.to(torch.int32), y.to(torch.int32)).to(torch.float32)
        return self.quantize(result)

    # Aggregation and Linear Algebra Functions
    def sum(self, x, axis=None):
        x = self.quantize(x)
        result = torch.sum(x, dim=axis)
        return self.quantize(result)

    def prod(self, x, axis=None):
        x = self.quantize(x)
        result = torch.prod(x, dim=axis)
        return self.quantize(result)

    def mean(self, x, axis=None):
        x = self.quantize(x)
        result = torch.mean(x, dim=axis)
        return self.quantize(result)

    def std(self, x, axis=None):
        x = self.quantize(x)
        result = torch.std(x, dim=axis)
        return self.quantize(result)

    def var(self, x, axis=None):
        x = self.quantize(x)
        result = torch.var(x, dim=axis)
        return self.quantize(result)

    def dot(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.dot(x, y)
        return self.quantize(result)

    def matmul(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.matmul(x, y)
        return self.quantize(result)

    # Rounding and Clipping Functions
    def floor(self, x):
        x = self.quantize(x)
        result = torch.floor(x)
        return self.quantize(result)

    def ceil(self, x):
        x = self.quantize(x)
        result = torch.ceil(x)
        return self.quantize(result)

    def round(self, x, decimals=0):
        x = self.quantize(x)
        if decimals == 0:
            result = torch.round(x)
        else:
            factor = 10 ** decimals
            result = torch.round(x * factor) / factor
        return self.quantize(result)

    def sign(self, x):
        x = self.quantize(x)
        result = torch.sign(x)
        return self.quantize(result)

    def clip(self, x, a_min, a_max):
        a_min = torch.tensor(a_min, dtype=torch.float32, device=x.device)
        a_max = torch.tensor(a_max, dtype=torch.float32, device=x.device)
        x = self.quantize(x)
        chopped_a_min = self.quantize(a_min)
        chopped_a_max = self.quantize(a_max)
        result = torch.clamp(x, min=chopped_a_min, max=chopped_a_max)
        return self.quantize(result)

    # Special Functions
    def erf(self, x):
        x = self.quantize(x)
        result = torch.erf(x)
        return self.quantize(result)

    def erfc(self, x):
        x = self.quantize(x)
        result = torch.erfc(x)
        return self.quantize(result)

    def gamma(self, x):
        x = self.quantize(x)
        result = torch.special.gamma(x)
        return self.quantize(result)

    # New Mathematical Functions
    def fabs(self, x):
        x = self.quantize(x)
        result = torch.abs(x)
        return self.quantize(result)

    def logaddexp(self, x, y):
        x = self.quantize(x)
        y = self.quantize(y)
        result = torch.logaddexp(x, y)
        return self.quantize(result)

    def cumsum(self, x, axis=None):
        x = self.quantize(x)
        result = torch.cumsum(x, dim=axis)
        return self.quantize(result)

    def cumprod(self, x, axis=None):
        x = self.quantize(x)
        result = torch.cumprod(x, dim=axis)
        return self.quantize(result)

    def degrees(self, x):
        x = self.quantize(x)
        result = torch.deg2rad(x) * (180 / torch.pi)
        return self.quantize(result)

    def radians(self, x):
        x = self.quantize(x)
        result = torch.rad2deg(x) * (torch.pi / 180)
        return self.quantize(result)
        
    def __call__(self, x: torch.Tensor):
        if _check_torch_tensor(x):
            return self.quantize(x)



class _PreserveSignedZeroSTE(torch.autograd.Function):
    """Keep zero's sign in the forward pass and the STE gradient in backward."""

    @staticmethod
    def forward(ctx, result, source):
        signed_zero = torch.copysign(torch.zeros_like(result), source)
        return torch.where(result == 0, signed_zero, result)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class LightChopSTE(nn.Module):
    """
    A PyTorch module for simulating low-precision floating-point quantization with Quantization-Aware Training (QAT) support.
    
    This implements a FakeQuant-style operator that mimics custom floating-point formats (e.g., FP8 E4M3/E5M2)
    by decomposing into sign/exponent/significand, applying configurable rounding, and supporting subnormals.
    It uses the Straight-Through Estimator (STE) for gradients during training.
    
    Parameters
    ----------
    exp_bits (int): Number of exponent bits (e.g., 4 for E4M3, 5 for E5M2).
    sig_bits (int): Number of significand (mantissa) bits, excluding the implicit leading 1.
    rmode (int): Rounding mode:
        1: Nearest (round ties to even is approximated)
        2: Positive infinity (ceil for positive, floor for negative)
        3: Negative infinity (floor for positive, ceil for negative)
        4: Towards zero (truncation)
        5: Stochastic proportional rounding
        6: Stochastic equal-probability rounding
        7-9: Additional specialized modes (ties handling, round-to-odd, etc.)
        10: CADNA-style random directed rounding
    subnormal (bool): Whether to support subnormal (denormal) numbers.
    """
    def __init__(
        self,
        exp_bits: int,
        sig_bits: int,
        rmode: int = 1,
        subnormal: bool = True,
        random_state: int = 42,
    ):
    #def __init__(self, exp_bits: int, sig_bits: int, rmode: int = 1, subnormal: bool = True):
        super().__init__()
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = rmode
        self.subnormal = subnormal
        self.max_exp = 2 ** (exp_bits - 1) - 1
        self.min_exp = -self.max_exp + 1
        self.bias = 2 ** (exp_bits - 1) - 1
        # Precompute constants
        self.sig_steps = 2 ** sig_bits
        self.min_exp_power = 2.0 ** self.min_exp
        self.exp_min = 0
        self.exp_max = 2 ** exp_bits - 1
        self.inv_sig_steps = 1.0 / self.sig_steps
        self.inv_min_exp_power = 1.0 / self.min_exp_power
        # Use per-instance RNGs for stochastic rounding modes.
        # This avoids modifying/depending on PyTorch's global RNG state.
        self.random_state = random_state
        self._torch_gens = {}
        if rmode == 10:
            self._cadna_gen = CADNARandomGenerator(seed=random_state, backend="torch")
        else:
            self._cadna_gen = None
        self._core_chop = LightChop_(exp_bits, sig_bits, rmode, subnormal,
                                     random_state=random_state)

    def _get_torch_generator(self, device: torch.device) -> torch.Generator:
        """Return a per-instance torch.Generator for the tensor device."""
        device = torch.device(device)
        key = (device.type, device.index)
        if key not in self._torch_gens:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.random_state)
            self._torch_gens[key] = gen
        return self._torch_gens[key]

    def _rand_like(self, x: torch.Tensor) -> torch.Tensor:
        """Per-instance replacement for torch.rand_like used by stochastic rounding."""
        return torch.rand(
            x.shape,
            dtype=x.dtype,
            device=x.device,
            generator=self._get_torch_generator(x.device),
        )

    def _to_custom_float(self, x):
        """Use the same decomposition as the public Torch LightChop path."""
        return self._core_chop._to_custom_float(x)

    def _cadna_directed_round(self, val: torch.Tensor, sign: torch.Tensor) -> torch.Tensor:
        """
        CADNA-style random directed rounding for positive scaled significands.

        bit = 0: upward rounding
        bit = 1: downward rounding
        """
        if self._cadna_gen is None:
            self._cadna_gen = CADNARandomGenerator(backend="torch")

        bits_np = self._cadna_gen.random_bits(tuple(val.shape))

        bits = torch.as_tensor(bits_np, device=val.device, dtype=torch.bool)

        floor_val = torch.floor(val)
        ceil_val = torch.ceil(val)

        # bit=0: upward   => sign>0 chooses ceil, sign<=0 chooses floor
        # bit=1: downward => sign>0 chooses floor, sign<=0 chooses ceil
        choose_floor = bits == (sign > 0)

        return torch.where(choose_floor, floor_val, ceil_val)
    
    def _quantize_components(self, x, sign, exponent, significand,
                             zero_mask, inf_mask, nan_mask):
        """Compatibility entry point preserving the STE gradient."""
        return self.quantize(x)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public interface: quantize the input tensor to the custom floating-point format.
        """
        rounded = self._core_chop.quantize(x)
        result = torch.where(torch.isfinite(x), x + (rounded - x).detach(), rounded)
        return _PreserveSignedZeroSTE.apply(result, x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Public interface (Compatible with nn.Module)
        """
        return self.quantize(x)
