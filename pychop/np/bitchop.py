import numpy as np
import struct

class chop(object):

    """
    Parameters
    ----------
    subnormal : boolean
        Whether or not support subnormal numbers are supported.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    rmode : int, default=1
        The supported rounding modes include:
        1. Round to nearest using round to even last bit to break ties (the default).
        2. Round towards plus infinity (round up).
        3. Round towards minus infinity (round down).
        4. Round towards zero.
        5. Stochastic rounding - round to the next larger or next smaller
           floating-point number with probability proportional to the distance 
           to those floating-point numbers.
        6. Stochastic rounding - round to the next larger or next smaller 
           floating-point number with equal probability.

    flip : boolean, default=False
        Default is False; If ``flip`` is True, then each element
        of the rounded result has a randomly generated bit in its significand flipped 
        with probability ``p``. This parameter is designed for soft error simulation. 

    explim : boolean, default=True
        Default is True; If ``explim`` is False, then the maximal exponent for
        the specified arithmetic is ignored, thus overflow, underflow, or subnormal numbers
        will be produced only if necessary for the data type.  
        This option is designed for exploring low precisions independent of range limitations.

    p : float, default=0.5
        The probability ``p` for each element of the rounded result has a randomly
        generated bit in its significand flipped  when ``flip`` is True

    randfunc : callable, default=None
        If ``randfunc`` is supplied, then the random numbers used for rounding  will be generated 
        using that function in stochastic rounding (i.e., ``rmode`` of 5 and 6). Default is numbers
        in uniform distribution between 0 and 1, i.e., np.random.uniform.

    customs : dataclass, default=None
        If customs is defined, then use customs.t and customs.emax for floating point arithmetic.
    
    support_denormals (bool, optional): If True, supports denormalized numbers (subnormals) when
                the exponent underflows, shifting the significand. If False, underflows result in zero.
                Defaults to True.
    
    random_state : int, default=0
        Random seed set for stochastic rounding settings.

    Methods
    ----------
    chop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
        
    """

    def __init__(self, exp_bits, sig_bits, rounding="nearest_even", support_denormals=True, rounding_value=42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rounding = rounding
        self.support_denormals = support_denormals
        self.bias = (1 << (exp_bits - 1)) - 1
        self.max_exp = self.bias
        self.min_exp = -self.bias + 1
        self.mask_sig = (1 << sig_bits) - 1
        self.rng = np.random.RandomState(rounding_value)
        

    def __call__(self, values):
        self.value = np.asarray(values)
        self.dtype = self.value.dtype

        if self.dtype not in (np.float32, np.float64):
            raise ValueError("Input must be float32 or float64")
        
        self.sign = np.zeros_like(self.value, dtype=np.uint8)
        self.exponent = np.zeros_like(self.value, dtype=np.uint32)
        self.significand = np.zeros_like(self.value, dtype=np.uint32)
        self.is_denormal = np.zeros_like(self.value, dtype=bool)
        self.rounding_value = np.zeros_like(self.value, dtype=self.dtype) 

        self._convert()
        return self.rounding_value


    def _extract_components(self, value):
        if self.dtype == np.float32: # Simulate depends on the data type
            bits = struct.unpack('I', struct.pack('f', float(value)))[0]
            sign = (bits >> 31) & 1
            exp = ((bits >> 23) & 0xFF) - 127  # 8-bit exponent, bias 127
            mantissa = bits & ((1 << 23) - 1)  # 23-bit mantissa
            mantissa_bits = 23
            min_exp = -126
            bias = 127
        else:  
            bits = struct.unpack('Q', struct.pack('d', float(value)))[0]
            sign = (bits >> 63) & 1
            exp = ((bits >> 52) & 0x7FF) - 1023  # 11-bit exponent, bias 1023
            mantissa = bits & ((1 << 52) - 1)  # 52-bit mantissa
            mantissa_bits = 52
            min_exp = -1022
            bias = 1023
        
        if exp == -bias and mantissa == 0:  # Zero
            return sign, 0, 0, False
        elif exp == -bias:  # Denormal
            mantissa_norm = mantissa / (1 << mantissa_bits)
            exp = min_exp
            return sign, exp, mantissa_norm, True
        else:  # Normalized
            mantissa_norm = 1 + mantissa / (1 << mantissa_bits)
            return sign, exp, mantissa_norm, False

    def _adjust_to_format(self, sign, exp, mantissa):
        mantissa_bits = int(mantissa * (1 << (self.sig_bits + 1))) & ((1 << (self.sig_bits + 1)) - 1)
        exact_mantissa = mantissa_bits >> 1
        remainder = mantissa_bits & 1
        half_bit = (remainder << 1) & (mantissa_bits & 2)

        if self.rounding == "toward_zero":
            rounded = exact_mantissa
            did_round_up = False
        elif self.rounding == "nearest_even":
            if remainder and (half_bit or exact_mantissa & 1):  # Tie to even (LSB = 0)
                rounded = exact_mantissa + 1
            else:
                rounded = exact_mantissa
            did_round_up = rounded > exact_mantissa
        elif self.rounding == "nearest_odd":
            if remainder and (half_bit and not (exact_mantissa & 1)):  # Tie to odd (LSB = 1)
                rounded = exact_mantissa + 1
            else:
                rounded = exact_mantissa
            did_round_up = rounded > exact_mantissa
        elif self.rounding == "stochastic_prop":
            prob = (mantissa * (1 << self.sig_bits) - exact_mantissa)
            rounded = exact_mantissa + (self.rng.random() < prob)
            did_round_up = rounded > exact_mantissa
        elif self.rounding == "stochastic_equal":
            rounded = exact_mantissa + (self.rng.random() < 0.5)
            did_round_up = rounded > exact_mantissa
        else:
            raise ValueError("Unknown rounding mode")

        if did_round_up and rounded >= (1 << self.sig_bits):
            rounded >>= 1
            exp += 1

        if exp > self.max_exp:
            raise OverflowError(f"Exponent {exp} too large")
        elif exp < self.min_exp:
            if not self.support_denormals:
                return sign, 0, 0, False, 0.0
            shift = self.min_exp - exp
            rounded >>= shift
            exp = self.min_exp
            is_denormal = True
        else:
            is_denormal = False

        biased_exp = exp + self.bias if not is_denormal else 0
        sig_int = rounded & self.mask_sig
        reconstructed = self._reconstruct_scalar(sign, biased_exp, sig_int, is_denormal)
        return sign, biased_exp, sig_int, is_denormal, reconstructed


    def _reconstruct_scalar(self, sign, exponent, significand, is_denormal):
        if exponent == 0 and significand == 0:
            return np.array(0.0, dtype=self.dtype)  # Match input precision
        elif is_denormal:
            sig_value = significand / (1 << self.sig_bits)
            exp_value = self.min_exp
        else:
            sig_value = 1 + significand / (1 << self.sig_bits)
            exp_value = exponent - self.bias
        return np.array((-1) ** sign * sig_value * (2 ** exp_value), dtype=self.dtype)  # Cast to input precision


    def _convert(self):
        extract_vec = np.vectorize(self._extract_components, otypes=[np.uint8, np.int32, self.dtype, bool])
        signs, exps, mantissas = extract_vec(self.value)[:3]
        adjust_vec = np.vectorize(self._adjust_to_format, otypes=[np.uint8, np.uint32, np.uint32, bool, self.dtype])
        results = adjust_vec(signs, exps, mantissas)
        self.sign, self.exponent, self.significand, self.is_denormal, self.rounding_value = results


    def __str__(self, num=10):
        lines = []
        for i in range(self.value.size[:num]):
            val = self.value.flat[i]
            s = self.sign.flat[i]
            e = self.exponent.flat[i]
            sig = bin(self.significand.flat[i])[2:].zfill(self.sig_bits)
            recon = self.rounding_value.flat[i]
            denorm = self.is_denormal.flat[i]
            lines.append(f"value: {val}, sign: {s}, exp: {e}, sig: {sig}, emulated value: {recon}, denorm: {denorm}")
        return "\n".join(lines)

# Example usage
if __name__ == "__main__":
    # Test with float32 input
    significand_bits = 4
    exponent_bits = 5

    values_float32 = np.array([3.14159, 0.1, -2.718], dtype=np.float32)
    bf_float32 = chop(exp_bits=exponent_bits, sig_bits=significand_bits, rounding="nearest_even")
    emulated_values = bf_float32(values_float32)
    print("Float32 emulated input:", emulated_values)
    print()

    # Test with float64 input
    values_float64 = np.array([3.14159, 0.1, -2.718], dtype=np.float64)
    bf_float64 = chop(exp_bits=exponent_bits, sig_bits=significand_bits, rounding="nearest_even")
    emulated_values = bf_float64(values_float64)
    print("Float64 emulated input:", emulated_values)
  