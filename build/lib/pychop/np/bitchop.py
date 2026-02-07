import numpy as np
import struct

class Bitchop(object):
    """
    Parameters
    ----------
    exp_bits : int
        Number of bits for the exponent in the target format. Determines the range
        of representable values (e.g., 5 bits gives a bias of 15, range -14 to 15).

    sig_bits : int
        Number of bits for the significand (mantissa) in the target format, excluding
        the implicit leading 1 for normalized numbers (e.g., 4 bits allows 0 to 15 plus implicit 1).

    subnormal : boolean
        Whether or not support subnormal numbers are supported.
        If set `subnormal=False`, subnormals are flushed to zero.
        
    rmode : int, default=1
        Rounding mode to use when quantizing the significand. Options are:
        - 1: Round to nearest value, ties to even (IEEE 754 default).
        - 0: Round to nearest value, ties to odd.
        - 2: Round towards plus infinity (round up).
        - 3: Round towards minus infinity (round down).
        - 4: Truncate toward zero (no rounding up).
        - 5: Stochastic rounding proportional to the fractional part.
        - 6: Stochastic rounding with 50% probability.
        
    random_state : int, default=0
        Random seed set for stochastic rounding settings.

    subnormal (bool, optional): If True, supports denormalized numbers (subnormals) when
        the exponent underflows, shifting the significand. If False, underflows result in zero.
        Defaults to True.

    Methods
    ----------
    bitchop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
        
    """

    def __init__(self, exp_bits, sig_bits, rmode=1, subnormal=True, random_state=42):
        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rmode = rmode
        
        self.subnormal = subnormal
        self.bias = (1 << (exp_bits - 1)) - 1
        self.max_exp = self.bias
        self.min_exp = -self.bias + 1
        self.mask_sig = (1 << sig_bits) - 1
        self.rng = np.random.RandomState(random_state)
        

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

        if self.rmode == 1:
            if remainder and (half_bit or exact_mantissa & 1):  # Tie to even (LSB = 0)
                rounded = exact_mantissa + 1
            else:
                rounded = exact_mantissa
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 0:
            if remainder and (half_bit and not (exact_mantissa & 1)):  # Tie to odd (LSB = 1)
                rounded = exact_mantissa + 1
            else:
                rounded = exact_mantissa
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 2:  # Round up
            if remainder and sign == 0:  # Positive numbers round up
                rounded = exact_mantissa + 1
            else:  # Negative numbers truncate (towards zero)
                rounded = exact_mantissa
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 3:  # Round down
            if remainder and sign == 1:  # Negative numbers round down
                rounded = exact_mantissa + 1
            else:  # Positive numbers truncate (towards zero)
                rounded = exact_mantissa
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 4:
            rounded = exact_mantissa
            did_round_up = False

        elif self.rmode == 5:
            prob = (mantissa * (1 << self.sig_bits) - exact_mantissa)
            rounded = exact_mantissa + (self.rng.random() < prob)
            did_round_up = rounded > exact_mantissa
            
        elif self.rmode == 6:
            rounded = exact_mantissa + (self.rng.random() < 0.5)
            did_round_up = rounded > exact_mantissa
        else:
            raise ValueError("Unknown rmode mode")

        if did_round_up and rounded >= (1 << self.sig_bits):
            rounded >>= 1
            exp += 1

        if exp > self.max_exp:
            raise OverflowError(f"Exponent {exp} too large")
        elif exp < self.min_exp:
            if not self.subnormal:
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
    bf_float32 = bitchop(exp_bits=exponent_bits, sig_bits=significand_bits, rmode=1)
    emulated_values = bf_float32(values_float32)
    print("Float32 emulated input:", emulated_values)
    print()

    # Test with float64 input
    values_float64 = np.array([3.14159, 0.1, -2.718], dtype=np.float64)
    bf_float64 = bitchop(exp_bits=exponent_bits, sig_bits=significand_bits, rmode=1)
    emulated_values = bf_float64(values_float64)
    print("Float64 emulated input:", emulated_values)
  