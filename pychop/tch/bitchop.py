import torch


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

    random_state : int, default=0
        Random seed set for stochastic rounding settings.

        
    Methods
    ----------
    chop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
        
    """

    def __init__(self, exp_bits, sig_bits, rounding="nearest_even", support_denormals=True, random_state=42, device="cpu"):

        self.exp_bits = exp_bits
        self.sig_bits = sig_bits
        self.rounding = rounding
        self.support_denormals = support_denormals
        self.device = device
        
        self.bias = (1 << (exp_bits - 1)) - 1
        self.max_exp = self.bias
        self.min_exp = -self.bias + 1
        self.mask_sig = (1 << sig_bits) - 1
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(random_state)


    def __call__(self, values):
        self.value = torch.as_tensor(values, device=device)
        self.dtype = self.value.dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Input must be torch.float32 or torch.float64")
        
        if self.value.dim() == 0:
            self.value = self.value.unsqueeze(0)
        
        self.sign = torch.zeros_like(self.value, dtype=torch.uint8, device=device)
        self.exponent = torch.zeros_like(self.value, dtype=torch.int32, device=device)
        self.significand = torch.zeros_like(self.value, dtype=torch.int32, device=device)
        self.is_denormal = torch.zeros_like(self.value, dtype=torch.bool, device=device)
        self.rounding_value = torch.zeros_like(self.value, dtype=self.dtype, device=device)
        self._convert()
        return self.rounding_value

    def _extract_components(self):
        if self.dtype == torch.float32:
            bits = self.value.view(torch.int32)
            sign = (bits >> 31) & 1
            exp = ((bits >> 23) & 0xFF) - 127
            mantissa = bits & ((1 << 23) - 1)
            mantissa_bits = 23
            min_exp = -126
            bias = 127
        else: 
            bits = self.value.view(torch.int64)
            sign = (bits >> 63) & 1
            exp = ((bits >> 52) & 0x7FF) - 1023
            mantissa = bits & ((1 << 52) - 1)
            mantissa_bits = 52
            min_exp = -1022
            bias = 1023
        
        is_zero = (exp == -bias) & (mantissa == 0)
        is_denorm = (exp == -bias) & (mantissa != 0)
        
        mantissa_norm = torch.where(
            is_denorm,
            mantissa.double() / (1 << mantissa_bits),
            1.0 + mantissa.double() / (1 << mantissa_bits)
        ).to(self.dtype)
        
        exp = torch.where(is_denorm, torch.tensor(min_exp, dtype=torch.int32, device=self.device), exp)
        
        return (sign.to(torch.uint8), exp.to(torch.int32), mantissa_norm, is_zero)

    def _adjust_to_format(self, sign, exp, mantissa):
        mantissa_bits = (mantissa * (1 << (self.sig_bits + 1))).to(torch.int64) & ((1 << (self.sig_bits + 1)) - 1)
        exact_mantissa = mantissa_bits >> 1
        remainder = mantissa_bits & 1
        half_bit = (remainder << 1) & (mantissa_bits & 2)

        if self.rounding == "toward_zero":
            rounded = exact_mantissa
            did_round_up = torch.zeros_like(rounded, dtype=torch.bool)
        elif self.rounding == "nearest_even":
            round_up = (remainder != 0) & (half_bit.bool() | (exact_mantissa & 1).bool())
            rounded = exact_mantissa + round_up.to(torch.int64)
            did_round_up = rounded > exact_mantissa
        elif self.rounding == "nearest_odd":
            round_up = (remainder != 0) & (half_bit.bool() & ~(exact_mantissa & 1).bool())
            rounded = exact_mantissa + round_up.to(torch.int64)
            did_round_up = rounded > exact_mantissa
        elif self.rounding == "stochastic_prop":
            prob = (mantissa * (1 << self.sig_bits) - exact_mantissa.to(self.dtype))
            rand = torch.rand(exact_mantissa.shape, generator=self.rng, device=self.device, dtype=self.dtype)
            rounded = exact_mantissa + (rand < prob).to(torch.int64)
            did_round_up = rounded > exact_mantissa
        elif self.rounding == "stochastic_equal":
            rand = torch.rand(exact_mantissa.shape, generator=self.rng, device=self.device, dtype=self.dtype)
            rounded = exact_mantissa + (rand < 0.5).to(torch.int64)
            did_round_up = rounded > exact_mantissa
        else:
            raise ValueError("Unknown rounding mode")
        
        overflow = did_round_up & (rounded >= (1 << self.sig_bits))
        rounded = torch.where(overflow, rounded >> 1, rounded)
        exp = exp + overflow.to(torch.int32)

        overflow_mask = exp > self.max_exp
        underflow_mask = exp < self.min_exp
        
        if overflow_mask.any():
            raise OverflowError(f"Exponent too large in {overflow_mask.sum()} elements")
        
        if underflow_mask.any():
            if not self.support_denormals:
                sign = torch.where(underflow_mask, torch.tensor(0, dtype=torch.uint8, device=self.device), sign)
                exp = torch.where(underflow_mask, torch.tensor(0, dtype=torch.int32, device=self.device), exp)
                rounded = torch.where(underflow_mask, torch.tensor(0, dtype=torch.int32, device=self.device), rounded)
                is_denormal = torch.zeros_like(underflow_mask)
            else:
                shift = (self.min_exp - exp).clamp(min=0)
                rounded = torch.where(underflow_mask, rounded >> shift, rounded)
                exp = torch.where(underflow_mask, torch.tensor(self.min_exp, dtype=torch.int32, device=self.device), exp)
                is_denormal = underflow_mask
        else:
            is_denormal = torch.zeros_like(exp, dtype=torch.bool)

        biased_exp = torch.where(is_denormal, torch.tensor(0, dtype=torch.int32, device=self.device), exp + self.bias)
        sig_int = rounded & self.mask_sig
        reconstructed = self._reconstruct(sign, biased_exp, sig_int, is_denormal)
        return sign, biased_exp, sig_int, is_denormal, reconstructed

    def _reconstruct(self, sign, exponent, significand, is_denormal):
        zero_mask = (exponent == 0) & (significand == 0)
        sig_value = torch.where(
            is_denormal,
            significand.to(self.dtype) / (1 << self.sig_bits),
            1.0 + significand.to(self.dtype) / (1 << self.sig_bits)
        )
        exp_value = torch.where(
            is_denormal,
            torch.tensor(self.min_exp, dtype=torch.int32, device=self.device),
            exponent - self.bias
        )
        return torch.where(
            zero_mask,
            torch.tensor(0.0, dtype=self.dtype, device=self.device),
            ((-1) ** sign.to(self.dtype)) * sig_value * (2.0 ** exp_value.to(self.dtype))
        ).to(self.dtype)

    def _convert(self):
        sign, exp, mantissa, is_zero = self._extract_components()
        self.sign, self.exponent, self.significand, self.is_denormal, self.rounding_value = self._adjust_to_format(sign, exp, mantissa)

    def __str__(self):
        lines = []
        for i in range(self.value.numel()):
            val = self.value.flatten()[i].item()
            s = self.sign.flatten()[i].item()
            e = self.exponent.flatten()[i].item()
            sig = bin(self.significand.flatten()[i].item())[2:].zfill(self.sig_bits)
            recon = self.rounding_value.flatten()[i].item()
            denorm = self.is_denormal.flatten()[i].item()
            lines.append(f"value: {val}, sign: {s}, exp: {e}, sig: {sig}, emulated value: {recon}, denorm: {denorm}")
        return "\n".join(lines)

# Example usage
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test with float32 input
    values_float32 = torch.tensor([3.14159, 0.1, -2.718], dtype=torch.float32, device=device)
    bf_float32= chop(exp_bits=5, sig_bits=4, rounding="nearest_even", device=device)
    emulated_values = bf_float32(values_float32)
    print("Float32 emulated input(CPU):", emulated_values)
    print()


    # Test with float64 input
    values_float64 = torch.tensor([3.14159, 0.1, -2.718], dtype=torch.float64, device=device)
    bf_float64 = chop(exp_bits=5, sig_bits=4, rounding="nearest_even", device=device)
    emulated_values = bf_float64(values_float64)
    print("Float32 emulated input(GPU):", emulated_values)
    print()
