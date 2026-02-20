import torch

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
    
    device : str or torch.device, optional, default="cpu" 
        Device to perform computations on (e.g., "cpu", "cuda").
    
    subnormal (bool, optional): If True, supports denormalized numbers (subnormals) when
        the exponent underflows, shifting the significand. If False, underflows result in zero.
        Defaults to True.
    
    Methods
    ----------
    bitchop(x):
        Method that convert ``x`` to the user-specific arithmetic format.
    """


    def __init__(self, exp_bits, sig_bits, rmode=1, subnormal=True, random_state=42, device="cpu"):

        self.exp_bits = exp_bits
        self.sig_bits = sig_bits

        self.rmode = rmode
        
        self.subnormal = subnormal
        self.device = device
        
        self.bias = (1 << (exp_bits - 1)) - 1
        self.max_exp = self.bias
        self.min_exp = -self.bias + 1
        self.mask_sig = (1 << sig_bits) - 1
        self.rng = torch.Generator(device=device)
        self.rng.manual_seed(random_state)


    def __call__(self, values):
        """
        value (array-like): Input value(s) to convert to the low-precision format. Can be a scalar,
        list, NumPy array, or PyTorch tensor. Automatically converted to a PyTorch tensor.
    
        """
        self.value = torch.as_tensor(values, device=self.device)
        self.dtype = self.value.dtype
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("Input must be torch.float32 or torch.float64")
        
        if self.value.dim() == 0:
            self.value = self.value.unsqueeze(0)
        
        self.sign = torch.zeros_like(self.value, dtype=torch.uint8, device=self.device)
        self.exponent = torch.zeros_like(self.value, dtype=torch.int32, device=self.device)
        self.significand = torch.zeros_like(self.value, dtype=torch.int32, device=self.device)
        self.is_denormal = torch.zeros_like(self.value, dtype=torch.bool, device=self.device)
        self.rounding_value = torch.zeros_like(self.value, dtype=self.dtype, device=self.device)
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

        if self.rmode == 1:
            round_up = (remainder != 0) & (half_bit.bool() | (exact_mantissa & 1).bool())
            rounded = exact_mantissa + round_up.to(torch.int64)
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 0:
            round_up = (remainder != 0) & (half_bit.bool() & ~(exact_mantissa & 1).bool())
            rounded = exact_mantissa + round_up.to(torch.int64)
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 2:  
            round_up = (remainder != 0) & (sign == 0) 
            rounded = exact_mantissa + round_up.to(torch.int64)
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 3: 
            round_up = (remainder != 0) & (sign == 1) 
            rounded = exact_mantissa + round_up.to(torch.int64)
            did_round_up = rounded > exact_mantissa

        elif self.rmode == 4:
            rounded = exact_mantissa
            did_round_up = torch.zeros_like(rounded, dtype=torch.bool)

        elif self.rmode == 5:
            prob = (mantissa * (1 << self.sig_bits) - exact_mantissa.to(self.dtype))
            rand = torch.rand(exact_mantissa.shape, generator=self.rng, device=self.device, dtype=self.dtype)
            rounded = exact_mantissa + (rand < prob).to(torch.int64)
            did_round_up = rounded > exact_mantissa
            
        elif self.rmode == 6:
            rand = torch.rand(exact_mantissa.shape, generator=self.rng, device=self.device, dtype=self.dtype)
            rounded = exact_mantissa + (rand < 0.5).to(torch.int64)
            did_round_up = rounded > exact_mantissa
        else:
            raise ValueError("Unknown rounding mode.")
        
        overflow = did_round_up & (rounded >= (1 << self.sig_bits))
        rounded = torch.where(overflow, rounded >> 1, rounded)
        exp = exp + overflow.to(torch.int32)

        overflow_mask = exp > self.max_exp
        underflow_mask = exp < self.min_exp
        
        if overflow_mask.any():
            raise OverflowError(f"Exponent too large in {overflow_mask.sum()} elements")
        
        if underflow_mask.any():
            if not self.subnormal:
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
    bf_float32= bitchop(exp_bits=5, sig_bits=4, rmode=1, device=device)
    emulated_values = bf_float32(values_float32)
    print("Float32 emulated input(CPU):", emulated_values)
    print()


    # Test with float64 input
    values_float64 = torch.tensor([3.14159, 0.1, -2.718], dtype=torch.float64, device=device)
    bf_float64 = bitchop(exp_bits=5, sig_bits=4, rmode=1, device=device)
    emulated_values = bf_float64(values_float64)
    print("Float32 emulated input(GPU):", emulated_values)
    print()
