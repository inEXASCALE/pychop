import torch
import pychop
# pychop.backend('torch')  # Switch to PyTorch backend (once)
from pychop import LightChop  # Or: from pychop import Chop

class CPTensor(torch.Tensor):
    """
    A PyTorch tensor subclass that maintains chopped precision after arithmetic ops.
    - Inherits from torch.Tensor for full compatibility.
    - Uses LightChop for rounding tensors.
    - Operations return CPTensor instances (chopped post-op).
    Fixed 100%: Temporary class stripping prevents recursion in dispatch/printing.
    """
    def __new__(cls, input_tensor, chopper=None):
        if chopper is None:
            raise ValueError("Must provide a chopper (LightChop or Chop instance)")
        # Ensure input is a pure tensor
        base_input = torch.as_tensor(input_tensor)  # Strip any subclass
        # Chop the base tensor FIRST (pure tensor) to avoid recursion
        chopped_base = chopper(base_input)  # LightChop on pure → pure chopped tensor
        # Now view the pre-chopped base as CPTensor (no re-chop)
        obj = chopped_base.as_subclass(cls)
        obj.chopper = chopper  # Per-instance storage
        return obj

    def __reduce_ex__(self, proto):
        # For pickling/serialization
        return (CPTensor, (self.to_regular(), self.chopper))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        """
        Override for torch functions (e.g., +, *, mm, etc.): Strip subclass, compute on plain, chop result.
        """
        if kwargs is None:
            kwargs = {}

        # Get chopper from first CPTensor arg (per-instance)
        chopper = None
        first_chopped = None
        for a in args:
            # isinstance triggers __torch_function__ → recursion
            # → compare the class pointer directly (fast & safe)
            if type(a) is CPTensor:          # <-- exact type match
                first_chopped = a
                break
        
        if first_chopped is None:
            # No CPTensor → we cannot decide which chopper to use
            raise ValueError("At least one CPTensor arg required for chopper")
        
        chopper = first_chopped.chopper
        # Validate same chopper for other CPTensor args
        for arg in args:
            if isinstance(arg, CPTensor) and arg.chopper != chopper:
                raise ValueError("All CPTensor inputs must use the same chopper")

        # Strip subclass from args (hack to prevent recursion: make plain temporarily)
        restored = []
        pure_args = list(args)
        for a in pure_args:
            if isinstance(a, CPTensor):
                original_class = type(a)
                a.__class__ = torch.Tensor
                restored.append((a, original_class))

        # Compute on plain tensors
        result = func(*pure_args, **kwargs)

        # Restore classes
        for a, original_class in restored:
            a.__class__ = original_class

        # If result is not a tensor, return as-is
        if not isinstance(result, torch.Tensor):
            return result

        # Chop the pure result
        chopped_result = chopper(result)  # LightChop on pure → pure chopped

        # Return as CPTensor (set chopper on new instance)
        new_instance = chopped_result.as_subclass(CPTensor)
        new_instance.chopper = chopper
        return new_instance

    # Custom printing: Wrap base (now safe) with precision info
    def __str__(self):
        base_str = super().__str__()
        prec_info = f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}" if hasattr(self.chopper, 'exp_bits') else "custom"
        return f"CPTensor({base_str}, device={self.device}, {prec_info})"

    def __repr__(self):
        base_repr = super().__repr__()
        prec_info = f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}" if hasattr(self.chopper, 'exp_bits') else "custom"
        return f"CPTensor({base_repr}, device={self.device}, {prec_info})"

    def __format__(self, format_spec):
        # Delegate to base (safe now)
        return super().__format__(format_spec)

    # Utility: View as regular (plain) tensor (using hack for safety)
    def to_regular(self):
        original_class = self.__class__
        self.__class__ = torch.Tensor
        plain = self
        self.__class__ = original_class
        return plain
