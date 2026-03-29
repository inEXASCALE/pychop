import torch
from pychop import Chop


class CPTensor(torch.Tensor):
    """
    A PyTorch tensor subclass that maintains chopped precision after torch ops.

    Notes:
    - Most torch operations dispatch through __torch_function__ (Python-level).
      We unwrap CPTensor -> base Tensor, call the original function, then chop
      and wrap outputs back to CPTensor.
    - This guarantees "chop after each torch.* call", but NOT chopping internal
      steps inside fused kernels / LAPACK calls (similar to SciPy/JAX behavior).
    """

    # help precedence in mixed-type ops
    __array_priority__ = 1000

    def __new__(cls, input_tensor, chopper=None):
        if chopper is None:
            raise ValueError("Must provide a chopper (Chop instance)")
        base = torch.as_tensor(input_tensor)

        # Chop once at construction
        chopped = chopper(base)

        # Create subclass instance
        obj = torch.Tensor._make_subclass(cls, chopped, require_grad=chopped.requires_grad)
        obj.chopper = chopper
        return obj

    def __reduce_ex__(self, proto):
        return (CPTensor, (self.to_regular(), self.chopper))

    @staticmethod
    def _unwrap(x):
        if type(x) is CPTensor:
            # view as base Tensor without triggering __torch_function__ recursion
            return x.as_subclass(torch.Tensor)
        return x

    @staticmethod
    def _unwrap_tree(x):
        if isinstance(x, (tuple, list)):
            return type(x)(CPTensor._unwrap_tree(v) for v in x)
        if isinstance(x, dict):
            return {k: CPTensor._unwrap_tree(v) for k, v in x.items()}
        return CPTensor._unwrap(x)

    @staticmethod
    def _wrap_tree(x, chopper):
        if isinstance(x, (tuple, list)):
            return type(x)(CPTensor._wrap_tree(v, chopper) for v in x)
        if isinstance(x, dict):
            return {k: CPTensor._wrap_tree(v, chopper) for k, v in x.items()}

        if isinstance(x, torch.Tensor):
            # Only chop numeric tensors; bool/int/float/complex are all fine for Chop
            chopped = chopper(x)
            out = torch.Tensor._make_subclass(CPTensor, chopped, require_grad=chopped.requires_grad)
            out.chopper = chopper
            return out

        # Non-tensor outputs pass through (e.g., shapes, ints, etc.)
        return x

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        # Only handle if a CPTensor is involved
        if not any(t is CPTensor or issubclass(t, CPTensor) for t in types):
            return NotImplemented

        # Find chopper from first CPTensor in args/kwargs
        chopper = None

        def find_chopper_in_tree(obj):
            nonlocal chopper
            if chopper is not None:
                return
            if type(obj) is CPTensor:
                chopper = obj.chopper
                return
            if isinstance(obj, (tuple, list)):
                for v in obj:
                    find_chopper_in_tree(v)
            elif isinstance(obj, dict):
                for v in obj.values():
                    find_chopper_in_tree(v)

        find_chopper_in_tree(args)
        if chopper is None:
            find_chopper_in_tree(kwargs)

        if chopper is None:
            raise ValueError("At least one CPTensor argument is required to determine the chopper.")

        # Validate chopper consistency
        def validate(obj):
            if type(obj) is CPTensor and obj.chopper != chopper:
                raise ValueError("All CPTensor inputs must use the same chopper.")
            if isinstance(obj, (tuple, list)):
                for v in obj:
                    validate(v)
            elif isinstance(obj, dict):
                for v in obj.values():
                    validate(v)

        validate(args)
        validate(kwargs)

        # Unwrap to base tensors to avoid recursion
        pure_args = cls._unwrap_tree(args)
        pure_kwargs = cls._unwrap_tree(kwargs)

        # Disable torch function dispatch while calling func on base tensors
        with torch._C.DisableTorchFunction():
            out = func(*pure_args, **pure_kwargs)

        # Wrap outputs back to CPTensor and chop
        return cls._wrap_tree(out, chopper)

    def to_regular(self):
        """Return a base torch.Tensor view (drops CPTensor subclass)."""
        return self.as_subclass(torch.Tensor)

    def __str__(self):
        base_str = self.to_regular().__str__()
        prec_info = (
            f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}"
            if hasattr(self.chopper, "exp_bits")
            else "custom"
        )
        return f"CPTensor({base_str}, device={self.device}, {prec_info})"

    def __repr__(self):
        base_repr = self.to_regular().__repr__()
        prec_info = (
            f"exp_bits={self.chopper.exp_bits}, sig_bits={self.chopper.sig_bits}"
            if hasattr(self.chopper, "exp_bits")
            else "custom"
        )
        return f"CPTensor({base_repr}, device={self.device}, {prec_info})"