"""Native PyTorch operations; imported only when a Torch tensor is supplied."""
import torch


class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quantized):
        return quantized

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class TorchOps:
    name = "torch"
    int_dtype = torch.int64
    float32 = torch.float32
    float64 = torch.float64

    def values(self, x):
        if x.dtype == torch.bool or x.is_complex():
            raise TypeError("P3109 requires real floating-point or integer tensors")
        dtype = x.dtype if x.dtype in (self.float32, self.float64) else self.float32
        return x.detach().to(dtype=dtype)

    def asarray(self, x, like):
        return torch.as_tensor(x, device=like.device)

    def cast(self, x, dtype):
        return x.to(dtype=dtype)

    def bitcast(self, x):
        return x.contiguous().view(torch.int64 if x.dtype == self.float64 else torch.int32)

    def where(self, condition, a, b):
        return torch.where(condition, a, b)

    def maximum(self, x, value):
        return torch.clamp_min(x, value)

    def power2(self, exponent, like):
        t, bias, maximum = (52, 1023, 2046) if like.dtype == self.float64 else (23, 127, 254)
        bits = torch.clamp(exponent + bias, 0, maximum) << t
        return bits.to(torch.int64 if t == 52 else torch.int32).contiguous().view(like.dtype)

    floor = staticmethod(torch.floor)
    ceil = staticmethod(torch.ceil)
    round = staticmethod(torch.round)
    isnan = staticmethod(torch.isnan)
    isinf = staticmethod(torch.isinf)
    isfinite = staticmethod(torch.isfinite)
    broadcast = staticmethod(torch.broadcast_to)

    def integer(self, x):
        return not (x.dtype == torch.bool or x.is_floating_point() or x.is_complex())

    def check(self, valid, message):
        if not bool(torch.all(valid)):
            raise ValueError(message)

    def stop(self, x):
        return x.detach()

    def ste(self, x, q):
        if not x.is_floating_point():
            raise TypeError("STE requires floating-point inputs")
        return _STE.apply(x, q)

    bit_and = staticmethod(torch.bitwise_and)
    left = staticmethod(torch.bitwise_left_shift)
    right = staticmethod(torch.bitwise_right_shift)

    def decode_dtype(self):
        return self.float64

    shape = staticmethod(lambda x: x.shape)

    def nearest_components(self, x, fmt):
        # Format guards ensure all host subnormal inputs are below half the
        # smallest target value. Thus FTZ cannot change nearest-even results.
        magnitude = torch.abs(x)
        shift = torch.clamp_min(torch.frexp(magnitude)[1] - fmt.precision,
                                2 - fmt.bias - fmt.precision)
        return torch.signbit(x), shift, torch.ldexp(magnitude, -shift)

    rescale = staticmethod(torch.ldexp)
