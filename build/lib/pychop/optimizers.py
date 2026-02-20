"""
pychop/optimizers.py

Quantized optimizers for floating-point quantization-aware training (QAT)
using ChopSTE from pychop.

These classes inherit directly from the corresponding torch.optim classes.
When a `chop` instance (ChopSTE) is provided, all optimizer internal states
(momentum buffers, variance, etc.) are fake-quantized after each `step()`.
This simulates low-precision optimizer behavior while preserving full
gradient flow via STE during training.

- chop=None → falls back to exact original PyTorch behavior (zero overhead).
- chop=ChopSTE(...) → enables floating-point/fixed-point QAT on optimizer states.

IQuantized* (integer) versions are omitted as requested.
"""

import torch
import torch.optim as optim
from typing import Optional, Tuple, Any, Callable


class QuantizedSGD(optim.SGD):
    """Quantized version of :class:`torch.optim.SGD` for floating-point QAT.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize or dicts defining parameter groups.
    lr : float, default=0.01
        Learning rate.
    momentum : float, default=0.0
        Momentum factor.
    dampening : float, default=0.0
        Dampening for momentum.
    weight_decay : float, default=0.0
        Weight decay (L2 penalty).
    nesterov : bool, default=False
        Enables Nesterov accelerated gradient.
    chop : ChopSTE or None, default=None
        Quantizer for optimizer states. If None, behaves like standard SGD.
    """

    def __init__(
        self,
        params: Any,
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        """Fake-quantize all optimizer state tensors (e.g. momentum_buffer)."""
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = self.chop(value)


class QuantizedAdam(optim.Adam):
    """Quantized version of :class:`torch.optim.Adam` for floating-point QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = self.chop(state[key])


class QuantizedAdamW(optim.AdamW):
    """Quantized version of :class:`torch.optim.AdamW` for floating-point QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, betas=betas, eps=eps,
            weight_decay=weight_decay, amsgrad=amsgrad
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = self.chop(state[key])


class QuantizedRMSprop(optim.RMSprop):
    """Quantized version of :class:`torch.optim.RMSprop` for floating-point QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, alpha=alpha, eps=eps,
            weight_decay=weight_decay, momentum=momentum, centered=centered
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key in ("square_avg", "momentum_buffer", "grad_avg"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = self.chop(state[key])


class QuantizedAdagrad(optim.Adagrad):
    """Quantized version of :class:`torch.optim.Adagrad` for floating-point QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value, eps=eps
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "sum" in state and isinstance(state["sum"], torch.Tensor):
                    state["sum"] = self.chop(state["sum"])


class QuantizedAdadelta(optim.Adadelta):
    """Quantized version of :class:`torch.optim.Adadelta` for floating-point QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key in ("square_avg", "acc_delta"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = self.chop(state[key])




# ====================== IQuantized Optimizers (Integer QAT) ======================
class IQuantizedSGD(optim.SGD):
    """Integer-quantized version of :class:`torch.optim.SGD` for QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.01,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, momentum=momentum, dampening=dampening,
            weight_decay=weight_decay, nesterov=nesterov
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = self.chop(value)   # ChopiSTE does quant + dequant


class IQuantizedAdam(optim.Adam):
    """Integer-quantized version of :class:`torch.optim.Adam` for QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        amsgrad: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = self.chop(state[key])


class IQuantizedAdamW(optim.AdamW):
    """Integer-quantized version of :class:`torch.optim.AdamW` for QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, amsgrad=amsgrad)
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = self.chop(state[key])


class IQuantizedRMSprop(optim.RMSprop):
    """Integer-quantized version of :class:`torch.optim.RMSprop` for QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, alpha=alpha, eps=eps,
            weight_decay=weight_decay, momentum=momentum, centered=centered
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key in ("square_avg", "momentum_buffer", "grad_avg"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = self.chop(state[key])


class IQuantizedAdagrad(optim.Adagrad):
    """Integer-quantized version of :class:`torch.optim.Adagrad` for QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 0.01,
        lr_decay: float = 0.0,
        weight_decay: float = 0.0,
        initial_accumulator_value: float = 0.0,
        eps: float = 1e-10,
        chop: Optional = None,
    ) -> None:
        super().__init__(
            params, lr=lr, lr_decay=lr_decay, weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value, eps=eps
        )
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if "sum" in state and isinstance(state["sum"], torch.Tensor):
                    state["sum"] = self.chop(state["sum"])


class IQuantizedAdadelta(optim.Adadelta):
    """Integer-quantized version of :class:`torch.optim.Adadelta` for QAT."""

    def __init__(
        self,
        params: Any,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        chop: Optional = None,
    ) -> None:
        super().__init__(params, lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        self.chop = chop

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        loss = super().step(closure)
        if self.chop is not None:
            self._quantize_states()
        return loss

    def _quantize_states(self) -> None:
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                for key in ("square_avg", "acc_delta"):
                    if key in state and isinstance(state[key], torch.Tensor):
                        state[key] = self.chop(state[key])