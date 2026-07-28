"""LoRA (Low-Rank Adaptation) applied to a frozen linear layer.

Shape convention (matches lora_kit.linear.Linear / torch.nn.Linear):
  x: (batch, in_features)
  W: (out_features, in_features)   -- frozen base weight
  b: (out_features,)               -- frozen base bias (optional)
  A: (r, in_features)              -- LoRA down-projection, random init
  B: (out_features, r)             -- LoRA up-projection, zero init

Forward pass:

    y = x @ W.T + b + scaling * (x @ A.T) @ B.T

where scaling = alpha / r. Because B is initialized to zero, the LoRA
update contributes nothing at initialization, so the adapted layer starts
out identical to the frozen base layer.

W and b are frozen: they never receive gradients. Only A and B are
trainable parameters.
"""

import numpy as np

from .linear import Linear


class LoRALinear:
    """A frozen Linear layer augmented with a trainable low-rank update B @ A."""

    def __init__(self, base: Linear, r, alpha=None, rng=None, dtype=np.float64):
        if r <= 0:
            raise ValueError("r must be a positive integer")

        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / self.r
        self.dtype = dtype

        if rng is None:
            rng = np.random.default_rng()

        # A is randomly initialized (small values), B is zero-initialized so
        # that the initial LoRA update is exactly zero.
        self.A = rng.normal(0.0, 0.01, size=(r, self.in_features)).astype(dtype)
        self.B = np.zeros((self.out_features, r), dtype=dtype)

        # Cache for backward pass.
        self._cache = None

    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)

        base_out = x @ self.base.W.T
        if self.base.b is not None:
            base_out = base_out + self.base.b

        xa = x @ self.A.T          # (batch, r)
        lora_out = self.scaling * (xa @ self.B.T)  # (batch, out_features)

        y = base_out + lora_out

        self._cache = (x, xa)
        return y

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_output):
        """Compute gradients of the loss w.r.t. A and B given dL/dy.

        grad_output: (batch, out_features), the upstream gradient dL/dy.

        Returns a dict {"A": dL/dA, "B": dL/dB}. W and b are frozen and are
        never differentiated -- this method deliberately does not compute or
        return any gradient for them.
        """
        if self._cache is None:
            raise RuntimeError("backward() called before forward()")

        x, xa = self._cache
        grad_output = np.asarray(grad_output, dtype=self.dtype)

        # y = base_out + scaling * (x @ A.T) @ B.T
        # Let xa = x @ A.T, lora_out = scaling * xa @ B.T
        # dL/dB = scaling * grad_output.T @ xa
        dB = self.scaling * (grad_output.T @ xa)

        # dL/d(xa) = scaling * grad_output @ B
        d_xa = self.scaling * (grad_output @ self.B)

        # xa = x @ A.T  =>  dL/dA = d_xa.T @ x
        dA = d_xa.T @ x

        return {"A": dA, "B": dB}
