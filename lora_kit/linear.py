"""A plain fully-connected linear layer implemented with numpy.

Convention: for input x of shape (batch, in_features), the forward pass is

    y = x @ W.T + b

where W has shape (out_features, in_features) and b has shape (out_features,).
This mirrors the convention used by torch.nn.Linear.
"""

import numpy as np


class Linear:
    """A plain linear (fully connected) layer: y = x @ W.T + b."""

    def __init__(self, in_features, out_features, bias=True, rng=None, dtype=np.float64):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        if rng is None:
            rng = np.random.default_rng()

        scale = 1.0 / np.sqrt(in_features)
        self.W = rng.uniform(-scale, scale, size=(out_features, in_features)).astype(dtype)
        if bias:
            self.b = rng.uniform(-scale, scale, size=(out_features,)).astype(dtype)
        else:
            self.b = None

    def forward(self, x):
        x = np.asarray(x, dtype=self.dtype)
        y = x @ self.W.T
        if self.b is not None:
            y = y + self.b
        return y

    def __call__(self, x):
        return self.forward(x)
