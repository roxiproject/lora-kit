"""A from-scratch Adam optimizer for numpy arrays.

Implements the standard Adam update rule (Kingma & Ba, 2015) with bias
correction, operating on an arbitrary dict of named numpy-array parameters
(in this project, the LoRA "A" and "B" matrices).
"""

import numpy as np


class Adam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        """params: dict[str, np.ndarray] of parameters to optimize in place."""
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        self.m = {name: np.zeros_like(p) for name, p in params.items()}
        self.v = {name: np.zeros_like(p) for name, p in params.items()}

    def step(self, grads):
        """grads: dict[str, np.ndarray] of dL/dparam, same keys as self.params."""
        self.t += 1
        b1, b2 = self.beta1, self.beta2

        for name, grad in grads.items():
            p = self.params[name]

            self.m[name] = b1 * self.m[name] + (1 - b1) * grad
            self.v[name] = b2 * self.v[name] + (1 - b2) * (grad ** 2)

            m_hat = self.m[name] / (1 - b1 ** self.t)
            v_hat = self.v[name] / (1 - b2 ** self.t)

            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_state(self):
        """Reset moment estimates and step counter (does not touch params)."""
        self.t = 0
        for name in self.m:
            self.m[name][...] = 0.0
            self.v[name][...] = 0.0
