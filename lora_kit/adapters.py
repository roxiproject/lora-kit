"""Manage multiple named LoRA adapters sharing a single frozen base layer.

This is useful for the "many tasks, one base model" workflow: keep a single
frozen Linear base, and swap between several small (A, B) adapter pairs
without ever touching the base weights.
"""

import numpy as np

from .lora import LoRALinear


class AdapterManager:
    def __init__(self, base, r, alpha=None, dtype=np.float64):
        self.base = base
        self.r = r
        self.alpha = alpha if alpha is not None else r
        self.dtype = dtype

        self._adapters = {}   # name -> LoRALinear
        self.active_name = None

    def add_adapter(self, name, rng=None, r=None, alpha=None):
        if name in self._adapters:
            raise ValueError(f"adapter '{name}' already exists")

        adapter = LoRALinear(
            self.base,
            r=r if r is not None else self.r,
            alpha=alpha if alpha is not None else self.alpha,
            rng=rng,
            dtype=self.dtype,
        )
        self._adapters[name] = adapter

        if self.active_name is None:
            self.active_name = name

        return adapter

    def remove_adapter(self, name):
        if name == self.active_name:
            raise ValueError("cannot remove the currently active adapter")
        del self._adapters[name]

    def activate(self, name):
        if name not in self._adapters:
            raise KeyError(f"no such adapter: '{name}'")
        self.active_name = name

    @property
    def active(self):
        if self.active_name is None:
            raise RuntimeError("no adapter has been added yet")
        return self._adapters[self.active_name]

    def names(self):
        return list(self._adapters.keys())

    def forward(self, x):
        return self.active.forward(x)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_output):
        return self.active.backward(grad_output)
