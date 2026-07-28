"""Verify that the base W and b are frozen: backward() never computes or
returns any gradient for them, and they are never mutated by forward/backward.
"""

import numpy as np
import pytest

from lora_kit.linear import Linear
from lora_kit.lora import LoRALinear


@pytest.mark.parametrize("in_f,out_f,r", [(4, 3, 1), (6, 6, 2), (8, 5, 4)])
def test_backward_does_not_return_W_or_b_grad(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    x = rng.normal(size=(3, in_f))
    y = lora.forward(x)
    grads = lora.backward(y)

    assert set(grads.keys()) == {"A", "B"}
    assert "W" not in grads
    assert "b" not in grads


@pytest.mark.parametrize("in_f,out_f,r", [(4, 3, 1), (6, 6, 2), (8, 5, 4)])
def test_base_weights_unchanged_after_forward_backward(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    W_before = base.W.copy()
    b_before = base.b.copy()

    x = rng.normal(size=(5, in_f))
    y = lora.forward(x)
    lora.backward(y)

    np.testing.assert_array_equal(base.W, W_before)
    np.testing.assert_array_equal(base.b, b_before)


def test_base_weights_unchanged_after_training_steps(rng):
    from lora_kit.optim import Adam

    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    W_before = base.W.copy()
    b_before = base.b.copy()

    opt = Adam({"A": lora.A, "B": lora.B}, lr=0.1)

    x = rng.normal(size=(4, 5))
    target = rng.normal(size=(4, 5))

    for _ in range(20):
        y = lora.forward(x)
        grad_output = y - target
        grads = lora.backward(grad_output)
        opt.step(grads)

    np.testing.assert_array_equal(base.W, W_before)
    np.testing.assert_array_equal(base.b, b_before)
