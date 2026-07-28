"""Gradient checks: compare analytic backward() gradients against central
finite differences. Uses float64 throughout for numerical stability.
"""

import numpy as np
import pytest

from lora_kit.linear import Linear
from lora_kit.lora import LoRALinear


def scalar_loss(y):
    """Simple sum-of-squares loss, differentiable and cheap."""
    return 0.5 * np.sum(y ** 2)


def grad_output_of_loss(y):
    return y  # dL/dy for L = 0.5 * sum(y^2)


def numeric_grad(f, param, eps=1e-6):
    """Central-difference numeric gradient of scalar function f w.r.t. param."""
    grad = np.zeros_like(param)
    it = np.nditer(param, flags=["multi_index"])
    for _ in it:
        idx = it.multi_index
        orig = param[idx]

        param[idx] = orig + eps
        f_plus = f()

        param[idx] = orig - eps
        f_minus = f()

        param[idx] = orig
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


CONFIGS = [
    (3, 2, 1),
    (4, 3, 2),
    (5, 5, 3),
    (6, 4, 2),
]


@pytest.mark.parametrize("in_f,out_f,r", CONFIGS)
def test_gradcheck_A(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)  # move B away from 0 so grads are non-trivial
    x = rng.normal(size=(4, in_f))

    def f():
        return scalar_loss(lora.forward(x))

    y = lora.forward(x)
    grads = lora.backward(grad_output_of_loss(y))
    analytic_dA = grads["A"]

    numeric_dA = numeric_grad(f, lora.A)

    max_abs_err = np.max(np.abs(analytic_dA - numeric_dA))
    max_rel_err = np.max(
        np.abs(analytic_dA - numeric_dA) / (np.abs(numeric_dA) + 1e-8)
    )
    assert max_abs_err < 1e-5, f"max abs err {max_abs_err}"
    assert max_rel_err < 1e-3, f"max rel err {max_rel_err}"


@pytest.mark.parametrize("in_f,out_f,r", CONFIGS)
def test_gradcheck_B(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)
    x = rng.normal(size=(4, in_f))

    def f():
        return scalar_loss(lora.forward(x))

    y = lora.forward(x)
    grads = lora.backward(grad_output_of_loss(y))
    analytic_dB = grads["B"]

    numeric_dB = numeric_grad(f, lora.B)

    max_abs_err = np.max(np.abs(analytic_dB - numeric_dB))
    max_rel_err = np.max(
        np.abs(analytic_dB - numeric_dB) / (np.abs(numeric_dB) + 1e-8)
    )
    assert max_abs_err < 1e-5, f"max abs err {max_abs_err}"
    assert max_rel_err < 1e-3, f"max rel err {max_rel_err}"


def test_gradcheck_A_at_init_zero_B(rng):
    """Even with B=0 (init state) gradcheck for dA must hold (it will be all zero)."""
    base = Linear(5, 4, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    x = rng.normal(size=(3, 5))

    def f():
        return scalar_loss(lora.forward(x))

    y = lora.forward(x)
    grads = lora.backward(grad_output_of_loss(y))
    numeric_dA = numeric_grad(f, lora.A)

    np.testing.assert_allclose(grads["A"], numeric_dA, atol=1e-5)
    np.testing.assert_allclose(grads["A"], 0.0, atol=1e-8)


def test_gradcheck_random_grad_output(rng):
    """Gradcheck using an arbitrary (non loss-derived) upstream gradient."""
    base = Linear(5, 4, rng=rng)
    lora = LoRALinear(base, r=3, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)
    x = rng.normal(size=(4, 5))
    g = rng.normal(size=(4, 4))

    def f_A():
        return np.sum(lora.forward(x) * g)

    lora.forward(x)
    grads = lora.backward(g)
    numeric_dA = numeric_grad(f_A, lora.A)
    numeric_dB = numeric_grad(f_A, lora.B)

    np.testing.assert_allclose(grads["A"], numeric_dA, atol=1e-5)
    np.testing.assert_allclose(grads["B"], numeric_dB, atol=1e-5)
