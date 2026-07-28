import numpy as np
import pytest

from lora_kit.linear import Linear
from lora_kit.lora import LoRALinear
from lora_kit.merge import merge
from lora_kit.optim import Adam


def test_lora_dtype_float64_by_default(rng):
    base = Linear(4, 4, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    assert lora.A.dtype == np.float64
    assert lora.B.dtype == np.float64


def test_lora_forward_dtype_float32_base():
    rng = np.random.default_rng(0)
    base = Linear(4, 4, rng=rng, dtype=np.float32)
    lora = LoRALinear(base, r=2, rng=rng, dtype=np.float32)
    x = rng.normal(size=(3, 4)).astype(np.float32)
    y = lora.forward(x)
    assert y.dtype == np.float32


def test_rank_equal_to_min_dim(rng):
    base = Linear(4, 4, rng=rng)
    lora = LoRALinear(base, r=4, rng=rng)
    x = rng.normal(size=(2, 4))
    y = lora.forward(x)
    assert y.shape == (2, 4)


def test_rank_larger_than_dims_still_works(rng):
    base = Linear(3, 2, rng=rng)
    lora = LoRALinear(base, r=10, rng=rng)  # unusual but should not crash
    x = rng.normal(size=(2, 3))
    y = lora.forward(x)
    assert y.shape == (2, 2)


def test_batch_size_one(rng):
    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    x = rng.normal(size=(1, 5))
    y = lora.forward(x)
    assert y.shape == (1, 5)
    grads = lora.backward(y)
    assert grads["A"].shape == lora.A.shape
    assert grads["B"].shape == lora.B.shape


def test_large_batch(rng):
    base = Linear(5, 3, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    x = rng.normal(size=(1000, 5))
    y = lora.forward(x)
    assert y.shape == (1000, 3)


def test_forward_before_backward_required(rng):
    base = Linear(4, 4, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    with pytest.raises(RuntimeError):
        lora.backward(np.zeros((2, 4)))


def test_grad_shapes_match_param_shapes(rng):
    base = Linear(6, 4, rng=rng)
    lora = LoRALinear(base, r=3, rng=rng)
    x = rng.normal(size=(5, 6))
    y = lora.forward(x)
    grads = lora.backward(y)
    assert grads["A"].shape == (3, 6)
    assert grads["B"].shape == (4, 3)


def test_merge_preserves_dtype(rng):
    base = Linear(4, 4, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    merged = merge(lora)
    assert merged.W.dtype == lora.dtype


def test_adam_preserves_param_dtype():
    p = np.zeros((3,), dtype=np.float64)
    opt = Adam({"p": p}, lr=0.1)
    opt.step({"p": np.ones(3)})
    assert p.dtype == np.float64


def test_negative_and_positive_inputs_symmetric(rng):
    base = Linear(4, 4, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)
    x = rng.normal(size=(3, 4))
    y_pos = lora.forward(x)
    y_neg = lora.forward(-x)
    # Linear map (no nonlinearity), so f(-x) - const_terms == -(f(x) - const_terms)
    const = lora.base.b
    np.testing.assert_allclose(y_pos - const, -(y_neg - const), atol=1e-10)
