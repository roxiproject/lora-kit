import numpy as np
import pytest

from lora_kit.linear import Linear
from lora_kit.lora import LoRALinear


CONFIGS = [
    (4, 3, 1),
    (4, 3, 2),
    (8, 8, 4),
    (10, 5, 1),
    (5, 10, 3),
    (16, 16, 8),
]


@pytest.mark.parametrize("in_f,out_f,r", CONFIGS)
def test_output_shape(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    x = rng.normal(size=(6, in_f))
    y = lora.forward(x)
    assert y.shape == (6, out_f)


@pytest.mark.parametrize("in_f,out_f,r", CONFIGS)
def test_zero_init_matches_base(in_f, out_f, r, rng):
    """At init, B=0 so LoRALinear output must equal the frozen base output."""
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    x = rng.normal(size=(5, in_f))

    y_lora = lora.forward(x)
    y_base = base.forward(x)
    np.testing.assert_allclose(y_lora, y_base, atol=1e-12)


@pytest.mark.parametrize("in_f,out_f,r", CONFIGS)
def test_forward_formula(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)  # perturb B away from zero
    x = rng.normal(size=(4, in_f))

    y = lora.forward(x)
    expected = x @ base.W.T + base.b + lora.scaling * (x @ lora.A.T) @ lora.B.T
    np.testing.assert_allclose(y, expected, atol=1e-10)


def test_B_initialized_to_zero(rng):
    base = Linear(6, 6, rng=rng)
    lora = LoRALinear(base, r=3, rng=rng)
    np.testing.assert_array_equal(lora.B, np.zeros_like(lora.B))


def test_A_not_all_zero(rng):
    base = Linear(6, 6, rng=rng)
    lora = LoRALinear(base, r=3, rng=rng)
    assert not np.allclose(lora.A, 0.0)


@pytest.mark.parametrize("alpha,r", [(1, 1), (8, 4), (16, 4), (32, 8), (2, 8)])
def test_scaling_value(alpha, r, rng):
    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=r, alpha=alpha, rng=rng)
    assert lora.scaling == pytest.approx(alpha / r)


def test_default_alpha_equals_r(rng):
    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=4, rng=rng)
    assert lora.alpha == 4
    assert lora.scaling == pytest.approx(1.0)


def test_invalid_rank_raises(rng):
    base = Linear(4, 4, rng=rng)
    with pytest.raises(ValueError):
        LoRALinear(base, r=0, rng=rng)
    with pytest.raises(ValueError):
        LoRALinear(base, r=-2, rng=rng)


def test_A_shape(rng):
    base = Linear(7, 3, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    assert lora.A.shape == (2, 7)


def test_B_shape(rng):
    base = Linear(7, 3, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    assert lora.B.shape == (3, 2)


def test_callable_matches_forward(rng):
    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    x = rng.normal(size=(3, 5))
    np.testing.assert_allclose(lora(x), lora.forward(x))


def test_base_untouched_by_lora_construction(rng):
    base = Linear(5, 5, rng=rng)
    W_before = base.W.copy()
    b_before = base.b.copy()
    LoRALinear(base, r=2, rng=rng)
    np.testing.assert_array_equal(base.W, W_before)
    np.testing.assert_array_equal(base.b, b_before)
