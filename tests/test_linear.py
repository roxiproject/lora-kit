import numpy as np
import pytest

from lora_kit.linear import Linear


SHAPES = [(1, 1), (3, 5), (5, 3), (10, 10), (4, 20), (20, 4)]


@pytest.mark.parametrize("in_f,out_f", SHAPES)
def test_output_shape(in_f, out_f, rng):
    layer = Linear(in_f, out_f, rng=rng)
    x = rng.normal(size=(7, in_f))
    y = layer.forward(x)
    assert y.shape == (7, out_f)


@pytest.mark.parametrize("in_f,out_f", SHAPES)
def test_forward_matches_manual_matmul(in_f, out_f, rng):
    layer = Linear(in_f, out_f, rng=rng)
    x = rng.normal(size=(4, in_f))
    y = layer.forward(x)
    expected = x @ layer.W.T + layer.b
    np.testing.assert_allclose(y, expected, atol=1e-12)


def test_no_bias(rng):
    layer = Linear(5, 3, bias=False, rng=rng)
    assert layer.b is None
    x = rng.normal(size=(2, 5))
    y = layer.forward(x)
    np.testing.assert_allclose(y, x @ layer.W.T, atol=1e-12)


def test_single_sample(rng):
    layer = Linear(6, 2, rng=rng)
    x = rng.normal(size=(1, 6))
    y = layer.forward(x)
    assert y.shape == (1, 2)


def test_callable_matches_forward(rng):
    layer = Linear(4, 4, rng=rng)
    x = rng.normal(size=(3, 4))
    np.testing.assert_allclose(layer(x), layer.forward(x))


def test_dtype_is_float64_by_default(rng):
    layer = Linear(3, 3, rng=rng)
    assert layer.W.dtype == np.float64


def test_weight_shape(rng):
    layer = Linear(7, 11, rng=rng)
    assert layer.W.shape == (11, 7)
    assert layer.b.shape == (11,)


def test_reproducible_with_seeded_rng():
    r1 = np.random.default_rng(42)
    r2 = np.random.default_rng(42)
    l1 = Linear(5, 5, rng=r1)
    l2 = Linear(5, 5, rng=r2)
    np.testing.assert_array_equal(l1.W, l2.W)
    np.testing.assert_array_equal(l1.b, l2.b)
