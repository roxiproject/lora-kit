import numpy as np
import pytest

from lora_kit.linear import Linear
from lora_kit.lora import LoRALinear
from lora_kit.merge import merge, unmerge


CONFIGS = [
    (4, 3, 1),
    (5, 5, 2),
    (8, 6, 4),
    (10, 10, 8),
    (3, 7, 2),
]


@pytest.mark.parametrize("in_f,out_f,r", CONFIGS)
def test_merged_output_matches_unmerged(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)  # nonzero B

    x = rng.normal(size=(6, in_f))

    y_unmerged = lora.forward(x)
    merged = merge(lora)
    y_merged = merged.forward(x)

    np.testing.assert_allclose(y_merged, y_unmerged, atol=1e-8, rtol=1e-6)


@pytest.mark.parametrize("in_f,out_f,r", CONFIGS)
def test_merge_at_zero_init_equals_base(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)  # B is zero
    merged = merge(lora)
    np.testing.assert_allclose(merged.W, base.W, atol=1e-12)
    np.testing.assert_allclose(merged.b, base.b, atol=1e-12)


@pytest.mark.parametrize("in_f,out_f,r", CONFIGS)
def test_unmerge_recovers_base(in_f, out_f, r, rng):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)

    merged = merge(lora)
    recovered_base = unmerge(merged, lora)

    np.testing.assert_allclose(recovered_base.W, base.W, atol=1e-10)
    np.testing.assert_allclose(recovered_base.b, base.b, atol=1e-10)


def test_merge_does_not_mutate_original(rng):
    base = Linear(6, 6, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)
    W_before = base.W.copy()

    merged = merge(lora)
    merged.W[0, 0] += 100.0  # mutate merged; must not affect base

    np.testing.assert_array_equal(base.W, W_before)


def test_merge_no_bias(rng):
    base = Linear(5, 5, bias=False, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)
    merged = merge(lora)
    assert merged.b is None

    x = rng.normal(size=(3, 5))
    np.testing.assert_allclose(merged.forward(x), lora.forward(x), atol=1e-8)


def test_merge_unmerge_round_trip_multiple_times(rng):
    base = Linear(6, 4, rng=rng)
    lora = LoRALinear(base, r=3, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)

    merged = merge(lora)
    recovered = unmerge(merged, lora)
    remerged = merge(lora)

    np.testing.assert_allclose(recovered.W, base.W, atol=1e-10)
    np.testing.assert_allclose(remerged.W, merged.W, atol=1e-10)
