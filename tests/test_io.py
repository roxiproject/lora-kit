import numpy as np
import pytest

from lora_kit.linear import Linear
from lora_kit.lora import LoRALinear
from lora_kit.io import save_adapter, load_adapter


@pytest.mark.parametrize("in_f,out_f,r,alpha", [
    (4, 3, 1, 1),
    (6, 6, 2, 8),
    (8, 5, 4, 16),
    (10, 10, 8, 32),
])
def test_save_load_round_trip_output(in_f, out_f, r, alpha, rng, tmp_path):
    base = Linear(in_f, out_f, rng=rng)
    lora = LoRALinear(base, r=r, alpha=alpha, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)

    x = rng.normal(size=(5, in_f))
    y_before = lora.forward(x)

    path = tmp_path / "adapter.npz"
    save_adapter(str(path), lora)

    loaded, meta = load_adapter(str(path), base)
    y_after = loaded.forward(x)

    np.testing.assert_allclose(y_after, y_before, atol=1e-12)


def test_save_load_preserves_A_B_exactly(rng, tmp_path):
    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=3, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)

    path = tmp_path / "adapter.npz"
    save_adapter(str(path), lora)
    loaded, _ = load_adapter(str(path), base)

    np.testing.assert_array_equal(loaded.A, lora.A)
    np.testing.assert_array_equal(loaded.B, lora.B)


def test_save_load_preserves_rank_and_alpha(rng, tmp_path):
    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=3, alpha=12, rng=rng)

    path = tmp_path / "adapter.npz"
    save_adapter(str(path), lora)
    loaded, _ = load_adapter(str(path), base)

    assert loaded.r == 3
    assert loaded.alpha == 12
    assert loaded.scaling == pytest.approx(4.0)


def test_save_load_metadata(rng, tmp_path):
    base = Linear(4, 4, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)

    path = tmp_path / "adapter.npz"
    save_adapter(str(path), lora, metadata={"task": "toy_regression", "epoch": 10})
    _, meta = load_adapter(str(path), base)

    assert meta == {"task": "toy_regression", "epoch": 10}


def test_load_shape_mismatch_raises(rng, tmp_path):
    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    path = tmp_path / "adapter.npz"
    save_adapter(str(path), lora)

    wrong_base = Linear(6, 6, rng=rng)
    with pytest.raises(ValueError):
        load_adapter(str(path), wrong_base)


def test_save_load_no_metadata_defaults_empty(rng, tmp_path):
    base = Linear(4, 4, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    path = tmp_path / "adapter.npz"
    save_adapter(str(path), lora)
    _, meta = load_adapter(str(path), base)
    assert meta == {}


def test_loaded_adapter_independent_of_base_file(rng, tmp_path):
    """Loading does not require the original in-memory adapter at all."""
    base = Linear(5, 5, rng=rng)
    lora = LoRALinear(base, r=2, rng=rng)
    lora.B = rng.normal(size=lora.B.shape)
    path = tmp_path / "adapter.npz"
    save_adapter(str(path), lora)

    del lora  # simulate a fresh process

    loaded, _ = load_adapter(str(path), base)
    x = rng.normal(size=(2, 5))
    y = loaded.forward(x)
    assert y.shape == (2, 5)
