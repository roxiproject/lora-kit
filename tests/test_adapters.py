import numpy as np
import pytest

from lora_kit.linear import Linear
from lora_kit.adapters import AdapterManager


def make_manager(rng, in_f=6, out_f=4, r=2):
    base = Linear(in_f, out_f, rng=rng)
    mgr = AdapterManager(base, r=r)
    return base, mgr


def test_add_adapter_becomes_active(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    assert mgr.active_name == "task_a"


def test_names_lists_all_adapters(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    mgr.add_adapter("task_b", rng=rng)
    assert set(mgr.names()) == {"task_a", "task_b"}


def test_activate_switches_active_adapter(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    mgr.add_adapter("task_b", rng=rng)
    mgr.activate("task_b")
    assert mgr.active_name == "task_b"
    assert mgr.active is mgr._adapters["task_b"]


def test_activate_unknown_raises(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    with pytest.raises(KeyError):
        mgr.activate("nope")


def test_forward_before_any_adapter_raises(rng):
    base, mgr = make_manager(rng)
    x = np.zeros((1, 6))
    with pytest.raises(RuntimeError):
        mgr.forward(x)


def test_swap_changes_output(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=np.random.default_rng(1))
    mgr.add_adapter("task_b", rng=np.random.default_rng(2))

    # Perturb B for both adapters away from zero-init so outputs actually differ.
    mgr._adapters["task_a"].B = np.random.default_rng(11).normal(
        size=mgr._adapters["task_a"].B.shape
    )
    mgr._adapters["task_b"].B = np.random.default_rng(22).normal(
        size=mgr._adapters["task_b"].B.shape
    )

    x = np.random.default_rng(3).normal(size=(5, 6))

    mgr.activate("task_a")
    y_a = mgr.forward(x)

    mgr.activate("task_b")
    y_b = mgr.forward(x)

    assert not np.allclose(y_a, y_b)


def test_swap_back_restores_exact_output(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=np.random.default_rng(1))
    mgr.add_adapter("task_b", rng=np.random.default_rng(2))
    mgr._adapters["task_a"].B = np.random.default_rng(11).normal(
        size=mgr._adapters["task_a"].B.shape
    )

    x = np.random.default_rng(3).normal(size=(5, 6))

    mgr.activate("task_a")
    y_before = mgr.forward(x)

    mgr.activate("task_b")
    mgr.forward(x)

    mgr.activate("task_a")
    y_after = mgr.forward(x)

    np.testing.assert_array_equal(y_before, y_after)


def test_adapters_share_base_weights(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    mgr.add_adapter("task_b", rng=rng)
    assert mgr._adapters["task_a"].base is base
    assert mgr._adapters["task_b"].base is base


def test_remove_non_active_adapter(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    mgr.add_adapter("task_b", rng=rng)
    mgr.remove_adapter("task_b")
    assert mgr.names() == ["task_a"]


def test_remove_active_adapter_raises(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    with pytest.raises(ValueError):
        mgr.remove_adapter("task_a")


def test_duplicate_adapter_name_raises(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    with pytest.raises(ValueError):
        mgr.add_adapter("task_a", rng=rng)


def test_backward_uses_active_adapter(rng):
    base, mgr = make_manager(rng)
    mgr.add_adapter("task_a", rng=rng)
    mgr.add_adapter("task_b", rng=rng)

    x = rng.normal(size=(3, 6))
    mgr.activate("task_a")
    y = mgr.forward(x)
    grads = mgr.backward(y)
    assert set(grads.keys()) == {"A", "B"}
