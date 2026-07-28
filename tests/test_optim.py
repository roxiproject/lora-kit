import numpy as np
import pytest

from lora_kit.optim import Adam


def test_first_step_hand_computed():
    p = np.array([1.0])
    opt = Adam({"p": p}, lr=0.1, betas=(0.9, 0.999), eps=1e-8)
    grad = np.array([0.5])

    opt.step({"p": grad})

    # By hand: m = 0.1*0.5 = 0.05, v = 0.001*0.25 = 0.00025
    # m_hat = 0.05 / (1-0.9) = 0.5, v_hat = 0.00025 / (1-0.999) = 0.25
    # update = lr * m_hat / (sqrt(v_hat) + eps) = 0.1 * 0.5 / (0.5 + 1e-8)
    expected_update = 0.1 * 0.5 / (0.5 + 1e-8)
    expected_p = 1.0 - expected_update

    assert p[0] == pytest.approx(expected_p, rel=1e-8)


def test_step_counter_increments():
    p = np.zeros(3)
    opt = Adam({"p": p})
    assert opt.t == 0
    for i in range(5):
        opt.step({"p": np.ones(3)})
        assert opt.t == i + 1


def test_zero_grad_no_change():
    p = np.array([1.0, 2.0, 3.0])
    p_before = p.copy()
    opt = Adam({"p": p}, lr=0.1)
    opt.step({"p": np.zeros(3)})
    np.testing.assert_array_equal(p, p_before)


def test_converges_on_quadratic():
    """Minimize f(x) = (x - target)^2 via Adam; should converge close to target."""
    target = np.array([3.0, -2.0, 0.5])
    x = np.zeros(3)
    opt = Adam({"x": x}, lr=0.1)

    for _ in range(500):
        grad = 2 * (x - target)
        opt.step({"x": grad})

    np.testing.assert_allclose(x, target, atol=1e-3)


def test_multiple_params_independent():
    p1 = np.array([1.0])
    p2 = np.array([10.0])
    opt = Adam({"p1": p1, "p2": p2}, lr=0.1)

    opt.step({"p1": np.array([1.0]), "p2": np.array([0.0])})

    assert p1[0] != 1.0
    assert p2[0] == 10.0  # zero grad -> no change


def test_zero_state_resets_moments_not_params():
    p = np.array([5.0])
    opt = Adam({"p": p}, lr=0.1)
    opt.step({"p": np.array([1.0])})
    p_after_step = p.copy()

    opt.zero_state()

    assert opt.t == 0
    np.testing.assert_array_equal(opt.m["p"], np.zeros_like(p))
    np.testing.assert_array_equal(opt.v["p"], np.zeros_like(p))
    np.testing.assert_array_equal(p, p_after_step)  # params untouched


@pytest.mark.parametrize("lr", [0.001, 0.01, 0.1])
def test_convergence_various_lr(lr):
    target = np.array([1.0])
    x = np.zeros(1)
    opt = Adam({"x": x}, lr=lr)
    for _ in range(5000):
        grad = 2 * (x - target)
        opt.step({"x": grad})
    np.testing.assert_allclose(x, target, atol=3e-2)


def test_matrix_shaped_params():
    p = np.zeros((3, 4))
    target = np.ones((3, 4)) * 2.0
    opt = Adam({"p": p}, lr=0.1)
    for _ in range(300):
        grad = 2 * (p - target)
        opt.step({"p": grad})
    np.testing.assert_allclose(p, target, atol=1e-3)
