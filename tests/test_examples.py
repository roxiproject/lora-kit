"""Integration tests exercising the example scripts as libraries."""

import numpy as np
import pytest

from examples.finetune_demo import main as finetune_main
from examples.rank_sweep import build_task, train_lora, TRUE_RANK


def test_finetune_demo_loss_decreases():
    result = finetune_main()
    assert result["final_loss"] < result["initial_loss"]


def test_finetune_demo_final_loss_near_zero():
    result = finetune_main()
    assert result["final_loss"] < 1e-6


def test_finetune_demo_final_rmse_small():
    result = finetune_main()
    assert result["final_rmse"] < 1e-4


def test_rank_sweep_low_rank_has_higher_error_than_true_rank():
    rng = np.random.default_rng(0)
    base, x, y_target = build_task(rng)

    loss_low, _ = train_lora(base, x, y_target, r=1, rng=np.random.default_rng(1001))
    loss_at_true_rank, _ = train_lora(
        base, x, y_target, r=TRUE_RANK, rng=np.random.default_rng(1000 + TRUE_RANK)
    )

    assert loss_at_true_rank < loss_low


def test_rank_sweep_error_generally_decreases():
    rng = np.random.default_rng(0)
    base, x, y_target = build_task(rng)

    ranks = [1, 2, 4, 8]
    losses = []
    for r in ranks:
        loss, _ = train_lora(base, x, y_target, r=r, rng=np.random.default_rng(2000 + r))
        losses.append(loss)

    # Error should be non-increasing overall (each step at least not worse
    # than the previous, allowing for tiny numerical noise near zero).
    for earlier, later in zip(losses, losses[1:]):
        assert later <= earlier + 1e-6


def test_rank_sweep_at_or_above_true_rank_reaches_near_zero_error():
    rng = np.random.default_rng(0)
    base, x, y_target = build_task(rng)

    loss, rmse = train_lora(
        base, x, y_target, r=TRUE_RANK, rng=np.random.default_rng(1000 + TRUE_RANK)
    )
    assert loss < 1e-4
    assert rmse < 1e-2
