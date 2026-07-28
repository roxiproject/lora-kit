"""Sweep LoRA rank on the toy shifted-linear-regression task and report how
final approximation error changes with r.

The true shift used to build the target task has a fixed underlying rank
(TRUE_RANK); this script trains LoRA adapters of several ranks against that
same target and prints a table of final loss / RMSE per rank.
"""

import numpy as np

from lora_kit.linear import Linear
from lora_kit.lora import LoRALinear
from lora_kit.optim import Adam


TRUE_RANK = 6


def build_task(rng, in_features=20, out_features=10, n_samples=256):
    base = Linear(in_features, out_features, rng=rng)
    true_B = rng.normal(0, 1.0, size=(out_features, TRUE_RANK))
    true_A = rng.normal(0, 1.0, size=(TRUE_RANK, in_features))
    true_shift = true_B @ true_A

    x = rng.normal(size=(n_samples, in_features))
    y_target = base.forward(x) + x @ true_shift.T
    return base, x, y_target


def train_lora(base, x, y_target, r, rng, lr=0.05, n_steps=500):
    lora = LoRALinear(base, r=r, alpha=r, rng=rng)
    opt = Adam({"A": lora.A, "B": lora.B}, lr=lr)

    n_samples = x.shape[0]
    for _ in range(n_steps):
        y_pred = lora.forward(x)
        diff = y_pred - y_target
        grad_output = diff / n_samples
        grads = lora.backward(grad_output)
        opt.step(grads)

    y_pred = lora.forward(x)
    diff = y_pred - y_target
    final_loss = 0.5 * np.mean(np.sum(diff ** 2, axis=1))
    final_rmse = np.sqrt(np.mean(diff ** 2))
    return final_loss, final_rmse


def main():
    rng = np.random.default_rng(0)
    base, x, y_target = build_task(rng)

    ranks = [1, 2, 4, 8, 16]
    results = []
    for r in ranks:
        loss, rmse = train_lora(base, x, y_target, r, np.random.default_rng(1000 + r))
        results.append((r, loss, rmse))

    print(f"=== lora-kit rank sweep (true underlying rank = {TRUE_RANK}) ===")
    print(f"{'rank':>6} | {'final loss':>14} | {'final RMSE':>14}")
    print("-" * 42)
    for r, loss, rmse in results:
        print(f"{r:>6} | {loss:>14.6e} | {rmse:>14.6e}")

    return results


if __name__ == "__main__":
    main()
