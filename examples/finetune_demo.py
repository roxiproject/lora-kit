"""Demo: freeze a random "pretrained" linear model and adapt it with LoRA
on a toy shifted-linear-regression task, training with the from-scratch
Adam optimizer.

The toy task: the "pretrained" base layer computes y = x @ W.T + b for some
random fixed W, b. The task we actually want to solve is a *shifted*
version of that mapping, y_target = x @ W.T + b + x @ C.T for some unknown
low-rank-ish shift C. LoRA is used to adapt the frozen base to approximate
this shifted target without touching W or b.
"""

import numpy as np

from lora_kit.linear import Linear
from lora_kit.lora import LoRALinear
from lora_kit.optim import Adam


def main():
    rng = np.random.default_rng(0)

    in_features, out_features = 20, 10
    n_samples = 256
    r = 8
    lr = 0.05
    n_steps = 500

    # A random "pretrained" base model -- frozen throughout.
    base = Linear(in_features, out_features, rng=rng)

    # The target task is a shift of the base mapping by a random low-rank
    # matrix, which is exactly the kind of update LoRA is designed to learn.
    true_B = rng.normal(0, 1.0, size=(out_features, r))
    true_A = rng.normal(0, 1.0, size=(r, in_features))
    true_shift = true_B @ true_A

    x = rng.normal(size=(n_samples, in_features))
    y_target = base.forward(x) + x @ true_shift.T

    lora = LoRALinear(base, r=r, alpha=r, rng=rng)
    opt = Adam({"A": lora.A, "B": lora.B}, lr=lr)

    losses = []
    for step in range(n_steps):
        y_pred = lora.forward(x)
        diff = y_pred - y_target
        loss = 0.5 * np.mean(np.sum(diff ** 2, axis=1))
        losses.append(loss)

        grad_output = diff / n_samples  # dL/dy for mean-squared-error style loss
        grads = lora.backward(grad_output)
        opt.step(grads)

    initial_loss = losses[0]
    final_loss = losses[-1]

    y_pred_final = lora.forward(x)
    final_rmse = np.sqrt(np.mean((y_pred_final - y_target) ** 2))

    print("=== lora-kit finetune demo ===")
    print(f"in_features={in_features} out_features={out_features} r={r} lr={lr} steps={n_steps}")
    print(f"initial loss: {initial_loss:.6f}")
    print(f"final loss:   {final_loss:.3e}")
    print(f"final RMSE:   {final_rmse:.3e}")
    print(f"loss reduction: {(1 - final_loss / initial_loss) * 100:.2f}%")

    return {
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_rmse": final_rmse,
    }


if __name__ == "__main__":
    main()
