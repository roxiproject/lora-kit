"""Merge and unmerge LoRA updates into a plain Linear layer.

Merging folds the low-rank update B @ A (scaled by alpha/r) directly into
the base weight matrix, producing a plain Linear layer whose forward pass
is numerically equivalent (up to floating point error) to the unmerged
LoRALinear forward pass, but with no A/B matrices involved at inference
time.
"""

import numpy as np

from .linear import Linear


def merge(lora_linear):
    """Return a new plain Linear with W' = W + scaling * B @ A folded in.

    The returned Linear is independent of the input LoRALinear (weights are
    copied), so mutating one does not affect the other.
    """
    merged = Linear.__new__(Linear)
    merged.in_features = lora_linear.in_features
    merged.out_features = lora_linear.out_features
    merged.dtype = lora_linear.dtype

    delta_w = lora_linear.scaling * (lora_linear.B @ lora_linear.A)
    merged.W = lora_linear.base.W + delta_w

    if lora_linear.base.b is not None:
        merged.b = lora_linear.base.b.copy()
    else:
        merged.b = None

    return merged


def unmerge(merged_linear, lora_linear):
    """Recover the original frozen base Linear from a merged Linear.

    Given the same lora_linear (A, B, scaling) that was used to produce
    merged_linear via merge(), reconstruct W = W' - scaling * B @ A.
    """
    base = Linear.__new__(Linear)
    base.in_features = merged_linear.in_features
    base.out_features = merged_linear.out_features
    base.dtype = merged_linear.dtype

    delta_w = lora_linear.scaling * (lora_linear.B @ lora_linear.A)
    base.W = merged_linear.W - delta_w

    if merged_linear.b is not None:
        base.b = merged_linear.b.copy()
    else:
        base.b = None

    return base
