"""Save and load LoRA adapters to/from disk, independent of base weights.

Only the small (A, B) adapter matrices plus rank/alpha/metadata are
persisted -- not the (potentially large) frozen base weights. This mirrors
how LoRA checkpoints are shared in practice: a small file that can be
applied on top of any matching base model.
"""

import json

import numpy as np


def save_adapter(path, lora_linear, metadata=None):
    """Save an adapter's A, B, rank, alpha and optional metadata to `path`."""
    meta = dict(metadata) if metadata is not None else {}
    np.savez(
        path,
        A=lora_linear.A,
        B=lora_linear.B,
        r=np.array(lora_linear.r),
        alpha=np.array(lora_linear.alpha),
        in_features=np.array(lora_linear.in_features),
        out_features=np.array(lora_linear.out_features),
        metadata_json=np.array(json.dumps(meta)),
    )


def load_adapter(path, base):
    """Load an adapter from `path` and attach it to the given frozen `base`.

    Returns (lora_linear, metadata).
    """
    from .lora import LoRALinear

    data = np.load(path, allow_pickle=False)

    r = int(data["r"])
    alpha = float(data["alpha"])
    in_features = int(data["in_features"])
    out_features = int(data["out_features"])

    if in_features != base.in_features or out_features != base.out_features:
        raise ValueError(
            "adapter shape does not match base layer: "
            f"adapter=({out_features}, {in_features}) base=({base.out_features}, {base.in_features})"
        )

    lora_linear = LoRALinear(base, r=r, alpha=alpha, dtype=base.W.dtype)
    lora_linear.A = data["A"].astype(lora_linear.dtype)
    lora_linear.B = data["B"].astype(lora_linear.dtype)

    metadata = json.loads(str(data["metadata_json"]))

    return lora_linear, metadata
