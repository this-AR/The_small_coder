"""Torch-free forward pass of the controller heads (numpy + stdlib only), so the phone needs no torch/scipy."""
from __future__ import annotations

import math

import numpy as np

_erf = np.vectorize(math.erf, otypes=[np.float64])


def numpy_forward(w: dict, x: np.ndarray):
    """Same as ControllerHeads.predict (pass probs, expected tokens). Mirrors nn.GELU (exact erf form)."""
    z = (x - w["mu"]) / w["sigma"]
    if "trunk.0.weight" in w:
        z = z @ w["trunk.0.weight"].T + w["trunk.0.bias"]
        z = 0.5 * z * (1 + _erf(z / math.sqrt(2)))
    p = 1 / (1 + np.exp(-(z @ w["pass_head.weight"].T + w["pass_head.bias"]) / w["t_pass"]))
    t = np.expm1(z @ w["tok_head.weight"].T + w["tok_head.bias"])  # trained on log1p(tokens)
    return p, t
