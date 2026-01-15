from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

def set_seed(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

def normalize01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x)
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + eps)

def angle_of(vec2: np.ndarray) -> np.ndarray:
    return np.arctan2(vec2[..., 1], vec2[..., 0])

def hsv_from_latent(z2: np.ndarray, mag: np.ndarray, conf: np.ndarray) -> np.ndarray:
    ang = angle_of(z2)
    hue = (ang + math.pi) / (2 * math.pi)
    val = normalize01(mag)
    sat = np.clip(conf, 0.0, 1.0)
    return np.stack([hue, sat, val], axis=-1)

def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int)
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i_mod = i % 6
    r = np.choose(i_mod, [v, q, p, p, t, v])
    g = np.choose(i_mod, [t, v, v, q, p, p])
    b = np.choose(i_mod, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1)

@dataclass
class EdgeIndex:
    i: np.ndarray
    j: np.ndarray
    @property
    def n_edges(self) -> int:
        return int(self.i.shape[0])

def pairwise_distances_subset(X: np.ndarray, edges: EdgeIndex) -> np.ndarray:
    d = X[edges.i] - X[edges.j]
    return np.linalg.norm(d, axis=-1)
