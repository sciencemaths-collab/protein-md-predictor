from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .utils import normalize01

@dataclass
class TripletDecomposition:
    plus: np.ndarray
    zero: np.ndarray
    minus: np.ndarray
    coherence: np.ndarray

def triplet_from_latent(z: np.ndarray) -> TripletDecomposition:
    z = np.asarray(z, dtype=float)
    T, m = z.shape
    dz = np.zeros_like(z)
    dz[1:] = z[1:] - z[:-1]
    nrm = np.linalg.norm(dz, axis=-1, keepdims=True) + 1e-12
    u = dz / nrm
    proj = np.sum(z * u, axis=-1)
    z_par = proj[:, None] * u
    z_perp = z - z_par
    zero = np.linalg.norm(z_perp, axis=-1)
    plus = np.maximum(proj, 0.0)
    minus = np.maximum(-proj, 0.0)
    coh = 1.0 - normalize01(zero)
    return TripletDecomposition(plus=plus, zero=zero, minus=minus, coherence=np.clip(coh, 0.0, 1.0))
