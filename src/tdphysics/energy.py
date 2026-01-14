from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from .utils import EdgeIndex, softplus

@dataclass
class EnergyStats:
    mean: np.ndarray
    std: np.ndarray
    bond_ref: np.ndarray
    bond_edge_mask: np.ndarray

@dataclass
class EnergyWeights:
    w_bond: float = 1.0
    w_rep: float = 0.2
    w_contact: float = 0.5
    w_smooth: float = 0.05
    r_min: float = 3.2

def compute_energy_stats(d: np.ndarray, edges: EdgeIndex) -> EnergyStats:
    mean = d.mean(axis=0)
    std = d.std(axis=0) + 1e-6
    bond_mask = (np.abs(edges.i - edges.j) == 1)
    bond_ref = mean[bond_mask].copy()
    return EnergyStats(mean=mean, std=std, bond_ref=bond_ref, bond_edge_mask=bond_mask)

def frame_energy(d_t: np.ndarray, d_prev: Optional[np.ndarray], stats: EnergyStats, w: EnergyWeights = EnergyWeights()) -> float:
    d_t = np.asarray(d_t)
    bond = 0.0
    if stats.bond_edge_mask.any():
        db = d_t[stats.bond_edge_mask]
        bond = float(((db - stats.bond_ref) ** 2).mean())
    rep = float((softplus(w.r_min - d_t) ** 2).mean())
    z = (d_t - stats.mean) / stats.std
    contact = float((z ** 2).mean())
    smooth = 0.0
    if d_prev is not None:
        smooth = float(((d_t - d_prev) ** 2).mean())
    return w.w_bond * bond + w.w_rep * rep + w.w_contact * contact + w.w_smooth * smooth

def token_energies_from_centroids(centroids_z: np.ndarray, pca_model, stats: EnergyStats, w: EnergyWeights = EnergyWeights()) -> np.ndarray:
    d_hat = pca_model.inverse_transform(centroids_z)
    E = np.empty((d_hat.shape[0],), dtype=np.float32)
    for k in range(d_hat.shape[0]):
        E[k] = frame_energy(d_hat[k], None, stats, w=w)
    return E
