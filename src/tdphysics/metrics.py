from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KabschResult:
    rmsd: float
    R: np.ndarray
    t: np.ndarray
    Y_aligned: np.ndarray


def kabsch_align(X: np.ndarray, Y: np.ndarray) -> KabschResult:
    """Align Y onto X (both shape [N,3]) using the Kabsch algorithm."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.ndim != 2 or Y.ndim != 2 or X.shape != Y.shape or X.shape[1] != 3:
        raise ValueError(f"X and Y must both be [N,3] with same shape; got {X.shape} and {Y.shape}")

    X_mean = X.mean(axis=0)
    Y_mean = Y.mean(axis=0)
    Xc = X - X_mean
    Yc = Y - Y_mean

    # covariance (Y -> X)
    H = Yc.T @ Xc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection correction
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = X_mean - (Y_mean @ R)
    Y_aligned = (Y @ R) + t

    diff = Y_aligned - X
    rmsd = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return KabschResult(rmsd=rmsd, R=R, t=t, Y_aligned=Y_aligned)


def rmsd_kabsch(X: np.ndarray, Y: np.ndarray) -> float:
    """RMSD after optimal rigid-body alignment (Kabsch)."""
    return kabsch_align(X, Y).rmsd


def per_site_displacement(X: np.ndarray, Y: np.ndarray, align: bool = True) -> np.ndarray:
    """Per-site displacement magnitude between X and Y.

    If align=True, Y is Kabsch-aligned onto X first.
    Returns an array of shape [N] (same ordering as input sites).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if align:
        Y = kabsch_align(X, Y).Y_aligned
    return np.sqrt(np.sum((Y - X) ** 2, axis=1))
