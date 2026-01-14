from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch

from .utils import EdgeIndex

ProgressCB = Callable[[str, dict], None]


def _emit(cb: Optional[ProgressCB], event: str, **payload) -> None:
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:
        return


@dataclass
class DecodeWeights:
    w_fit: float = 1.0
    w_bond: float = 1.0
    w_rep: float = 0.2
    w_smooth: float = 0.1
    r_min: float = 3.2


def _dist_edges_torch(X: torch.Tensor, edges: EdgeIndex) -> torch.Tensor:
    ii = torch.tensor(edges.i, device=X.device)
    jj = torch.tensor(edges.j, device=X.device)
    d = X[ii] - X[jj]
    return torch.linalg.norm(d, dim=-1)


def decode_coordinates_from_distances(
    d_target: np.ndarray,
    X_init: np.ndarray,
    edges: EdgeIndex,
    bond_mask: np.ndarray,
    bond_ref: np.ndarray,
    w: DecodeWeights = DecodeWeights(),
    steps: int = 250,
    lr: float = 5e-2,
    device: str = "cpu",
    progress_cb: Optional[ProgressCB] = None,
    report_every: Optional[int] = None,
) -> np.ndarray:
    """Constrained reconstruction of coordinates from target distances.

    progress_cb events:
      - decode.step (p, step, steps, loss, fit, bond, rep, smooth)
      - decode.done

    Notes:
      - This is a *refinement* step, not force-field MD.
      - The objective enforces distance fit + bond prior + steric repulsion + temporal smoothness.
    """
    device_t = torch.device(device)

    X0 = torch.tensor(X_init, dtype=torch.float32, device=device_t)
    X = X0.clone().detach().requires_grad_(True)

    d_tar = torch.tensor(np.asarray(d_target, dtype=np.float32), dtype=torch.float32, device=device_t)
    bond_mask_t = torch.tensor(np.asarray(bond_mask, dtype=np.bool_), device=device_t)
    bond_ref_t = torch.tensor(np.asarray(bond_ref, dtype=np.float32), device=device_t)

    opt = torch.optim.Adam([X], lr=float(lr))

    steps = int(steps)
    if report_every is None:
        report_every = max(1, steps // 50)

    _emit(progress_cb, 'decode.start', p=0.0, steps=steps)

    for si in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        d = _dist_edges_torch(X, edges)
        fit = torch.mean((d - d_tar) ** 2)

        bond = torch.tensor(0.0, device=device_t)
        if bond_mask_t.any():
            db = d[bond_mask_t]
            bond = torch.mean((db - bond_ref_t) ** 2)

        rep = torch.mean(torch.nn.functional.softplus(float(w.r_min) - d) ** 2)
        smooth = torch.mean((X - X0) ** 2)

        loss = float(w.w_fit) * fit + float(w.w_bond) * bond + float(w.w_rep) * rep + float(w.w_smooth) * smooth

        loss.backward()
        torch.nn.utils.clip_grad_norm_([X], 10.0)
        opt.step()

        if (si == 1) or (si == steps) or (si % report_every == 0):
            _emit(
                progress_cb,
                'decode.step',
                p=si / float(max(1, steps)),
                step=si,
                steps=steps,
                loss=float(loss.item()),
                fit=float(fit.item()),
                bond=float(bond.item()),
                rep=float(rep.item()),
                smooth=float(smooth.item()),
            )

    _emit(progress_cb, 'decode.done', p=1.0, steps=steps)
    return X.detach().cpu().numpy()
