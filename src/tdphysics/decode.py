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
    ring_spec: Optional[dict] = None,
    ring_emit_every: Optional[int] = None,
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

    # Optional residue-network ring emission (small K nodes + M edges) for UI.
    _ring = ring_spec or None
    _ring_node_idx = None
    _ring_edges_abs = None
    _ring_edges_node = None
    if isinstance(_ring, dict):
        try:
            _ring_node_idx = np.asarray(_ring.get("node_idx", None), dtype=int) if _ring.get("node_idx", None) is not None else None
            _ring_edges_abs = np.asarray(_ring.get("edges_abs", None), dtype=int) if _ring.get("edges_abs", None) is not None else None
            _ring_edges_node = np.asarray(_ring.get("edges_node", None), dtype=int) if _ring.get("edges_node", None) is not None else None
        except Exception:
            _ring_node_idx = None
            _ring_edges_abs = None
            _ring_edges_node = None

    _ring_prev_d = None
    _ring_ei_t = None
    _ring_ej_t = None
    _ring_ea0_t = None
    _ring_ea1_t = None
    if _ring_edges_abs is not None and _ring_edges_node is not None:
        try:
            _ring_ei_t = torch.tensor(_ring_edges_node[:, 0], dtype=torch.long, device=device_t)
            _ring_ej_t = torch.tensor(_ring_edges_node[:, 1], dtype=torch.long, device=device_t)
            _ring_ea0_t = torch.tensor(_ring_edges_abs[:, 0], dtype=torch.long, device=device_t)
            _ring_ea1_t = torch.tensor(_ring_edges_abs[:, 1], dtype=torch.long, device=device_t)
        except Exception:
            _ring_ei_t = _ring_ej_t = _ring_ea0_t = _ring_ea1_t = None

    def _maybe_ring_payload() -> Optional[dict]:
        nonlocal _ring_prev_d
        if (_ring_node_idx is None) or (_ring_ei_t is None) or (_ring_ea0_t is None) or (_ring_ea1_t is None):
            return None
        try:
            Xa = X[_ring_ea0_t, :]
            Xb = X[_ring_ea1_t, :]
            d = torch.linalg.norm(Xa - Xb, dim=1)
            if _ring_prev_d is None:
                dw = torch.zeros_like(d)
            else:
                dw = torch.abs(d - _ring_prev_d)
            _ring_prev_d = d.detach()

            K = int(_ring_node_idx.shape[0])
            node = torch.zeros((K,), dtype=torch.float32, device=device_t)
            node.index_add_(0, _ring_ei_t, dw)
            node.index_add_(0, _ring_ej_t, dw)

            eps = 1e-9
            if dw.numel() > 0:
                try:
                    q = torch.quantile(dw, 0.95).item()
                except Exception:
                    q = torch.max(dw).item()
                q = float(q) + eps
                dw_n = torch.clamp(dw / q, 0.0, 1.0)
            else:
                dw_n = dw

            if node.numel() > 0:
                try:
                    qn = torch.quantile(node, 0.95).item()
                except Exception:
                    qn = torch.max(node).item()
                qn = float(qn) + eps
                node_n = torch.clamp(node / qn, 0.0, 1.0)
            else:
                node_n = node

            return {
                "ring_nodes": node_n.detach().cpu().numpy().astype(np.float32).tolist(),
                "ring_ei": _ring_edges_node[:, 0].astype(int).tolist(),
                "ring_ej": _ring_edges_node[:, 1].astype(int).tolist(),
                "ring_ew": dw_n.detach().cpu().numpy().astype(np.float32).tolist(),
            }
        except Exception:
            return None

    X0 = torch.tensor(X_init, dtype=torch.float32, device=device_t)
    X = X0.clone().detach().requires_grad_(True)

    # Guardrails: keep a best-known finite solution so decode never returns NaNs.
    best_loss = float("inf")
    X_best = X0.clone().detach()

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

        # If the objective goes non-finite, abort and return the best-known finite structure.
        if (not torch.isfinite(loss)) or (not torch.isfinite(X).all()):
            _emit(progress_cb, 'decode.abort', p=si / float(max(1, steps)), step=si, steps=steps, reason='non_finite')
            break

        loss.backward()
        torch.nn.utils.clip_grad_norm_([X], 10.0)
        opt.step()

        # Track best finite iterate.
        try:
            cur_loss = float(loss.item())
            if np.isfinite(cur_loss) and cur_loss < best_loss and torch.isfinite(X).all():
                best_loss = cur_loss
                X_best = X.detach().clone()
        except Exception:
            pass

        if (si == 1) or (si == steps) or (si % report_every == 0):

            rp = None
            # Optionally emit ring signals at a stride (defaults to report cadence).
            try:
                stride = int(ring_emit_every) if ring_emit_every is not None else 1
            except Exception:
                stride = 1
            if (_ring is not None) and (stride > 0) and ((si % stride) == 0):
                rp = _maybe_ring_payload()
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
                **(rp or {}),
            )

    _emit(progress_cb, 'decode.done', p=1.0, steps=steps)
    # Return best known finite structure (falls back to init if needed).
    try:
        out = X_best.detach().cpu().numpy()
        if np.isfinite(out).all():
            return out
    except Exception:
        pass
    return X0.detach().cpu().numpy()
