from __future__ import annotations

"""Future-focused training + prediction utilities.

This module is an **add-on** layer that you can import and call from the app
without rewriting the existing pipeline.

It provides:
  1) Beam-search rollout for higher-stability long horizons
  2) Physics-plus decode terms (elastic + sticky priors)
  3) Multi-model PDB export (a playable "movie")
  4) Optional combined-token energy shaping for training

The philosophy is simple:
  - base model learns *sequence likelihood*
  - physics priors penalize implausible futures
  - beam search explores multiple futures and keeps the best

All functions are self-contained and only rely on existing tdphysics modules.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .dataset import LevelSpec
from .model import energy_biased_logits
from .decode import DecodeWeights
from .decode import decode_coordinates_from_distances as _decode_base
from .io_pdb import write_ca_pdb


ProgressCB = Callable[[str, dict], None]


def _emit(cb: Optional[ProgressCB], event: str, **payload) -> None:
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:
        return


@dataclass
class PhysicsPriors:
    """Edge-level priors derived from the trajectory distances."""

    d0: np.ndarray  # reference distances per edge
    k: np.ndarray  # stiffness per edge (normalized)
    sticky_mask: np.ndarray  # bool per edge
    sticky_margin: float  # allowed slack beyond d0
    cutoff: float  # contact cutoff used to compute occupancy
    occupancy: np.ndarray  # fraction of frames where edge is within cutoff


def compute_physics_priors(
    d_series: np.ndarray,
    cutoff: float = 8.0,
    sticky_q: float = 0.80,
    sticky_margin: float = 1.2,
    eps: float = 1e-6,
) -> PhysicsPriors:
    """Compute elastic stiffness + sticky contact priors from edge-distance time series.

    Parameters
    ----------
    d_series:
        Array of shape [T, E] where E is #edges.
    cutoff:
        Distance threshold used to define a contact.
    sticky_q:
        An edge is "sticky" if its contact occupancy >= sticky_q.
    sticky_margin:
        How much an edge is allowed to stretch beyond its reference distance d0
        before a sticky penalty kicks in.
    """
    D = np.asarray(d_series, dtype=np.float32)
    d0 = D.mean(axis=0)
    var = D.var(axis=0)
    k = 1.0 / (var + float(eps))
    # normalize stiffness so typical magnitude is ~1
    k = k / (np.mean(k) + float(eps))
    occ = (D < float(cutoff)).mean(axis=0)
    sticky = occ >= float(sticky_q)
    return PhysicsPriors(
        d0=d0.astype(np.float32),
        k=k.astype(np.float32),
        sticky_mask=sticky.astype(bool),
        sticky_margin=float(sticky_margin),
        cutoff=float(cutoff),
        occupancy=occ.astype(np.float32),
    )


def _elastic_energy(d: np.ndarray, pri: PhysicsPriors) -> float:
    dd = np.asarray(d, dtype=np.float32) - pri.d0
    return float(np.mean(pri.k * (dd * dd)))


def _sticky_violation(d: np.ndarray, pri: PhysicsPriors) -> float:
    if not np.any(pri.sticky_mask):
        return 0.0
    d = np.asarray(d, dtype=np.float32)
    slack = pri.d0 + float(pri.sticky_margin)
    v = np.maximum(0.0, d - slack)
    return float(np.mean((v[pri.sticky_mask]) ** 2))


def combine_token_energies(
    token_energy: np.ndarray,
    d_centroids: np.ndarray,
    pri: PhysicsPriors,
    w_elastic: float = 0.35,
    w_sticky: float = 0.25,
) -> np.ndarray:
    """Create a single energy per token that includes elastic + sticky priors.

    This keeps training code untouched: you simply pass the returned array
    as `token_energy` into the existing trainer.
    """
    E0 = np.asarray(token_energy, dtype=np.float32)
    V = int(min(len(E0), d_centroids.shape[0]))
    E = E0[:V].copy()
    for t in range(V):
        E[t] = E[t] + float(w_elastic) * _elastic_energy(d_centroids[t], pri) + float(w_sticky) * _sticky_violation(
            d_centroids[t], pri
        )
    return E


@dataclass
class BeamConfig:
    beam_width: int = 8
    topk: int = 24
    work_penalty: float = 0.50  # penalize increases in energy relative to current token
    length_norm: float = 0.0  # 0 = off; >0 favors shorter steps by normalizing score


def beam_rollout_tokens(
    model,
    tokens_seed: np.ndarray,
    levels: List[LevelSpec],
    token_energy: np.ndarray,
    horizon_steps: int,
    context: int,
    beta_energy: float = 1.0,
    temperature: float = 1.0,
    device: str = "cpu",
    beam: BeamConfig = BeamConfig(),
    progress_cb: Optional[ProgressCB] = None,
) -> Dict[str, np.ndarray]:
    """Beam-search token rollout with energy guidance.

    Returns dict with:
      - tokens: [context + steps]
      - entropies: [steps]
      - levels_used: [steps]
      - beam_scores: [steps] (best beam score per step)
    """
    from .predict import greedy_level_schedule

    device_t = torch.device(device)
    model = model.to(device_t)
    model.eval()

    V = int(model.vocab_size)
    E = torch.tensor(np.asarray(token_energy[:V], dtype=np.float32), device=device_t)

    plan = greedy_level_schedule(int(horizon_steps), levels)
    n_steps = len(plan)

    seed = list(np.asarray(tokens_seed, dtype=np.int64).tolist())
    if len(seed) < context:
        seed = [seed[0]] * (context - len(seed)) + seed

    # Beam state: list of (seq, score)
    beams: List[Tuple[List[int], float]] = [(seed.copy(), 0.0)]
    entropies: List[float] = []
    levels_used: List[int] = []
    best_scores: List[float] = []

    _emit(progress_cb, "predict.start", p=0.0, n_steps=n_steps, mode="beam")

    for si, li in enumerate(plan, start=1):
        new_beams: List[Tuple[List[int], float]] = []
        level_id = torch.tensor([int(li)], dtype=torch.long, device=device_t)

        # expand each beam
        for seq, score in beams:
            ctx = seq[-context:]
            x = torch.tensor(ctx, dtype=torch.long, device=device_t).unsqueeze(0)
            logits = model(x, level_id)
            logits = energy_biased_logits(logits, E, beta=float(beta_energy))

            # optional temperature (softens logits before top-k)
            if float(temperature) > 0:
                logits = logits / float(max(1e-6, temperature))

            logp = F.log_softmax(logits, dim=-1).squeeze(0)  # [V]
            # top-k expansions
            k = int(min(max(1, beam.topk), V))
            vals, idx = torch.topk(logp, k=k, dim=-1)

            cur_tok = int(ctx[-1])
            curE = float(E[cur_tok].item())

            for lp, nxt in zip(vals.tolist(), idx.tolist()):
                nxt = int(nxt)
                nxtE = float(E[nxt].item())
                # work penalty: discourage uphill energy jumps
                work = max(0.0, nxtE - curE)
                new_score = float(score) + float(lp) - float(beam.work_penalty) * float(work)

                new_seq = seq + [nxt]
                new_beams.append((new_seq, new_score))

        # keep best beams
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[: int(max(1, beam.beam_width))]

        # diagnostics based on best beam's distribution
        # (entropy is approximated from best-beam logits at this step)
        best_seq, best_score = beams[0]
        best_scores.append(float(best_score))
        levels_used.append(int(li))

        # entropy computed from the *best beam* ctx
        ctx = best_seq[-context - 1 : -1]
        x = torch.tensor(ctx, dtype=torch.long, device=device_t).unsqueeze(0)
        logits = model(x, level_id)
        logits = energy_biased_logits(logits, E, beta=float(beta_energy))
        p = F.softmax(logits, dim=-1)
        H = float((-p * torch.log(p + 1e-12)).sum(dim=-1).item())
        entropies.append(H)

        if (si == 1) or (si == n_steps) or (si % max(1, n_steps // 120) == 0):
            _emit(
                progress_cb,
                "predict.step",
                p=si / float(max(1, n_steps)),
                step=si,
                n_steps=n_steps,
                entropy=H,
                level_id=int(li),
                token=int(best_seq[-1]),
                beam_score=float(best_score),
            )

    _emit(progress_cb, "predict.done", p=1.0, n_steps=n_steps, mode="beam")

    best = beams[0][0]
    return {
        "tokens": np.asarray(best, dtype=np.int64),
        "entropies": np.asarray(entropies, dtype=np.float32),
        "levels_used": np.asarray(levels_used, dtype=np.int64),
        "beam_scores": np.asarray(best_scores, dtype=np.float32),
    }


@dataclass
class DecodePhysicsPlus:
    """Extra decode terms on top of the base decoder."""

    w_elastic: float = 0.25
    w_sticky: float = 0.25


def decode_physicsplus(
    d_target: np.ndarray,
    X_init: np.ndarray,
    edges,
    bond_mask: np.ndarray,
    bond_ref: np.ndarray,
    pri: PhysicsPriors,
    w: DecodeWeights = DecodeWeights(),
    w_plus: DecodePhysicsPlus = DecodePhysicsPlus(),
    steps: int = 250,
    lr: float = 5e-2,
    device: str = "cpu",
    progress_cb: Optional[ProgressCB] = None,
) -> np.ndarray:
    """Decode with additional elastic + sticky penalties.

    We reuse the same optimizer structure as the base decoder but add two
    extra terms. Implemented separately to keep the original decoder stable.
    """
    device_t = torch.device(device)
    X0 = torch.tensor(np.asarray(X_init, dtype=np.float32), device=device_t)
    X = X0.clone().detach().requires_grad_(True)

    # Guardrails: keep a best-known finite solution so long-horizon decodes
    # don't poison downstream RMSD/movie export with NaNs.
    best_loss = float("inf")
    X_best = X0.clone().detach()

    d_tar = torch.tensor(np.asarray(d_target, dtype=np.float32), device=device_t)
    k = torch.tensor(np.asarray(pri.k, dtype=np.float32), device=device_t)
    d0 = torch.tensor(np.asarray(pri.d0, dtype=np.float32), device=device_t)
    sticky = torch.tensor(np.asarray(pri.sticky_mask, dtype=np.bool_), device=device_t)
    slack = d0 + float(pri.sticky_margin)

    bond_mask_t = torch.tensor(np.asarray(bond_mask, dtype=np.bool_), device=device_t)
    bond_ref_t = torch.tensor(np.asarray(bond_ref, dtype=np.float32), device=device_t)

    # Precompute edge indices
    ii = torch.tensor(edges.i, device=device_t)
    jj = torch.tensor(edges.j, device=device_t)

    opt = torch.optim.Adam([X], lr=float(lr))
    steps = int(steps)
    report_every = max(1, steps // 60)

    _emit(progress_cb, "decode.start", p=0.0, steps=steps, mode="physicsplus")

    for si in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        d = torch.linalg.norm(X[ii] - X[jj], dim=-1)
        fit = torch.mean((d - d_tar) ** 2)

        bond = torch.tensor(0.0, device=device_t)
        if bond_mask_t.any():
            db = d[bond_mask_t]
            bond = torch.mean((db - bond_ref_t) ** 2)

        rep = torch.mean(torch.nn.functional.softplus(float(w.r_min) - d) ** 2)
        smooth = torch.mean((X - X0) ** 2)

        elastic = torch.mean(k * (d - d0) ** 2)

        sticky_pen = torch.tensor(0.0, device=device_t)
        if sticky.any():
            v = torch.relu(d - slack)
            sticky_pen = torch.mean((v[sticky]) ** 2)

        loss = (
            float(w.w_fit) * fit
            + float(w.w_bond) * bond
            + float(w.w_rep) * rep
            + float(w.w_smooth) * smooth
            + float(w_plus.w_elastic) * elastic
            + float(w_plus.w_sticky) * sticky_pen
        )

        # If the objective or coordinates become non-finite, abort and return
        # the best-known finite iterate.
        if (not torch.isfinite(loss)) or (not torch.isfinite(X).all()):
            _emit(progress_cb, "decode.abort", p=si / float(max(1, steps)), step=si, steps=steps, reason="non_finite")
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
            _emit(
                progress_cb,
                "decode.step",
                p=si / float(max(1, steps)),
                step=si,
                steps=steps,
                loss=float(loss.item()),
                fit=float(fit.item()),
                bond=float(bond.item()),
                rep=float(rep.item()),
                smooth=float(smooth.item()),
                elastic=float(elastic.item()),
                sticky=float(sticky_pen.item()),
            )

    _emit(progress_cb, "decode.done", p=1.0, steps=steps, mode="physicsplus")
    # Return best known finite structure (falls back to init if needed).
    try:
        out = X_best.detach().cpu().numpy()
        if np.isfinite(out).all():
            return out
    except Exception:
        pass
    return X0.detach().cpu().numpy()


def write_multimodel_ca_pdb(
    path: str,
    frames: np.ndarray,
    resids: np.ndarray,
    resnames: Sequence[str],
    chain_id: str = "A",
    bfactor: Optional[np.ndarray] = None,
) -> None:
    """Write a multi-model CA PDB that plays like a movie in PyMOL/VMD."""
    frames = np.asarray(frames, dtype=float)
    if frames.ndim != 3:
        raise ValueError("frames must be [F, N, 3]")
    Fm, N, _ = frames.shape
    if bfactor is None:
        bfactor = np.zeros((Fm, N), dtype=float)
    bfactor = np.asarray(bfactor, dtype=float)
    if bfactor.shape != (Fm, N):
        raise ValueError("bfactor must be [F, N]")

    lines: List[str] = []
    for mi in range(Fm):
        lines.append(f"MODEL     {mi+1}")
        atom_serial = 1
        for i in range(N):
            x, y, z = frames[mi, i]
            resn = str(resnames[i])[:3].rjust(3)
            resid = int(resids[i])
            bf = float(bfactor[mi, i])
            lines.append(
                f"ATOM  {atom_serial:5d}  CA  {resn} {chain_id}{resid:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bf:6.2f}           C"
            )
            atom_serial += 1
        lines.append("ENDMDL")
    lines.append("END")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def predict_future_movie(
    engine,
    horizon_time: float,
    n_frames: int = 20,
    use_beam: bool = True,
    beam_cfg: BeamConfig = BeamConfig(),
    beta_energy: float = 1.0,
    temperature: float = 1.0,
    greedy: bool = False,
    decode_w: DecodeWeights = DecodeWeights(),
    decode_steps: int = 200,
    decode_physics: bool = True,
    priors: Optional[PhysicsPriors] = None,
    device: Optional[str] = None,
    ring_spec: Optional[dict] = None,
    ring_emit_every: int = 1,
    progress_cb: Optional[ProgressCB] = None,
) -> Dict[str, object]:
    """Predict a future trajectory and decode multiple frames (a "movie").

    The movie is produced by decoding a subset of predicted tokens along the
    rollout. For best visual continuity, each frame is decoded using the
    previous decoded coordinates as initialization.
    """
    if device is None:
        device = engine.train_cfg.device

    horizon_steps = max(1, int(round(float(horizon_time) / float(engine.traj.dt))))
    context = int(engine.train_cfg.context)
    seed_tokens = engine.tok.tokens[-context:]

    # Optional residue-network ring spec (small subset of residues + edges),
    # used to emit live "interaction-change" signals for the UI.
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

    def _ring_payload(X_cur: np.ndarray, X_prev: Optional[np.ndarray]) -> Optional[dict]:
        if _ring_node_idx is None or _ring_edges_abs is None or _ring_edges_node is None:
            return None
        try:
            ea = _ring_edges_abs
            en = _ring_edges_node
            # distances along ring edges
            v1 = X_cur[ea[:, 0], :] - X_cur[ea[:, 1], :]
            d1 = np.linalg.norm(v1, axis=1)
            if X_prev is None:
                dw = np.zeros_like(d1)
            else:
                v0 = X_prev[ea[:, 0], :] - X_prev[ea[:, 1], :]
                d0 = np.linalg.norm(v0, axis=1)
                dw = np.abs(d1 - d0)

            # node activity = incident edge delta-sum
            K = int(_ring_node_idx.shape[0])
            node = np.zeros((K,), dtype=np.float32)
            # en endpoints are in [0..K-1]
            np.add.at(node, en[:, 0], dw.astype(np.float32))
            np.add.at(node, en[:, 1], dw.astype(np.float32))

            # robust normalize (avoid flicker on single spikes)
            eps = 1e-9
            if dw.size > 0:
                q = float(np.percentile(dw, 95)) + eps
                dw_n = np.clip(dw / q, 0.0, 1.0).astype(np.float32)
            else:
                dw_n = dw.astype(np.float32)
            if node.size > 0:
                qn = float(np.percentile(node, 95)) + eps
                node_n = np.clip(node / qn, 0.0, 1.0).astype(np.float32)
            else:
                node_n = node.astype(np.float32)

            return {
                "ring_nodes": node_n.tolist(),
                "ring_ei": en[:, 0].astype(int).tolist(),
                "ring_ej": en[:, 1].astype(int).tolist(),
                "ring_ew": dw_n.tolist(),
            }
        except Exception:
            return None

    # Reference for |Δd| emission across decoded frames
    X_prev_ring = engine.traj.X[-1].copy()


    # Rollout
    if use_beam and (not greedy):
        roll = beam_rollout_tokens(
            engine.model,
            seed_tokens,
            engine.levels,
            engine.token_energy,
            horizon_steps=horizon_steps,
            context=context,
            beta_energy=float(beta_energy),
            temperature=float(temperature),
            device=str(device),
            beam=beam_cfg,
            progress_cb=progress_cb,
        )
    else:
        from .predict import rollout_tokens, RolloutConfig

        roll = rollout_tokens(
            engine.model,
            seed_tokens,
            engine.levels,
            engine.token_energy,
            horizon_steps=horizon_steps,
            context=context,
            cfg=RolloutConfig(beta_energy=float(beta_energy), temperature=float(temperature), greedy=bool(greedy)),
            device=str(device),
            progress_cb=progress_cb,
        )

    seq = np.asarray(roll["tokens"], dtype=np.int64)
    # choose token indices to decode (exclude the seed prefix)
    start = max(0, len(seq) - horizon_steps)  # approximate start of predicted region
    start = max(start, context)  # keep seed prefix untouched
    idxs = np.linspace(start, len(seq) - 1, num=int(max(2, n_frames)), dtype=int)
    idxs = np.unique(idxs)

    # priors on edges
    if priors is None:
        priors = compute_physics_priors(engine.traj.d)

    frames: List[np.ndarray] = []
    tokens_used: List[int] = []
    X_init = engine.traj.X[-1].copy()

    _emit(progress_cb, "movie.start", p=0.0, n_frames=int(len(idxs)))
    for fi, ti in enumerate(idxs, start=1):
        tok_id = int(seq[int(ti)])
        d_hat = engine.tok.pca.inverse_transform(engine.tok.centroids_z[tok_id]).astype(np.float32)

        if decode_physics:
            X_rec = decode_physicsplus(
                d_hat,
                X_init,
                engine.traj.edges,
                engine.stats.bond_edge_mask,
                engine.stats.bond_ref,
                pri=priors,
                w=decode_w,
                steps=int(decode_steps),
                lr=5e-2,
                device=str(device),
                progress_cb=None,  # keep UI fast; movie loop has its own progress
            )
        else:
            X_rec = _decode_base(
                d_hat,
                X_init,
                engine.traj.edges,
                engine.stats.bond_edge_mask,
                engine.stats.bond_ref,
                w=decode_w,
                steps=int(decode_steps),
                lr=5e-2,
                device=str(device),
                progress_cb=None,
            )

        frames.append(X_rec)
        tokens_used.append(tok_id)
        X_init = X_rec  # chained init for continuity

        rp = _ring_payload(X_rec, X_prev_ring)
        if (rp is not None) and (int(ring_emit_every) > 0) and ((fi % int(ring_emit_every)) == 0):
            _emit(progress_cb, "movie.frame", p=fi / float(max(1, len(idxs))), frame=fi, n_frames=len(idxs), token=tok_id, **rp)
        else:
            _emit(progress_cb, "movie.frame", p=fi / float(max(1, len(idxs))), frame=fi, n_frames=len(idxs), token=tok_id)

        X_prev_ring = X_rec

    _emit(progress_cb, "movie.done", p=1.0, n_frames=int(len(idxs)))

    frames_np = np.stack(frames, axis=0).astype(np.float32)

    return {
        "rollout": roll,
        "frames": frames_np,
        "frame_tokens": np.asarray(tokens_used, dtype=np.int64),
        "decode_physics": bool(decode_physics),
    }


def export_future_movie_pdb(
    engine,
    movie: Dict[str, object],
    out_pdb: str,
    bfactor_mode: str = "token_energy",
) -> None:
    frames = np.asarray(movie["frames"], dtype=np.float32)
    toks = np.asarray(movie["frame_tokens"], dtype=np.int64)
    Fm, N, _ = frames.shape

    if bfactor_mode == "token_energy":
        bf = np.zeros((Fm, N), dtype=float)
        for i in range(Fm):
            bf[i, :] = float(engine.token_energy[int(toks[i])])
    else:
        bf = np.zeros((Fm, N), dtype=float)

    write_multimodel_ca_pdb(out_pdb, frames, engine.traj.resids, engine.traj.resnames, chain_id="A", bfactor=bf)


# ---------------------------------------------------------------------------
# Optional: physics-plus training (pipeline integration)
# ---------------------------------------------------------------------------

def build_engine_future(
    traj,
    m: int = 48,
    K: int = 256,
    energy_w=None,
    level_targets: Optional[List[float]] = None,
    train_cfg=None,
    seed: int = 0,
    # priors
    cutoff: float = 8.0,
    sticky_q: float = 0.80,
    sticky_margin: float = 1.2,
    w_elastic: float = 0.35,
    w_sticky: float = 0.25,
    progress_cb: Optional[ProgressCB] = None,
):
    """Build an engine like `tdphysics.pipeline.build_engine`, but shape training
    with **elastic + sticky** priors.

    This is intentionally implemented here (add-on) so your original pipeline
    remains stable. The returned engine is the same EngineArtifacts object, with
    extra attributes attached:
      - engine.physics_priors
      - engine.token_energy_raw
    """
    from .pipeline import propose_levels, EngineArtifacts
    from .tokenize import fit_tokenizer
    from .energy import compute_energy_stats, token_energies_from_centroids, EnergyWeights
    from .train import TrainConfig, train_multilevel

    if energy_w is None:
        energy_w = EnergyWeights()
    if train_cfg is None:
        train_cfg = TrainConfig(context=64, epochs=10, device="cpu")
    if level_targets is None:
        level_targets = [traj.dt, 10.0 * traj.dt, 100.0 * traj.dt, 1000.0 * traj.dt]

    _emit(progress_cb, "pipeline.tokenize.start", p=0.0)
    tok = fit_tokenizer(traj.d, m=m, K=K, seed=seed, progress_cb=progress_cb)

    _emit(progress_cb, "pipeline.energy.start", p=0.0)
    stats = compute_energy_stats(traj.d, traj.edges)
    token_energy_raw = token_energies_from_centroids(tok.centroids_z, tok.pca, stats, w=energy_w)

    pri = compute_physics_priors(traj.d, cutoff=cutoff, sticky_q=sticky_q, sticky_margin=sticky_margin)

    # centroid distances in the original edge-distance space
    d_centroids = tok.pca.inverse_transform(tok.centroids_z).astype(np.float32)
    token_energy = combine_token_energies(token_energy_raw, d_centroids, pri, w_elastic=w_elastic, w_sticky=w_sticky)
    _emit(progress_cb, "pipeline.energy.done", p=1.0)

    max_lag = int(len(tok.tokens) - train_cfg.context - 1)
    if max_lag < 1:
        raise ValueError(
            f"Trajectory too short for context={train_cfg.context}. "
            f"Need at least context+2 frames; got {len(tok.tokens)}."
        )
    levels = propose_levels(traj.dt, traj.time_unit, targets=level_targets, max_lag_steps=max_lag)

    _emit(progress_cb, "pipeline.train.start", p=0.0)
    model, report = train_multilevel(
        tokens=tok.tokens,
        levels=levels,
        token_energy=token_energy,
        codebook_z=tok.centroids_z,
        cfg=train_cfg,
        seed=seed,
        progress_cb=progress_cb,
    )
    _emit(progress_cb, "pipeline.train.done", p=1.0)
    _emit(progress_cb, "pipeline.done", p=1.0)

    engine = EngineArtifacts(
        traj=traj,
        tok=tok,
        stats=stats,
        token_energy=token_energy,
        levels=levels,
        model=model,
        train_report=report,
        train_cfg=train_cfg,
        energy_w=energy_w,
    )

    # Attach extras without changing EngineArtifacts signature
    setattr(engine, "physics_priors", pri)
    setattr(engine, "token_energy_raw", token_energy_raw)
    return engine
