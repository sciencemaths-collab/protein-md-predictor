from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict, Optional

import numpy as np

from .mdio import TrajectoryData
from .tokenize import fit_tokenizer, TokenizationResult
from .energy import (
    compute_energy_stats,
    token_energies_from_centroids,
    frame_energy,
    EnergyStats,
    EnergyWeights,
)
from .dataset import LevelSpec
from .train import train_multilevel, TrainConfig, TrainReport
from .predict import rollout_tokens, RolloutConfig
from .decode import decode_coordinates_from_distances, DecodeWeights
from .io_pdb import write_ca_pdb
from .triplet import triplet_from_latent

ProgressCB = Callable[[str, dict], None]


def _emit(cb: Optional[ProgressCB], event: str, **payload) -> None:
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:
        return


@dataclass
class EngineArtifacts:
    traj: TrajectoryData
    tok: TokenizationResult
    stats: EnergyStats
    token_energy: np.ndarray
    levels: List[LevelSpec]
    model: object
    train_report: TrainReport
    train_cfg: TrainConfig
    energy_w: EnergyWeights


def propose_levels(
    dt: float,
    time_unit: str,
    targets: List[float],
    max_lag_steps: Optional[int] = None,
) -> List[LevelSpec]:
    """Create level specs from target times (same unit as dt).

    If max_lag_steps is provided, lags are clipped to keep dataset valid.
    """
    levels: List[LevelSpec] = []
    for t in targets:
        g = max(1, int(round(float(t) / float(dt))))
        if max_lag_steps is not None:
            g = min(g, int(max_lag_steps))
        levels.append(LevelSpec(name=f"{t:g}{time_unit}", lag_steps=g, weight=1.0))
    uniq = {lv.lag_steps: lv for lv in levels}
    return sorted(uniq.values(), key=lambda x: x.lag_steps)


def compute_frame_energies(traj: TrajectoryData, stats: EnergyStats, w: EnergyWeights) -> np.ndarray:
    T = traj.d.shape[0]
    E = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        prev = traj.d[t - 1] if t > 0 else None
        E[t] = frame_energy(traj.d[t], prev, stats, w=w)
    return E


def build_engine(
    traj: TrajectoryData,
    m: int = 48,
    K: int = 256,
    energy_w: EnergyWeights = EnergyWeights(),
    level_targets: Optional[List[float]] = None,
    train_cfg: Optional[TrainConfig] = None,
    seed: int = 0,
    progress_cb: Optional[ProgressCB] = None,
) -> EngineArtifacts:
    """Full pipeline build:

      distances -> PCA -> KMeans tokens -> per-token energy -> multi-level transformer training

    progress_cb events are forwarded from tokenize/train plus:
      - pipeline.energy (p)
      - pipeline.done
    """
    if train_cfg is None:
        train_cfg = TrainConfig(context=64, epochs=10, device="cpu")

    if level_targets is None:
        level_targets = [traj.dt, 10.0 * traj.dt, 100.0 * traj.dt, 1000.0 * traj.dt]

    _emit(progress_cb, 'pipeline.tokenize.start', p=0.0)
    tok = fit_tokenizer(traj.d, m=m, K=K, seed=seed, progress_cb=progress_cb)

    _emit(progress_cb, 'pipeline.energy.start', p=0.0)
    stats = compute_energy_stats(traj.d, traj.edges)
    token_energy = token_energies_from_centroids(tok.centroids_z, tok.pca, stats, w=energy_w)
    _emit(progress_cb, 'pipeline.energy.done', p=1.0)

    max_lag = int(len(tok.tokens) - train_cfg.context - 1)
    if max_lag < 1:
        raise ValueError(
            f"Trajectory too short for context={train_cfg.context}. "
            f"Need at least context+2 frames; got {len(tok.tokens)}."
        )

    levels = propose_levels(traj.dt, traj.time_unit, targets=level_targets, max_lag_steps=max_lag)

    _emit(progress_cb, 'pipeline.train.start', p=0.0)
    model, report = train_multilevel(
        tokens=tok.tokens,
        levels=levels,
        token_energy=token_energy,
        codebook_z=tok.centroids_z,
        cfg=train_cfg,
        seed=seed,
        progress_cb=progress_cb,
    )
    _emit(progress_cb, 'pipeline.train.done', p=1.0)

    _emit(progress_cb, 'pipeline.done', p=1.0)

    return EngineArtifacts(
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


def predict_pause_structures(
    engine: EngineArtifacts,
    horizon_time: float,
    rollout_cfg: RolloutConfig = RolloutConfig(),
    decode_w: DecodeWeights = DecodeWeights(),
    decode_steps: int = 250,
    device: Optional[str] = None,
    context_override: Optional[int] = None,
    progress_cb: Optional[ProgressCB] = None,
    ring_spec: Optional[dict] = None,
    ring_emit_every: Optional[int] = None,
) -> Dict[str, object]:
    """Roll out tokens (physics-biased) and decode a single pause structure."""
    if device is None:
        device = engine.train_cfg.device

    horizon_steps = max(1, int(round(float(horizon_time) / float(engine.traj.dt))))
    context = int(context_override) if context_override is not None else int(engine.train_cfg.context)

    seed_tokens = engine.tok.tokens[-context:]

    roll = rollout_tokens(
        engine.model,
        seed_tokens,
        engine.levels,
        engine.token_energy,
        horizon_steps,
        context,
        rollout_cfg,
        device=device,
        progress_cb=progress_cb,
    )

    tok_last = int(roll["tokens"][-1])
    d_hat = engine.tok.pca.inverse_transform(engine.tok.centroids_z[tok_last]).astype(np.float32)

    X_init = engine.traj.X[-1].copy()
    X_rec = decode_coordinates_from_distances(
        d_hat,
        X_init,
        engine.traj.edges,
        engine.stats.bond_edge_mask,
        engine.stats.bond_ref,
        w=decode_w,
        steps=int(decode_steps),
        lr=5e-2,
        device=device,
        progress_cb=progress_cb,
        ring_spec=ring_spec,
        ring_emit_every=ring_emit_every,
    )

    tri = triplet_from_latent(engine.tok.z)
    return {"rollout": roll, "token_last": tok_last, "d_hat": d_hat, "X_rec": X_rec, "triplet": tri}


def export_pause_pdb(
    engine: EngineArtifacts,
    X_rec: np.ndarray,
    token_last: int,
    out_pdb: str,
    bfactor_mode: str = "token_energy",
) -> None:
    if bfactor_mode == "token_energy":
        bf = np.full((X_rec.shape[0],), float(engine.token_energy[token_last]), dtype=float)
    else:
        bf = np.zeros((X_rec.shape[0],), dtype=float)
    write_ca_pdb(out_pdb, X_rec, engine.traj.resids, engine.traj.resnames, bfactor=bf, chain_id="A")
