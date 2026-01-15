from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

from .utils import EdgeIndex, pairwise_distances_subset


ProgressCB = Callable[[str, dict], None]


def _emit(cb: Optional[ProgressCB], event: str, **payload) -> None:
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:
        return


@dataclass
class TrajectoryData:
    """In-memory trajectory container (typically CA-only)."""

    resids: np.ndarray
    resnames: np.ndarray
    X_ref: np.ndarray
    X: np.ndarray
    edges: EdgeIndex
    d: np.ndarray
    dt: float
    time_unit: str


def _require_mdanalysis() -> None:
    try:
        import MDAnalysis as mda  # noqa
    except Exception as e:
        raise RuntimeError("MDAnalysis is required. Install requirements.txt") from e


def _convert_ps_to_unit(dt_ps: float, time_unit: str) -> float:
    """MDAnalysis dt is typically in ps. Convert to requested time_unit."""
    tu = str(time_unit).lower()
    if tu == "ps":
        return float(dt_ps)
    if tu == "ns":
        return float(dt_ps) / 1000.0
    if tu == "us":
        return float(dt_ps) / 1e6
    if tu == "ms":
        return float(dt_ps) / 1e9
    if tu == "s":
        return float(dt_ps) / 1e12
    return float(dt_ps)


def load_ca_trajectory(
    top_path: str,
    traj_path: Optional[str],
    selection: str = "protein and name CA",
    stride: int = 1,
    align: bool = True,
    progress_cb: Optional[ProgressCB] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
    """Load a trajectory as coordinates for a given selection (default: protein CA).

    Returns (resids, resnames, X[T,N,3], dt_ps_if_available).

    progress_cb events:
      - preprocess.load (p, frame, n_frames)
    """
    _require_mdanalysis()
    import MDAnalysis as mda
    from MDAnalysis.analysis import align as mda_align

    u = mda.Universe(top_path, traj_path) if traj_path else mda.Universe(top_path)
    ag = u.select_atoms(selection)
    if ag.n_atoms == 0:
        raise ValueError(f"Selection returned 0 atoms: {selection}")

    resids = ag.resids.copy()
    resnames = np.array([r.resname for r in ag.residues], dtype=object)

    dt_ps = None
    if traj_path:
        try:
            dt_ps = float(getattr(u.trajectory, "dt", None))
            if not np.isfinite(dt_ps):
                dt_ps = None
        except Exception:
            dt_ps = None

    if align and traj_path:
        ref = mda.Universe(top_path, traj_path)
        mda_align.AlignTraj(u, ref, select=selection, in_memory=False).run()

    # Stream frames (progress-friendly)
    frames = []
    # Best-effort frame count
    n_total = None
    if traj_path:
        try:
            n_total = int(u.trajectory.n_frames)
        except Exception:
            n_total = None

    # Step through with stride
    idx = 0
    for ts in u.trajectory[::max(1, int(stride))]:
        frames.append(ag.positions.astype(np.float32).copy())
        idx += 1
        denom = (n_total // max(1, int(stride))) if n_total else max(1, idx + 1)
        _emit(progress_cb, "preprocess.load", p=min(1.0, idx / float(max(1, denom))), frame=idx, n_frames=denom)

    X = np.stack(frames, axis=0)
    _emit(progress_cb, "preprocess.load", p=1.0, frame=X.shape[0], n_frames=X.shape[0])
    return resids, resnames, X, dt_ps


def build_fixed_knn_edges(X_ref: np.ndarray, k: int = 8, include_backbone: bool = True) -> EdgeIndex:
    """Build a symmetric fixed kNN edge set from the reference geometry."""
    from sklearn.neighbors import NearestNeighbors

    N = X_ref.shape[0]
    nn = NearestNeighbors(n_neighbors=min(int(k) + 1, N)).fit(X_ref)
    _, idx = nn.kneighbors(X_ref, return_distance=True)
    pairs = set()
    for i in range(N):
        for j in idx[i, 1:]:
            a, b = i, int(j)
            if a == b:
                continue
            if a > b:
                a, b = b, a
            pairs.add((a, b))
    if include_backbone:
        for i in range(N - 1):
            pairs.add((i, i + 1))
    pairs = sorted(pairs)
    ii = np.array([p[0] for p in pairs], dtype=int)
    jj = np.array([p[1] for p in pairs], dtype=int)
    return EdgeIndex(ii, jj)


def trajectory_to_distances(
    X: np.ndarray,
    edges: EdgeIndex,
    progress_cb: Optional[ProgressCB] = None,
) -> np.ndarray:
    """Convert trajectory coordinates to a fixed-edge distance signal matrix.

    progress_cb events:
      - preprocess.distances (p, frame, n_frames, edge0)
      - preprocess.done
    """
    T = int(X.shape[0])
    d = np.empty((T, edges.n_edges), dtype=np.float32)
    for t in range(T):
        d[t] = pairwise_distances_subset(X[t], edges)
        if (t == 0) or (t == T - 1) or (t % max(1, T // 120) == 0):
            _emit(
                progress_cb,
                "preprocess.distances",
                p=(t + 1) / float(max(1, T)),
                frame=t + 1,
                n_frames=T,
                edge0=float(d[t, 0]) if d.shape[1] > 0 else 0.0,
            )
    _emit(progress_cb, "preprocess.done", p=1.0)
    return d


def infer_dt(
    time_total: Optional[float],
    n_frames: int,
    fallback_dt: float = 1.0,
    dt_ps: Optional[float] = None,
    time_unit: str = "ns",
) -> float:
    """Infer dt in requested units.

    Priority:
      1) if time_total given -> time_total/(n_frames-1)
      2) else if dt_ps available from MDAnalysis -> convert ps->time_unit
      3) else fallback_dt
    """
    if time_total is not None and n_frames >= 2:
        return float(time_total) / float(n_frames - 1)
    if dt_ps is not None:
        return _convert_ps_to_unit(float(dt_ps), time_unit=time_unit)
    return float(fallback_dt)


def make_trajectory_data(
    top_path: str,
    traj_path: Optional[str],
    selection: str = "protein and name CA",
    stride: int = 1,
    time_total: Optional[float] = None,
    time_unit: str = "ns",
    k: int = 8,
    align: bool = True,
    progress_cb: Optional[ProgressCB] = None,
) -> TrajectoryData:
    _emit(progress_cb, "preprocess.start", p=0.0)
    resids, resnames, X, dt_ps = load_ca_trajectory(
        top_path, traj_path, selection=selection, stride=stride, align=align, progress_cb=progress_cb
    )
    X_ref = X[0].copy()
    edges = build_fixed_knn_edges(X_ref, k=k, include_backbone=True)
    d = trajectory_to_distances(X, edges, progress_cb=progress_cb)
    dt = infer_dt(time_total=time_total, n_frames=X.shape[0], fallback_dt=1.0, dt_ps=dt_ps, time_unit=time_unit)
    return TrajectoryData(resids=resids, resnames=resnames, X_ref=X_ref, X=X, edges=edges, d=d, dt=dt, time_unit=time_unit)
