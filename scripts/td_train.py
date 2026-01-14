#!/usr/bin/env python3
from __future__ import annotations

"""CLI: train a TDPhysics engine.

This script is intentionally path-robust (works from repo root without installing).
"""

import argparse
import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tdphysics.mdio import make_trajectory_data
from tdphysics.pipeline import build_engine
from tdphysics.train import TrainConfig
from tdphysics.energy import EnergyWeights

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--top", required=True)
    p.add_argument("--traj", default=None)
    p.add_argument("--time-total", type=float, default=None)
    p.add_argument("--time-unit", default="ns")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--out", default="engine_artifacts.npz")
    args = p.parse_args()
    traj = make_trajectory_data(args.top, args.traj, time_total=args.time_total, time_unit=args.time_unit)
    eng = build_engine(traj, train_cfg=TrainConfig(epochs=args.epochs), energy_w=EnergyWeights())
    np.savez_compressed(args.out, token_energy=eng.token_energy, centroids_z=eng.tok.centroids_z)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
