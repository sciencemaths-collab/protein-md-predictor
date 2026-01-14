"""Tokenized Dynamics Physics v4.

Public API intentionally small.

Core pipeline:
  - tdphysics.pipeline.build_engine
  - tdphysics.pipeline.predict_pause_structures

Optional "future" add-ons:
  - tdphysics.future.predict_future_movie
  - tdphysics.future.export_future_movie_pdb
"""

from .pipeline import build_engine, predict_pause_structures, export_pause_pdb

# Add-on utilities (safe to ignore if you don't need them)
from .future import (
    build_engine_future,
    predict_future_movie,
    export_future_movie_pdb,
    BeamConfig,
    DecodePhysicsPlus,
    PhysicsPriors,
    compute_physics_priors,
)

__all__ = [
    "build_engine",
    "build_engine_future",
    "predict_pause_structures",
    "export_pause_pdb",
    "predict_future_movie",
    "export_future_movie_pdb",
    "BeamConfig",
    "DecodePhysicsPlus",
    "PhysicsPriors",
    "compute_physics_priors",
]
