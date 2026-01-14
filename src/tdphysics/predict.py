from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict, Optional

import numpy as np
import torch

from .dataset import LevelSpec
from .model import energy_biased_logits, sample_from_logits, dist_entropy_from_logits

ProgressCB = Callable[[str, dict], None]


def _emit(cb: Optional[ProgressCB], event: str, **payload) -> None:
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:
        return


@dataclass
class RolloutConfig:
    beta_energy: float = 1.0
    temperature: float = 1.0
    greedy: bool = False


def greedy_level_schedule(horizon_steps: int, levels: List[LevelSpec]) -> List[int]:
    order = sorted(range(len(levels)), key=lambda i: levels[i].lag_steps, reverse=True)
    remaining = int(horizon_steps)
    plan: List[int] = []
    while remaining > 0:
        chosen = None
        for i in order:
            if levels[i].lag_steps <= remaining:
                chosen = i
                break
        if chosen is None:
            chosen = order[-1]
        plan.append(chosen)
        remaining -= int(levels[chosen].lag_steps)
        if len(plan) > 200000:
            raise RuntimeError("Schedule exploded; check lags/horizon.")
    return plan


def rollout_tokens(
    model,
    tokens_seed: np.ndarray,
    levels: List[LevelSpec],
    token_energy: np.ndarray,
    horizon_steps: int,
    context: int,
    cfg: RolloutConfig,
    device: str = "cpu",
    progress_cb: Optional[ProgressCB] = None,
) -> Dict[str, np.ndarray]:
    """Roll out tokens with physics-guided sampling.

    progress_cb events:
      - predict.step (p, step, n_steps, entropy, level_id, token)
      - predict.done
    """
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    V = model.vocab_size
    E = torch.tensor(token_energy[:V], dtype=torch.float32, device=device)

    plan = greedy_level_schedule(int(horizon_steps), levels)
    n_steps = len(plan)

    seq = list(np.asarray(tokens_seed, dtype=np.int64).tolist())
    entropies = []
    levels_used = []

    _emit(progress_cb, 'predict.start', p=0.0, n_steps=n_steps)

    for si, li in enumerate(plan, start=1):
        ctx = seq[-context:]
        if len(ctx) < context:
            ctx = [ctx[0]] * (context - len(ctx)) + ctx

        x = torch.tensor(ctx, dtype=torch.long, device=device).unsqueeze(0)
        level_id = torch.tensor([li], dtype=torch.long, device=device)

        logits = model(x, level_id)
        logits = energy_biased_logits(logits, E, beta=float(cfg.beta_energy))

        if cfg.greedy or float(cfg.temperature) <= 0:
            nxt = int(torch.argmax(logits, dim=-1).item())
        else:
            nxt = int(sample_from_logits(logits, temperature=float(cfg.temperature)).item())

        H = float(dist_entropy_from_logits(logits).item())

        entropies.append(H)
        levels_used.append(li)
        seq.append(nxt)

        if (si == 1) or (si == n_steps) or (si % max(1, n_steps // 120) == 0):
            _emit(
                progress_cb,
                'predict.step',
                p=si / float(max(1, n_steps)),
                step=si,
                n_steps=n_steps,
                entropy=H,
                level_id=int(li),
                token=int(nxt),
            )

    _emit(progress_cb, 'predict.done', p=1.0, n_steps=n_steps)

    return {
        "tokens": np.array(seq, dtype=np.int64),
        "entropies": np.array(entropies, dtype=np.float32),
        "levels_used": np.array(levels_used, dtype=np.int64),
    }
