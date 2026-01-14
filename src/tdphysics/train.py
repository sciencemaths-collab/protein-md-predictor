from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dataset import MultiLevelTokenDataset, LevelSpec
from .model import TinyTransformer

ProgressCB = Callable[[str, dict], None]


def _emit(cb: Optional[ProgressCB], event: str, **payload) -> None:
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:
        return


@dataclass
class TrainConfig:
    context: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1
    max_len: int = 512
    batch_size: int = 128
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-2
    gamma_energy: float = 0.5
    eta_work: float = 0.5
    lam_smooth: float = 0.2
    device: str = "cpu"


@dataclass
class TrainReport:
    history: List[Dict[str, float]]
    vocab_size: int
    n_levels: int


def train_multilevel(
    tokens: np.ndarray,
    levels: List[LevelSpec],
    token_energy: np.ndarray,
    codebook_z: np.ndarray,
    cfg: TrainConfig,
    seed: int = 0,
    progress_cb: Optional[ProgressCB] = None,
) -> Tuple[TinyTransformer, TrainReport]:
    """Train the multi-level transformer with physics-shaped losses.

    progress_cb events:
      - train.epoch (p, epoch, epochs, loss_ce, exp_energy, work, smooth)
    """

    torch.manual_seed(int(seed))
    device = torch.device(cfg.device)

    tokens = np.asarray(tokens, dtype=np.int64)
    V = int(max(tokens.max() + 1, len(token_energy)))
    n_levels = len(levels)

    ds = MultiLevelTokenDataset(tokens=tokens, levels=levels, context=int(cfg.context))
    dl = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=True)

    model = TinyTransformer(
        vocab_size=V,
        n_levels=n_levels,
        d_model=int(cfg.d_model),
        n_heads=int(cfg.n_heads),
        n_layers=int(cfg.n_layers),
        dropout=float(cfg.dropout),
        max_len=int(cfg.max_len),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))

    E = torch.tensor(token_energy[:V], dtype=torch.float32, device=device)
    C = torch.tensor(codebook_z[:V], dtype=torch.float32, device=device)

    hist: List[Dict[str, float]] = []
    epochs = int(cfg.epochs)
    n_batches = max(1, len(dl))

    _emit(progress_cb, 'train.start', p=0.0, epochs=epochs)

    for ep in range(1, epochs + 1):
        model.train()
        tot = 0
        tot_ce = tot_e = tot_work = tot_sm = 0.0

        for bi, (x, li, y) in enumerate(dl, start=1):
            x, li, y = x.to(device), li.to(device), y.to(device)
            logits = model(x, li)

            ce = F.cross_entropy(logits, y)
            probs = F.softmax(logits, dim=-1)

            # Expected energy under predicted distribution
            expE = (probs * E.unsqueeze(0)).sum(dim=-1)  # [B]

            # Positive work penalty relative to current token energy
            cur = x[:, -1]
            curE = E[cur]
            work = F.relu(expE - curE).mean()

            # Latent smoothness (expected squared jump in codebook space)
            curC = C[cur]
            diff = C.unsqueeze(0) - curC.unsqueeze(1)
            dist2 = (diff ** 2).sum(dim=-1)
            smooth = (probs * dist2).sum(dim=-1).mean()

            loss = ce + float(cfg.gamma_energy) * expE.mean() + float(cfg.eta_work) * work + float(cfg.lam_smooth) * smooth

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bsz = int(x.shape[0])
            tot += bsz
            tot_ce += float(ce.item()) * bsz
            tot_e += float(expE.mean().item()) * bsz
            tot_work += float(work.item()) * bsz
            tot_sm += float(smooth.item()) * bsz

            # Periodic progress
            if (bi == 1) or (bi == n_batches) or (bi % max(1, n_batches // 10) == 0):
                p = ((ep - 1) + (bi / n_batches)) / max(1, epochs)
                _emit(progress_cb, 'train.batch', p=p, epoch=ep, epochs=epochs, batch=bi, n_batches=n_batches)

        rec = dict(
            epoch=float(ep),
            loss_ce=tot_ce / max(1, tot),
            exp_energy=tot_e / max(1, tot),
            work=tot_work / max(1, tot),
            smooth=tot_sm / max(1, tot),
        )
        hist.append(rec)


        payload = dict(rec)
        payload.pop("epoch", None)

        p = ep / max(1, epochs)

        _emit(progress_cb, "train.epoch", p=p, epoch=ep, epochs=epochs, **payload)
    _emit(progress_cb, 'train.done', p=1.0, epochs=epochs)
    return model, TrainReport(history=hist, vocab_size=V, n_levels=n_levels)
