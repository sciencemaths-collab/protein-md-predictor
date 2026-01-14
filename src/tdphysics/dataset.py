from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

@dataclass
class LevelSpec:
    name: str
    lag_steps: int
    weight: float = 1.0

class MultiLevelTokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, levels: List[LevelSpec], context: int = 64):
        self.tokens = tokens.astype(np.int64)
        self.levels = levels
        self.context = int(context)
        self._index: List[Tuple[int, int]] = []
        T = len(tokens)
        for li, lv in enumerate(levels):
            g = int(lv.lag_steps)
            for t_end in range(context - 1, T - g):
                self._index.append((t_end, li))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        t_end, li = self._index[idx]
        g = int(self.levels[li].lag_steps)
        x = self.tokens[t_end - self.context + 1 : t_end + 1]
        y = self.tokens[t_end + g]
        return torch.from_numpy(x), torch.tensor(li, dtype=torch.long), torch.tensor(y, dtype=torch.long)
