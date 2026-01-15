from __future__ import annotations
from typing import Optional, Sequence
import numpy as np

def write_ca_pdb(path: str, coords: np.ndarray, resids: np.ndarray, resnames: Sequence[str],
                 bfactor: Optional[np.ndarray] = None, chain_id: str = "A") -> None:
    coords = np.asarray(coords)
    resids = np.asarray(resids)
    if bfactor is None:
        bfactor = np.zeros((coords.shape[0],), dtype=float)
    lines = []
    atom_serial = 1
    for i in range(coords.shape[0]):
        x, y, z = coords[i]
        resn = str(resnames[i])[:3].rjust(3)
        resid = int(resids[i])
        bf = float(bfactor[i])
        lines.append(
            f"ATOM  {atom_serial:5d}  CA  {resn} {chain_id}{resid:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{bf:6.2f}           C"
        )
        atom_serial += 1
    lines.append("END")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
