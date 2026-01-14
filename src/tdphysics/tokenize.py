from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

ProgressCB = Callable[[str, dict], None]


@dataclass
class TokenizationResult:
    """Outputs of tokenization.

    pca/kmeans are sklearn objects; z is latent; tokens are discrete ids; centroids_z are codebook vectors.
    """

    pca: object
    kmeans: object
    z: np.ndarray
    tokens: np.ndarray
    centroids_z: np.ndarray


def _emit(cb: Optional[ProgressCB], event: str, **payload) -> None:
    if cb is None:
        return
    try:
        cb(event, payload)
    except Exception:
        # UI callbacks must never crash core compute
        return


def fit_tokenizer(
    d: np.ndarray,
    m: int = 48,
    K: int = 256,
    seed: int = 0,
    progress_cb: Optional[ProgressCB] = None,
) -> TokenizationResult:
    """Physics-aware tokenization (streamable).

    - Incremental PCA via partial_fit over chunks (progress-friendly)
    - MiniBatchKMeans for scalable codebook learning (progress-friendly)

    progress_cb events:
      - tokenize.pca (p, chunk, n_chunks)
      - tokenize.kmeans (p, chunk, n_chunks)
      - tokenize.done
    """
    from sklearn.decomposition import IncrementalPCA
    from sklearn.cluster import MiniBatchKMeans

    d = np.asarray(d, dtype=np.float32)
    T, D = d.shape
    m = int(max(2, min(m, D)))

    # ---- PCA (partial fit) ----
    bs = int(min(2048, max(64, T // 8)))
    ipca = IncrementalPCA(n_components=m, batch_size=bs)
    n_chunks = int((T + bs - 1) // bs)
    for ci in range(n_chunks):
        a = ci * bs
        b = min(T, (ci + 1) * bs)
        ipca.partial_fit(d[a:b])
        _emit(progress_cb, 'tokenize.pca', p=(ci + 1) / max(1, n_chunks), chunk=ci + 1, n_chunks=n_chunks)

    z = ipca.transform(d).astype(np.float32)

    # ---- KMeans (minibatch) ----
    mb_bs = int(min(4096, max(256, T // 4)))
    km = MiniBatchKMeans(
        n_clusters=int(K),
        random_state=seed,
        batch_size=mb_bs,
        n_init='auto',
        reassignment_ratio=0.01,
        max_no_improvement=20,
    )
    n_chunks2 = int((T + mb_bs - 1) // mb_bs)
    for ci in range(n_chunks2):
        a = ci * mb_bs
        b = min(T, (ci + 1) * mb_bs)
        km.partial_fit(z[a:b])
        _emit(progress_cb, 'tokenize.kmeans', p=(ci + 1) / max(1, n_chunks2), chunk=ci + 1, n_chunks=n_chunks2)

    tokens = km.predict(z).astype(np.int64)
    centroids = km.cluster_centers_.astype(np.float32)

    _emit(progress_cb, 'tokenize.done', p=1.0)
    return TokenizationResult(pca=ipca, kmeans=km, z=z, tokens=tokens, centroids_z=centroids)


def transform_tokens(d: np.ndarray, pca, kmeans) -> Tuple[np.ndarray, np.ndarray]:
    z = pca.transform(d).astype(np.float32)
    tokens = kmeans.predict(z).astype(np.int64)
    return z, tokens
