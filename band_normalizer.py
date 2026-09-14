"""
utils/band_normalizer.py — AlbedoNet App
==========================================
Version allegee de BandNormalizer (cf. utils/mae_utils.py, version complete
utilisee pour l'entrainement). Cette version ne depend PAS de h5py ni de
matplotlib : ces librairies ne servent qu'a fit_from_h5() et
visualize_reconstruction(), jamais a l'inference. On evite donc de les
tirer comme dependances dans l'app Streamlit.

Ne PAS utiliser utils/mae_utils.py depuis l'app (cf. notes techniques du
suivi projet) -- inference.py importe exclusivement ce module-ci.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class BandNormalizer:
    """Normalisation Z-score par bande spectrale.

    Usage (inference uniquement) :
        normalizer = BandNormalizer(means=band_stats["mean"], stds=band_stats["std"])
        x_norm = normalizer.normalize(x)   # (B, C, H, W) ou (C, H, W)
    """

    def __init__(self, means: np.ndarray | list[float], stds: np.ndarray | list[float]):
        self.means = torch.tensor(np.asarray(means, dtype=np.float32))
        self.stds = torch.tensor(np.asarray(stds, dtype=np.float32))

    def _broadcast(self, x: torch.Tensor):
        m = self.means.to(x.device)
        s = self.stds.to(x.device)
        if x.ndim == 4:
            return m[None, :, None, None], s[None, :, None, None]
        return m[:, None, None], s[:, None, None]

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._broadcast(x)
        return (x - m) / (s + 1e-8)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        m, s = self._broadcast(x)
        return x * (s + 1e-8) + m

    @staticmethod
    def load(path: str) -> "BandNormalizer":
        d = np.load(path)
        return BandNormalizer(d["means"], d["stds"])

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, means=self.means.numpy(), stds=self.stds.numpy())

    def __repr__(self) -> str:
        return (
            f"BandNormalizer(means={self.means.numpy().round(4)}, "
            f"stds={self.stds.numpy().round(4)})"
        )
