"""
utils/mae_utils.py — AlbedoNet Pretraining
==========================================
Fonctions partagées :
  - Positional embeddings 2D sin-cos (standard MAE He et al. 2022)
  - patchify / unpatchify
  - BandNormalizer : stats Z-score par bande calculées depuis le HDF5
  - visualize_reconstruction : grille original / masqué / reconstruit
"""

from __future__ import annotations

import random
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


# ──────────────────────────────────────────────────────────────────────────────
# Positional embeddings sin-cos 2D
# ──────────────────────────────────────────────────────────────────────────────

def _1d_sincos(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    half  = embed_dim // 2
    omega = np.arange(half, dtype=np.float64) / half
    omega = 1.0 / (10_000 ** omega)            # (half,)
    pos   = pos.reshape(-1)                     # (M,)
    out   = np.einsum("m,d->md", pos, omega)    # (M, half)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1).astype(np.float32)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    """
    Retourne (grid_size**2, embed_dim) — positional embeddings 2D sin-cos.
    Standard He et al. 2022 MAE.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid   = np.meshgrid(grid_w, grid_h)                       # 2 × (H, W)
    grid   = np.stack(grid, axis=0).reshape(2, 1, grid_size, grid_size)
    emb_h  = _1d_sincos(embed_dim // 2, grid[0])               # (G²,  D/2)
    emb_w  = _1d_sincos(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)              # (G², D)


# ──────────────────────────────────────────────────────────────────────────────
# Patchify / Unpatchify
# ──────────────────────────────────────────────────────────────────────────────

def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    imgs : (B, C, H, W)  — H == W, H % patch_size == 0
    out  : (B, num_patches, patch_size**2 * C)
    """
    B, C, H, W = imgs.shape
    assert H == W and H % patch_size == 0, (
        f"Image {H}×{W} incompatible avec patch_size={patch_size}"
    )
    p = patch_size
    h = H // p
    x = imgs.reshape(B, C, h, p, h, p)
    x = torch.einsum("bchpwq->bhwpqc", x)
    return x.reshape(B, h * h, p * p * C)


def unpatchify(
    patches:  torch.Tensor,
    patch_size: int,
    img_size:   int,
    in_chans:   int,
) -> torch.Tensor:
    """
    patches : (B, num_patches, patch_size**2 * C)
    out     : (B, C, H, W)
    """
    B, N, _ = patches.shape
    p = patch_size
    h = img_size // p
    x = patches.reshape(B, h, h, p, p, in_chans)
    x = torch.einsum("bhwpqc->bchpwq", x)
    return x.reshape(B, in_chans, img_size, img_size)


# ──────────────────────────────────────────────────────────────────────────────
# Normalisation Z-score par bande
# ──────────────────────────────────────────────────────────────────────────────

class BandNormalizer:
    """
    Normalisation Z-score par bande spectrale.
    Calculée une fois depuis un échantillon du HDF5, persistée en .npz.

    Usage :
        normalizer = BandNormalizer.fit_from_h5("pretrain_patches_final.h5")
        normalizer.save("checkpoints/mae/band_stats.npz")
        x_norm = normalizer.normalize(x)   # (B, C, H, W) ou (C, H, W)
    """

    def __init__(self, means: np.ndarray, stds: np.ndarray):
        self.means = torch.tensor(means, dtype=torch.float32)
        self.stds  = torch.tensor(stds,  dtype=torch.float32)

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
    def fit_from_h5(h5_path: str, n_samples: int = 50_000) -> "BandNormalizer":
        """
        Calcule mean/std par bande sur un échantillon aléatoire.
        Fonctionne pour n'importe quel nombre de bandes (3 RGB ou 6).
        """
        with h5py.File(h5_path, "r") as f:
            N    = f["fine"].shape[0]
            idx  = sorted(random.sample(range(N), min(n_samples, N)))
            data = f["fine"][idx].astype(np.float32)   # (n, C, H, W) — float [0,1] (pas uint8)
            # Ne pas diviser par 255 : le HDF5 produit par le patch extractor
            # stocke déjà en float [0,1]. Diviser ici produirait des stats en
            # [0, 255] space → normalisation complètement fausse à l'inférence.
        means = data.mean(axis=(0, 2, 3))   # (C,)
        stds  = data.std(axis=(0, 2, 3))    # (C,)
        return BandNormalizer(means, stds)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, means=self.means.numpy(), stds=self.stds.numpy())
        print(f"Stats bandes sauvegardées → {path}")

    @staticmethod
    def load(path: str) -> "BandNormalizer":
        d = np.load(path)
        return BandNormalizer(d["means"], d["stds"])

    def __repr__(self) -> str:
        return (
            f"BandNormalizer(\n"
            f"  means={self.means.numpy().round(4)}\n"
            f"  stds ={self.stds.numpy().round(4)}\n)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation reconstruction
# ──────────────────────────────────────────────────────────────────────────────

def visualize_reconstruction(
    original:      torch.Tensor,    # (B, C, H, W)
    reconstructed: torch.Tensor,    # (B, C, H, W)
    mask:          torch.Tensor,    # (B, num_patches)  — 1=masqué
    patch_size:    int,
    normalizer:    BandNormalizer,
    save_path:     str,
    n_examples:    int = 4,
):
    """
    Sauvegarde une grille : original | masqué | reconstruit.
    Prend les 3 premières bandes pour l'affichage RGB.
    """
    B, C, H, W = original.shape
    n    = min(n_examples, B)
    p    = patch_size
    side = H // p

    # Masque → espace pixel
    mask_px = mask[:n].reshape(n, side, side)
    mask_px = mask_px.repeat_interleave(p, dim=1).repeat_interleave(p, dim=2)

    # Dénormaliser → numpy → RGB (3 premières bandes)
    orig_rgb  = normalizer.denormalize(original[:n].cpu())[:, :3].permute(0, 2, 3, 1).numpy()
    recon_rgb = normalizer.denormalize(reconstructed[:n].cpu())[:, :3].permute(0, 2, 3, 1).numpy()
    mask_np   = mask_px.cpu().numpy().astype(bool)

    def _clip01(arr: np.ndarray) -> np.ndarray:
        vmax = np.percentile(arr, 99)
        return np.clip(arr / (vmax + 1e-8), 0, 1)

    orig_rgb  = _clip01(orig_rgb)
    recon_rgb = _clip01(recon_rgb)

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n), dpi=80)
    if n == 1:
        axes = axes[np.newaxis]

    for i in range(n):
        masked_img = orig_rgb[i].copy()
        masked_img[mask_np[i]] = 0.3    # gris pour tokens masqués

        axes[i, 0].imshow(orig_rgb[i]);  axes[i, 0].set_title("Original",  fontsize=9)
        axes[i, 1].imshow(masked_img);   axes[i, 1].set_title(f"Masqué ({int(mask[i].float().mean().item()*100)}%)", fontsize=9)
        axes[i, 2].imshow(recon_rgb[i]); axes[i, 2].set_title("Reconstruit", fontsize=9)
        for ax in axes[i]:
            ax.axis("off")

    plt.suptitle("MAE — reconstruction RGB", fontsize=10)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)