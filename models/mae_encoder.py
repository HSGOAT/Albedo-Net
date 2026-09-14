"""
models/mae_encoder.py — AlbedoNet Pretraining
==============================================
Encodeur MAE construit autour de ViT-Small.

Rôle :
  1. PatchEmbed + PE sin-cos (délégué à ViTSmall)
  2. Masquage : sélectionne uniquement les tokens NON masqués
  3. Encode les tokens visibles avec les blocs transformer
  4. Retourne (latent, mask, ids_restore) pour le décodeur

Pourquoi masquer AVANT l'encodeur :
  À 75% masked, l'encodeur voit 16 tokens sur 64.
  Compute réduit de ~4× sur les blocs attention (O(N²) sur 16 vs 64).
  C'est l'accélération centrale du MAE vs BEiT/iBOT qui encodent tout.

AUCUN import timm / torchvision.models / transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.vit import ViTSmall
from utils.masking import make_mask


class MAEEncoder(nn.Module):
    """
    Encodeur MAE : masque les tokens avant les blocs transformer.

    Interface forward :
        latent, mask, ids_restore = encoder(x, mask_override=None)

    Args :
        img_size       : taille image carrée (64)
        patch_size     : taille patch (8) → 8×8=64 tokens
        in_chans       : bandes (3 RGB IGN)
        embed_dim      : dimension tokens (384)
        depth          : blocs transformer (12)
        num_heads      : têtes attention (6)
        mlp_ratio      : ratio MLP (4.0)
        drop_path_rate : stochastic depth (0.1)
        mask_ratio     : fraction masquée (0.75)
        mask_strategy  : 'random' | 'block' | 'grid'
    """

    PATCH_SIZE = 8
    IMG_SIZE   = 64

    def __init__(
        self,
        img_size:       int   = 64,
        patch_size:     int   = 8,
        in_chans:       int   = 3,
        embed_dim:      int   = 384,
        depth:          int   = 12,
        num_heads:      int   = 6,
        mlp_ratio:      float = 4.0,
        drop_path_rate: float = 0.1,
        mask_ratio:     float = 0.75,
        mask_strategy:  str   = "grid",
    ):
        super().__init__()

        self.PATCH_SIZE    = patch_size
        self.IMG_SIZE      = img_size
        self.in_chans      = in_chans
        self.embed_dim     = embed_dim
        self.mask_ratio    = mask_ratio
        self.mask_strategy = mask_strategy
        self.num_patches   = (img_size // patch_size) ** 2   # 64

        # ── ViT-Small backbone ────────────────────────────────────────────────
        self.vit = ViTSmall(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
        )

    def forward(
        self,
        x:             torch.Tensor,               # (B, C, H, W)
        mask_override: torch.BoolTensor | None = None,  # (B, num_patches)
    ) -> tuple[torch.Tensor, torch.BoolTensor, torch.Tensor]:
        """
        Encode uniquement les tokens NON masqués.

        Args :
            x             : images normalisées (B, 3, 64, 64)
            mask_override : masque externe (pour val avec masques fixes)
                            Si None, génère un masque selon mask_strategy.

        Returns :
            latent      : (B, len_keep, embed_dim)  tokens encodés visibles
            mask        : (B, num_patches) bool      True=masqué
            ids_restore : (B, num_patches) long      pour restaurer l'ordre
        """
        B = x.shape[0]
        device = x.device

        # ── Génération du masque ──────────────────────────────────────────────
        if mask_override is not None:
            mask = mask_override.to(device)         # (B, num_patches) bool
        else:
            mask = make_mask(
                B,
                strategy=self.mask_strategy,
                num_tokens=self.num_patches,
                mask_ratio=self.mask_ratio,
                device=str(device),
            )                                       # (B, 64) bool

        # ── Patch embed + PE ──────────────────────────────────────────────────
        tokens = self.vit.patch_embed(x)            # (B, 64, 384)
        tokens = tokens + self.vit.pe               # (B, 64, 384)

        # ── Sélection des tokens visibles ─────────────────────────────────────
        # ids_shuffle : indices dans l'ordre aléatoire (visibles en premier)
        # On place les tokens visibles (~mask) en tête
        # Utiliser argsort pour construire ids_restore (inverse de la permutation)
        not_mask = ~mask                            # (B, 64) bool  True=visible
        len_keep = int(not_mask.float().sum(dim=1).min().item())

        # Construire une permutation stable : visibles en premier, masqués ensuite
        # Score 0 pour visible, 1 pour masqué → argsort stable
        scores     = mask.float()                  # (B, 64)  0=visible 1=masqué
        ids_shuffle = torch.argsort(scores, dim=1, stable=True)   # (B, 64)
        ids_restore = torch.argsort(ids_shuffle,  dim=1)          # (B, 64)

        # Extraire uniquement les tokens visibles
        ids_keep = ids_shuffle[:, :len_keep]       # (B, len_keep)
        tokens_visible = torch.gather(
            tokens,
            1,
            ids_keep.unsqueeze(-1).expand(-1, -1, self.embed_dim),
        )                                          # (B, len_keep, 384)

        # ── Blocs transformer sur les tokens visibles uniquement ──────────────
        for blk in self.vit.blocks:
            tokens_visible = blk(tokens_visible)   # (B, len_keep, 384)

        latent = self.vit.norm(tokens_visible)     # (B, len_keep, 384)

        return latent, mask, ids_restore

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        return (
            f"MAEEncoder(\n"
            f"  backbone=ViT-Small ({self.embed_dim}d, {len(self.vit.blocks)} blocs)\n"
            f"  num_patches={self.num_patches}, mask_ratio={self.mask_ratio}\n"
            f"  mask_strategy='{self.mask_strategy}'\n"
            f"  params={self.n_params()/1e6:.1f}M\n)"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Tests rapides
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    enc = MAEEncoder(mask_ratio=0.75, mask_strategy="grid")
    x   = torch.randn(2, 3, 64, 64)

    latent, mask, ids_restore = enc(x)

    assert latent.shape[0] == 2
    assert latent.shape[2] == 384
    assert mask.shape   == (2, 64)
    assert ids_restore.shape == (2, 64)
    assert not latent.isnan().any(), "NaN dans latent !"

    ratio_eff = mask.float().mean().item()
    assert abs(ratio_eff - 0.75) < 0.05, f"mask_ratio effectif = {ratio_eff:.3f}"

    print(enc)
    print(f"\n✅ MAEEncoder — latent={tuple(latent.shape)}, mask={tuple(mask.shape)}")
    print(f"   mask_ratio effectif = {ratio_eff:.3f}")