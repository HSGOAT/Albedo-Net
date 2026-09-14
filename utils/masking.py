"""
utils/masking.py — Stub pour l'inférence.
==========================================
En inférence (finetune/production), MAEEncoder est TOUJOURS appelé avec un
mask_override explicite (masque de zéros → tous les tokens sont visibles).
La fonction make_mask() n'est donc jamais exécutée en pratique, mais elle
doit exister pour que `from utils.masking import make_mask` ne lève pas
d'ImportError au chargement de models/mae_encoder.py.

Si un jour ce projet a besoin de relancer un pretrain MAE (masquage réel),
il faudra remplacer ce stub par l'implémentation d'origine (random / block /
grid masking) utilisée pendant l'entraînement.
"""

from __future__ import annotations

import torch


def make_mask(
    batch_size: int,
    strategy: str = "grid",
    num_tokens: int = 64,
    mask_ratio: float = 0.75,
    device: str = "cpu",
) -> torch.BoolTensor:
    raise NotImplementedError(
        "make_mask() n'est pas implémenté dans ce projet d'inférence — "
        "il ne devrait jamais être appelé (mask_override est toujours fourni)."
    )
