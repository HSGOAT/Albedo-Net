"""
zone_scan/progress.py — AlbedoNet App
=======================================
Progression "reelle" : mapping etape -> plage de pourcentage global.

Chaque etape du pipeline recoit une plage [lo, hi] sur 0-100. Le callback
recoit (i, total) POUR CETTE ETAPE UNIQUEMENT (ex: tuile 3/12) ; on projette
lineairement i/total sur [lo, hi] pour obtenir un pourcentage global stable,
utilisable tel quel par une barre de progression (SSE, Streamlit, etc.).

Poids approximatifs bases sur le cout relatif observe en pratique :
tuiles + extraction/inference dominent largement le temps total, geocodage
et batiments (WFS) sont quasi-instantanes en comparaison.

Extrait de l'ancien zone_scan.py (Tier 3, decoupage en sous-modules) --
aucun changement de comportement, seulement de localisation.
"""

from __future__ import annotations

_STAGE_WEIGHTS: dict[str, tuple[float, float]] = {
    "geocodage":       (0.0, 2.0),
    "batiments":       (2.0, 8.0),
    "tuiles":          (8.0, 40.0),
    "inference":       (40.0, 88.0),  # phase 1 : extraction patch + materiau, par batiment
    "inference_batch": (88.0, 97.0),  # phase 2 : forward pass ViT par lots
    "stats":           (97.0, 100.0),
}

_STAGE_LABELS: dict[str, str] = {
    "geocodage":       "Géocodage du centre de la zone…",
    "batiments":       "Récupération des bâtiments (BD TOPO)…",
    "tuiles":          "Téléchargement des tuiles orthophoto…",
    "inference":       "Analyse IA des toits (extraction)…",
    "inference_batch": "Analyse IA des toits (inférence)…",
    "stats":           "Calcul des statistiques…",
}


def _progress_percent(stage: str, i: int, total: int) -> float:
    """Projette (i, total) d'UNE etape sur un pourcentage GLOBAL 0-100,
    d'apres _STAGE_WEIGHTS. Etape inconnue -> 0-100 brut (filet de securite,
    ne devrait pas arriver en usage normal)."""
    lo, hi = _STAGE_WEIGHTS.get(stage, (0.0, 100.0))
    if total <= 0:
        return lo
    frac = min(max(i / total, 0.0), 1.0)
    return lo + frac * (hi - lo)
