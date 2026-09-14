"""
versioning.py — AlbedoNet App
================================
Metadonnees de version pour chaque run (mode adresse unique / lot / zone),
pour repondre a "quel modele, quelle version des seuils a produit ce
chiffre ?" (cf. suivi projet, "Reproductibilite/versionnement des runs").

Volontairement simple : pas de dependance a git (l'environnement
d'execution -- poste de Pablo, serveur Streamlit -- n'a pas necessairement
un depot git initialise/accessible a l'endroit ou tourne l'app). A la
place :

  - hash SHA256 (tronque) du fichier checkpoint utilise -- identifie de
    facon fiable QUEL fichier .pt a servi, y compris si deux checkpoints
    partagent le meme nom de dossier mais ont ete remplaces entre deux runs
    (un nom de fichier seul ne garantit rien si le contenu a change).
  - date/heure du run (UTC).
  - un numero de version des seuils/logique metier (THRESHOLDS_VERSION),
    a incrementer MANUELLEMENT (cf. commentaire en tete de la constante)
    a chaque changement de seuil. Pas de detection automatique de diff de
    code ici -- necessite de la discipline manuelle a chaque modification.
    A defaut de mieux pour l'instant : mieux qu'aucune tracabilite.

Utilisation : appeler build_run_metadata() une fois par run (adresse/lot/
zone) et inclure le resultat dans tout export (CSV/GeoJSON/rapport).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────
# A INCREMENTER MANUELLEMENT a chaque changement des seuils/heuristiques
# suivants (liste indicative, pas figee, mais c'est le contrat implicite de
# ce numero -- si tu changes un de ces fichiers, mets a jour la version ET
# ajoute une ligne a THRESHOLDS_VERSION_HISTORY) :
#   - materials.py (RED_DOMINANCE_THRESHOLD, LOW_SATURATION_THRESHOLD,
#     BETON_BRIGHTNESS_THRESHOLD, ZINC_BRIGHTNESS_THRESHOLD)
#   - heat_island_calibration.py (CITY_DARK_THRESHOLDS, CITY_BRIGHT_THRESHOLDS,
#     DEFAULT_DARK_THRESHOLD, DEFAULT_BRIGHT_THRESHOLD)
#   - postprocess_plausibility.py (LITERATURE_RANGES, SUSPECT_MARGIN_RATIO,
#     DEFAULT_ANCHOR_ALPHA)
#   - confidence.py (PENALTY_*, SCORE_THRESHOLDS, DISTANCE_WARNING_KM)
# ──────────────────────────────────────────────────────────────────────────
THRESHOLDS_VERSION: str = "2026-07-10.1"

THRESHOLDS_VERSION_HISTORY: list[tuple[str, str]] = [
    (
        "2026-07-10.1",
        "Version initiale de ce fichier -- reprend l'etat des seuils au "
        "10/07/2026 (seuils ilots calibres sur 1 scan/ville, "
        "DEFAULT_ANCHOR_ALPHA=0.0, ajout de confidence.py avec ses propres "
        "penalites).",
    ),
]

# Chemin du checkpoint -> hash calcule une seule fois par run (cache simple,
# le fichier ne change pas en cours d'execution).
_checkpoint_hash_cache: dict[str, str] = {}


def checkpoint_short_hash(checkpoint_path: str) -> Optional[str]:
    """SHA256 tronque (12 caracteres) du fichier checkpoint, ou None si le
    fichier est introuvable. Ne leve jamais d'exception -- purement
    informatif, ne doit jamais faire echouer une prediction."""
    if checkpoint_path in _checkpoint_hash_cache:
        return _checkpoint_hash_cache[checkpoint_path]
    try:
        path = Path(checkpoint_path)
        if not path.exists():
            return None
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        short = h.hexdigest()[:12]
        _checkpoint_hash_cache[checkpoint_path] = short
        return short
    except Exception:
        return None


@dataclass
class RunMetadata:
    run_timestamp_utc: str
    thresholds_version: str
    checkpoint_path: str
    checkpoint_hash: Optional[str]
    city_key_used: str
    mode: str  # "adresse_unique" / "lot" / "zone"


def build_run_metadata(checkpoint_path: str, city_key_used: str, mode: str) -> RunMetadata:
    return RunMetadata(
        run_timestamp_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        thresholds_version=THRESHOLDS_VERSION,
        checkpoint_path=checkpoint_path,
        checkpoint_hash=checkpoint_short_hash(checkpoint_path),
        city_key_used=city_key_used,
        mode=mode,
    )


def format_run_metadata_footer(meta: RunMetadata) -> str:
    """Ligne de synthese lisible, a afficher/exporter en pied de page ou de
    rapport (cf. app.py, generate_report.py)."""
    chash = meta.checkpoint_hash or "hash indisponible"
    return (
        f"Run {meta.run_timestamp_utc} | mode={meta.mode} | "
        f"modele={meta.city_key_used} (checkpoint {chash}) | "
        f"seuils v{meta.thresholds_version}"
    )
