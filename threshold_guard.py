"""
threshold_guard.py — AlbedoNet App
=====================================
Utilitaire partage pour empecher une modification SILENCIEUSE d'un seuil
metier (materials.py, heat_island_calibration.py, postprocess_plausibility.py)
sans mise a jour explicite du numero de version associe.

Principe : chaque module de seuils declare
  - THRESHOLDS_VERSION   (string libre, incrementee a la main)
  - un tuple de TOUTES ses constantes de seuils
  - THRESHOLDS_HASH      (hash fige de ce tuple, genere une fois)

Un test (cf. tests/test_threshold_versions.py) recalcule le hash courant et
le compare a THRESHOLDS_HASH. Si quelqu'un change un seuil sans regenerer le
hash, le test echoue -- impossible de merger un changement de seuil "en
silence". Ca ne valide PAS que le nouveau seuil est correct (ca reste une
decision humaine), ca garantit juste qu'un changement de seuil est TOUJOURS
visible et delibere (diff sur THRESHOLDS_VERSION + THRESHOLDS_HASH dans la
review), jamais un effet de bord d'une autre modification.

Usage dans un module de seuils :

    from threshold_guard import compute_snapshot_hash

    THRESHOLDS_VERSION = "1.0"
    _THRESHOLDS_SNAPSHOT = (SEUIL_A, SEUIL_B, SEUIL_C)
    THRESHOLDS_HASH = "xxxxxxxxxxxx"  # regenere via regenerate_hash.py

Pour regenerer le hash apres un changement de seuil DELIBERE :
    python regenerate_hash.py materials
    (affiche le nouveau hash a coller dans le module, + rappelle de
    bumper THRESHOLDS_VERSION)
"""

from __future__ import annotations

import hashlib


def compute_snapshot_hash(values: tuple) -> str:
    """Hash court et stable d'un tuple de constantes numeriques/str.

    repr() plutot que str() pour distinguer 0 (int) de 0.0 (float) --
    change de type = change de comportement possible ailleurs, donc doit
    aussi faire echouer le hash.
    """
    payload = repr(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def assert_snapshot_matches(
    module_name: str,
    snapshot: tuple,
    expected_hash: str,
    version: str,
) -> None:
    """A appeler depuis un test (pas depuis le module lui-meme au chargement
    -- on ne veut pas qu'une app en prod plante si le hash est perime, juste
    que la CI/les tests le detectent avant merge).
    """
    actual = compute_snapshot_hash(snapshot)
    if actual != expected_hash:
        raise AssertionError(
            f"[{module_name}] Les constantes de seuils ont change sans mise a "
            f"jour de THRESHOLDS_HASH (version declaree : {version}).\n"
            f"Hash attendu : {expected_hash}\n"
            f"Hash calcule : {actual}\n"
            f"Si ce changement est delibere : incrementer THRESHOLDS_VERSION "
            f"dans {module_name}.py, regenerer le hash "
            f"(python regenerate_hash.py {module_name.replace('.py', '')}) "
            f"et coller le nouveau hash dans le module."
        )
