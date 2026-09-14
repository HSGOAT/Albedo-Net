"""Audit de synchronisation confidence.py <-> materials.py.

Remplace l'ancien garde-fou de confidence.py (_check_materiau_proba_threshold_sync,
un simple logger.warning au chargement du module) par un test BLOQUANT en CI.

Pourquoi le warning seul ne suffisait pas : il ne s'affiche que si quelqu'un
charge confidence.py localement et regarde ses logs -- rien n'empechait de
merger une desynchronisation sans jamais la voir. Meme classe de bug que
celle deja traitee pour EXPECTED_MATERIAL_KEYS (shadow_calibration.py) et
pour les cles de ville (city_registry.py) : deux constantes qui doivent
rester egales, dans deux modules distincts, sans import croise dur entre
eux (choix delibere, cf. docstring confidence.py, pour ne pas coupler le
chargement des deux modules) -- donc rien ne force leur synchronisation
sauf un test explicite.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from confidence import MATERIAU_PROBA_WARNING  # noqa: E402
from materials import MIN_RELIABLE_PROBA  # noqa: E402


def test_materiau_proba_threshold_in_sync():
    """confidence.MATERIAU_PROBA_WARNING et materials.MIN_RELIABLE_PROBA
    doivent rester identiques -- cf. docstring confidence.py : un changement
    de l'un doit s'accompagner d'une revue de l'autre. Avant ce test, seul
    un logger.warning signalait l'ecart, sans jamais bloquer la CI."""
    assert MATERIAU_PROBA_WARNING == MIN_RELIABLE_PROBA, (
        f"Desynchronisation : confidence.MATERIAU_PROBA_WARNING="
        f"{MATERIAU_PROBA_WARNING} != materials.MIN_RELIABLE_PROBA="
        f"{MIN_RELIABLE_PROBA}. Mettre a jour les deux constantes ensemble "
        "(pas d'import croise dur entre les deux modules par choix de "
        "design, cf. docstring confidence.py -- la synchronisation est "
        "donc manuelle et doit etre revue a chaque changement de l'une "
        "des deux valeurs)."
    )
