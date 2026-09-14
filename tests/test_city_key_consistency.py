"""Audit de synchronisation du vocabulaire des cles de ville.

Etend le principe deja en place pour les materiaux (EXPECTED_MATERIAL_KEYS,
cf. shadow_calibration.py) aux cles de ville, dupliquees a travers config.py,
heat_island_calibration.py et confidence.py. Sans ce test, un ajout/retrait
de ville dans config.CITY_MODELS (la source canonique) peut diverger
silencieusement des autres modules -- le bug de fond derriere les
incidents recurrents (seuils ilot de chaleur mal routes, vocabulaire
desynchronise) : un .get(cle, DEFAULT) absorbe l'ecart sans jamais lever
d'erreur, cf. docstring de shadow_calibration.py pour un exemple deja
rencontre sur les materiaux.

Chaque test ci-dessous compare un module a city_registry.py (la source
unique de verite) plutot qu'a une copie figee dans ce fichier de test --
si city_registry.py est mis a jour avec une nouvelle exception documentee,
ces tests le refletent automatiquement.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from city_registry import (  # noqa: E402
    CANONICAL_CITY_KEYS,
    CITIES_WITHOUT_APPROX_CENTER,
    CITIES_WITHOUT_HEAT_ISLAND_CALIBRATION,
    EXTRA_SCRAPING_ONLY_CITY_KEYS,
)
from config import COMMUNE_TO_CITY_KEY, REGION_TO_CITY_KEYS  # noqa: E402
from confidence import CITY_CENTERS_APPROX  # noqa: E402
from heat_island_calibration import CITY_BRIGHT_THRESHOLDS_RAW, CITY_DARK_THRESHOLDS_RAW  # noqa: E402


def test_heat_island_dark_and_bright_cover_same_cities():
    """Les seuils sombre et clair doivent porter EXACTEMENT sur les memes
    villes -- une ville avec un seuil sombre mais pas de seuil clair (ou
    l'inverse) est un signe de desynchronisation lors d'un ajout/retrait."""
    dark_keys = set(CITY_DARK_THRESHOLDS_RAW)
    bright_keys = set(CITY_BRIGHT_THRESHOLDS_RAW)
    assert dark_keys == bright_keys, (
        f"CITY_DARK_THRESHOLDS et CITY_BRIGHT_THRESHOLDS divergent : "
        f"seulement dans dark={sorted(dark_keys - bright_keys)}, "
        f"seulement dans bright={sorted(bright_keys - dark_keys)}"
    )


def test_heat_island_thresholds_match_canonical_cities():
    """CITY_DARK/BRIGHT_THRESHOLDS doit couvrir exactement CANONICAL_CITY_KEYS,
    a l'exception documentee de CITIES_WITHOUT_HEAT_ISLAND_CALIBRATION."""
    dark_keys = set(CITY_DARK_THRESHOLDS_RAW)
    expected = CANONICAL_CITY_KEYS - CITIES_WITHOUT_HEAT_ISLAND_CALIBRATION

    missing = expected - dark_keys
    assert not missing, (
        f"Ville(s) dans config.CITY_MODELS mais absente(s) de "
        f"heat_island_calibration (et non exemptee(s)) : {sorted(missing)} -- "
        "ajouter une calibration OU documenter l'exception dans "
        "city_registry.CITIES_WITHOUT_HEAT_ISLAND_CALIBRATION."
    )

    unexpected = dark_keys - CANONICAL_CITY_KEYS
    assert not unexpected, (
        f"Ville(s) calibree(s) dans heat_island_calibration mais absente(s) "
        f"de config.CITY_MODELS : {sorted(unexpected)} -- reference "
        "pendante (ville retiree de CITY_MODELS sans nettoyer les seuils, "
        "cf. cas 'beauce' documente dans config.py)."
    )


def test_commune_mapping_targets_valid_cities():
    """Toute valeur de COMMUNE_TO_CITY_KEY doit pointer vers une ville
    reellement modelisee -- sinon le geocodage matche une commune vers une
    cle inexistante, qui retombe silencieusement sur GENERIC_FALLBACK_MODEL
    (config.get_checkpoint_path) sans jamais signaler l'incoherence."""
    targets = set(COMMUNE_TO_CITY_KEY.values())
    dangling = targets - CANONICAL_CITY_KEYS
    assert not dangling, (
        f"COMMUNE_TO_CITY_KEY pointe vers des cles absentes de "
        f"config.CITY_MODELS : {sorted(dangling)}"
    )


def test_region_fallback_targets_valid_cities():
    """Idem pour REGION_TO_CITY_KEYS : toute cle de secours proposee doit
    correspondre a un modele reellement disponible."""
    all_targets: set[str] = set()
    for region, city_keys in REGION_TO_CITY_KEYS.items():
        all_targets.update(city_keys)
    dangling = all_targets - CANONICAL_CITY_KEYS
    assert not dangling, (
        f"REGION_TO_CITY_KEYS pointe vers des cles absentes de "
        f"config.CITY_MODELS : {sorted(dangling)}"
    )


def test_confidence_city_centers_cover_canonical_cities():
    """CITY_CENTERS_APPROX doit au moins couvrir toutes les villes
    modelisees (a l'exception documentee de provence_rurale) -- un modele
    CITY_MODELS sans centre approximatif perd silencieusement le signal de
    distance dans compute_confidence() (aucune erreur, juste un signal en
    moins, cf. docstring confidence.py)."""
    center_keys = set(CITY_CENTERS_APPROX)
    expected = CANONICAL_CITY_KEYS - CITIES_WITHOUT_APPROX_CENTER
    missing = expected - center_keys
    assert not missing, (
        f"Ville(s) dans config.CITY_MODELS sans centre approximatif dans "
        f"confidence.CITY_CENTERS_APPROX (et non exemptee(s)) : "
        f"{sorted(missing)} -- ajouter des coordonnees OU documenter "
        "l'exception dans city_registry.CITIES_WITHOUT_APPROX_CENTER."
    )


def test_confidence_extra_keys_are_all_declared():
    """Toute cle de CITY_CENTERS_APPROX qui n'est PAS un CITY_MODELS doit
    etre explicitement declaree dans EXTRA_SCRAPING_ONLY_CITY_KEYS -- sinon
    c'est soit une faute de frappe, soit une ville ajoutee sans mise a jour
    du registre (le meme oubli qui a cause les bugs recurrents de
    vocabulaire)."""
    center_keys = set(CITY_CENTERS_APPROX)
    extra = center_keys - CANONICAL_CITY_KEYS
    undeclared = extra - EXTRA_SCRAPING_ONLY_CITY_KEYS
    assert not undeclared, (
        f"Cle(s) dans confidence.CITY_CENTERS_APPROX ni dans "
        f"config.CITY_MODELS ni declaree(s) dans "
        f"city_registry.EXTRA_SCRAPING_ONLY_CITY_KEYS : {sorted(undeclared)} -- "
        "faute de frappe probable, ou nouvelle ville a declarer explicitement."
    )
