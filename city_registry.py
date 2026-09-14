"""
city_registry.py — AlbedoNet App
===================================
Source UNIQUE de verite pour le vocabulaire des cles de ville, sur le meme
principe que EXPECTED_MATERIAL_KEYS (shadow_calibration.py) pour les
materiaux.

Contexte : les cles de ville sont dupliquees a travers 4 modules distincts
(config.py, heat_island_calibration.py, confidence.py), et rien n'empeche
aujourd'hui l'un de diverger silencieusement des autres. Deux mecanismes
existants ne couvrent PAS ce risque :
  - EXPECTED_MATERIAL_KEYS (shadow_calibration.py) : couvre le vocabulaire
    MATERIAUX, pas les villes.
  - threshold_guard.py (hash de version) : detecte qu'UN module a change,
    mais ne compare jamais deux modules ENTRE EUX -- un hash different sur
    chacun des 4 fichiers ci-dessus serait juge "normal" meme si leurs
    ensembles de cles ville ont diverge de facon incoherente.

Ce module ne redefinit AUCUNE cle : il importe CITY_MODELS depuis config.py
(deja la source canonique de facto -- c'est la liste des checkpoints .pt
reellement entraines) et expose des exceptions EXPLICITES, documentees,
pour les ecarts volontaires entre modules. tests/test_city_key_consistency.py
s'appuie sur ce module pour verifier que tout ecart NON documente ici casse
la CI.

Si un ecart est intentionnel, il doit etre ajoute a l'un des ensembles
d'exception ci-dessous -- AVEC une justification en commentaire -- plutot
que de laisser le test echouer ou, pire, de l'assouplir en douce.
"""
from __future__ import annotations

from config import CITY_MODELS

# Source canonique : toute ville ayant un modele entraine (checkpoint .pt).
# Ne PAS redupliquer cette liste ailleurs -- importer CANONICAL_CITY_KEYS.
CANONICAL_CITY_KEYS: frozenset[str] = frozenset(CITY_MODELS)

# ──────────────────────────────────────────────────────────────────────────
# Exceptions documentees -- tout ecart au principe "les 4 modules doivent
# couvrir exactement CANONICAL_CITY_KEYS" doit etre declare ici.
# ──────────────────────────────────────────────────────────────────────────

# "provence_rurale" n'est pas une commune : pas de scan de calibration
# ilot de chaleur dedie (cf. heat_island_calibration.py, DEFAULT_*_THRESHOLD
# volontairement pessimistes servent de repli). A retirer de cette
# exception le jour ou une vraie calibration existe pour ce modele.
CITIES_WITHOUT_HEAT_ISLAND_CALIBRATION: frozenset[str] = frozenset({"provence_rurale"})

# Cles techniques utilisees UNIQUEMENT pour le scraping OSM / le signal de
# distance au centre-ville (confidence.py), sans modele CITY_MODELS associe.
# Ajoutees le 23/07/2026 pour combler le deficit d'echantillons zinc --
# statut a confirmer (cf. commentaire confidence.py::CITY_CENTERS_APPROX) :
# certaines pourraient devenir des CITY_MODELS a part entiere plus tard.
EXTRA_SCRAPING_ONLY_CITY_KEYS: frozenset[str] = frozenset({
    "rouen", "reims", "dijon", "bordeaux", "rennes", "amiens", "caen",
    "toulon", "aix_en_provence", "bayonne", "pau", "brest",
})

# "provence_rurale" n'a pas de centre-ville ponctuel pertinent (ce n'est pas
# une commune) -- absence volontaire de confidence.CITY_CENTERS_APPROX,
# le signal de distance est simplement ignore pour ce modele (cf. docstring
# confidence.py).
CITIES_WITHOUT_APPROX_CENTER: frozenset[str] = frozenset({"provence_rurale"})
