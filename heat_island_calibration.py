"""
heat_island_calibration.py — AlbedoNet App
=============================================
Seuils de noirceur/clarte absolue calibres PAR VILLE, regenere
automatiquement par run_all_cities.py.

METHODE (10/07/2026, mise a jour) : p15 (seuil sombre) et p85 (seuil clair)
REELS, calcules sur la distribution complete d'albedo d'un scan de zone par
ville (pas une estimation a la main comme dans la version precedente).
Toujours calibre sur UN SEUL scan par ville pour l'instant -- a stabiliser
des que plusieurs scans par ville seront disponibles.

Pour une ville non presente dans CITY_DARK_THRESHOLDS/CITY_BRIGHT_THRESHOLDS,
on utilise DEFAULT_DARK_THRESHOLD / DEFAULT_BRIGHT_THRESHOLD (volontairement
pessimistes -- bas pour le seuil sombre, haut pour le seuil clair -- pour
eviter de sur-detecter des faux positifs sur une ville non calibree, ex.
provence_rurale qui n'est pas une commune et n'est jamais matchee par
geocodage direct).

CORRECTIF 07/09/2026 : les seuils par ville deviennent des VALEURS PAR
DEFAUT personnalisables plutot qu'une verite figee. La calibration reste
utile comme point de depart (elle evite a l'utilisateur de partir d'un
slider a zero), mais get_dark_threshold()/get_bright_threshold() acceptent
desormais un user_override explicite qui prend le pas dessus -- typiquement
la valeur choisie via un slider cote app. get_default_thresholds() renvoie
le couple a pre-remplir dans ce slider, et validate_user_thresholds() fait
un controle de coherence minimal (bornes 0-1, sombre < clair) sans juger
si le choix de l'utilisateur est "bon".
"""

from __future__ import annotations

CITY_DARK_THRESHOLDS: dict[str, float] = {
    "clermont":        0.218,
    "grenoble":        0.216,
    "lille":           0.155,
    "lyon":            0.218,
    "marseille":       0.289,
    "metz":            0.299,
    "montpellier":     0.279,
    "nantes":          0.177,
    "nice":            0.3,
    "paris":           0.194,
    "strasbourg":      0.218,
    "toulouse":        0.231,
    "tours":           0.228,
}

DEFAULT_DARK_THRESHOLD: float = 0.18


# ──────────────────────────────────────────────────────────────────────────
# Correctif 07/09/2026 : shrinkage vers la mediane inter-villes.
# ──────────────────────────────────────────────────────────────────────────
# Constat : Lille (seule ville "du nord" calibree a ce jour) a des seuils
# sombre/clair ~30-36% sous la mediane des autres villes calibrees, dans le
# meme sens et la meme ampleur pour les deux seuils simultanement -- signe
# probable d'un scan de calibration non representatif (10/07/2026, un seul
# scan pour Lille, cf. note en tete de fichier) plutot que d'une vraie
# difference physique d'albedo des toits lillois.
#
# Faute de pouvoir relancer run_all_cities.py avec plusieurs scans (pas de
# donnees supplementaires disponibles), on applique un shrinkage statistique
# leger : tout seuil calibre qui s'ecarte de plus de OUTLIER_DEVIATION de la
# mediane des AUTRES villes est rapproche de cette mediane de moitie
# (SHRINKAGE_FACTOR). Mecanisme generique : s'applique automatiquement a
# toute nouvelle ville ajoutee (nord ou ailleurs) tant qu'elle reste calibree
# sur un seul scan et s'ecarte fortement du reste -- pas seulement a Lille.
#
# A retirer/ajuster des que plusieurs scans par ville seront disponibles et
# qu'une vraie moyenne/mediane par ville pourra remplacer ce garde-fou.
OUTLIER_DEVIATION: float = 0.25   # ecart relatif a la mediane a partir duquel on corrige
SHRINKAGE_FACTOR: float = 0.25    # 0 = pas de correction, 1 = ramene pile sur la mediane
# NB (07/09/2026) : 0.5 corrigeait trop fort -- le seuil clair de Lille
# remontait a ~0.28, ce qui faisait basculer ~1/4 des batiments du scan
# Merignies en "chaud" alors qu'avant (aucun shrinkage) tout ressortait
# "froid". 0.25 est un compromis : seuil clair Lille ~0.249 (dark ~0.172)
# au lieu de 0.280 (dark 0.189) avec 0.5, ou 0.219 (dark 0.155) sans
# shrinkage. A rejuger des que plusieurs scans par ville seront dispo.


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _shrink_outliers(thresholds: dict[str, float]) -> dict[str, float]:
    """Rapproche de la mediane inter-villes tout seuil qui s'en ecarte de
    plus de OUTLIER_DEVIATION, cf. note ci-dessus. La mediane de reference
    pour une ville donnee exclut cette ville elle-meme, pour ne pas laisser
    un outlier se "valider" avec sa propre valeur."""
    corrected: dict[str, float] = {}
    for city, value in thresholds.items():
        others = [v for k, v in thresholds.items() if k != city]
        if not others:
            corrected[city] = value
            continue
        ref_median = _median(others)
        deviation = (ref_median - value) / ref_median if ref_median else 0.0
        if abs(deviation) > OUTLIER_DEVIATION:
            corrected[city] = value + SHRINKAGE_FACTOR * (ref_median - value)
        else:
            corrected[city] = value
    return corrected


# Valeurs brutes issues de run_all_cities.py, conservees telles quelles pour
# audit/tracabilite. Ne pas utiliser directement pour la detection -- passer
# par get_dark_threshold()/get_bright_threshold(), qui appliquent le
# shrinkage ci-dessus.
CITY_DARK_THRESHOLDS_RAW: dict[str, float] = dict(CITY_DARK_THRESHOLDS)
CITY_DARK_THRESHOLDS = _shrink_outliers(CITY_DARK_THRESHOLDS)


def get_dark_threshold(city_key: str | None, user_override: float | None = None) -> float:
    """Seuil sombre a utiliser pour la detection.

    Si l'utilisateur a choisi sa propre valeur (slider cote app), elle est
    prioritaire sur la calibration par ville -- cf. get_default_thresholds()
    pour obtenir la valeur suggeree a pre-remplir dans ce slider.
    """
    if user_override is not None:
        return user_override
    if not city_key:
        return DEFAULT_DARK_THRESHOLD
    return CITY_DARK_THRESHOLDS.get(city_key, DEFAULT_DARK_THRESHOLD)


CITY_BRIGHT_THRESHOLDS: dict[str, float] = {
    "clermont":        0.341,
    "grenoble":        0.351,
    "lille":           0.219,
    "lyon":            0.314,
    "marseille":       0.427,
    "metz":            0.383,
    "montpellier":     0.392,
    "nantes":          0.298,
    "nice":            0.37,
    "paris":           0.296,
    "strasbourg":      0.34,
    "toulouse":        0.329,
    "tours":           0.32,
}

DEFAULT_BRIGHT_THRESHOLD: float = 0.45

# Meme shrinkage que pour les seuils sombres (cf. note detaillee plus haut).
CITY_BRIGHT_THRESHOLDS_RAW: dict[str, float] = dict(CITY_BRIGHT_THRESHOLDS)
CITY_BRIGHT_THRESHOLDS = _shrink_outliers(CITY_BRIGHT_THRESHOLDS)


def get_bright_threshold(city_key: str | None, user_override: float | None = None) -> float:
    """Seuil clair a utiliser pour la detection. Meme logique d'override
    que get_dark_threshold() ci-dessus."""
    if user_override is not None:
        return user_override
    if not city_key:
        return DEFAULT_BRIGHT_THRESHOLD
    return CITY_BRIGHT_THRESHOLDS.get(city_key, DEFAULT_BRIGHT_THRESHOLD)


def get_default_thresholds(city_key: str | None) -> tuple[float, float]:
    """(seuil_sombre, seuil_clair) suggeres pour pre-remplir les sliders
    cote app -- la calibration par ville sert desormais de point de depart
    personnalisable, plus de valeur figee imposee a l'utilisateur."""
    return get_dark_threshold(city_key), get_bright_threshold(city_key)


def validate_user_thresholds(dark: float, bright: float) -> str | None:
    """Garde-fou minimal cote app quand l'utilisateur choisit ses propres
    seuils : renvoie un message d'erreur si la config n'a pas de sens,
    sinon None. Ne juge pas si les valeurs sont "bonnes", seulement si
    elles sont coherentes (borne 0-1 d'albedo, sombre < clair)."""
    if not (0.0 <= dark <= 1.0) or not (0.0 <= bright <= 1.0):
        return "Les seuils doivent etre des albedos entre 0 et 1."
    if dark >= bright:
        return "Le seuil sombre doit etre strictement inferieur au seuil clair."
    return None


# ──────────────────────────────────────────────────────────────────────────
# Verrou de version des seuils (cf. threshold_guard.py) -- NE PAS modifier
# une valeur ci-dessus sans regenerer THRESHOLDS_HASH et incrementer
# THRESHOLDS_VERSION, sinon tests/test_threshold_versions.py echoue.
# ──────────────────────────────────────────────────────────────────────────
THRESHOLDS_VERSION: str = "2.2"
_THRESHOLDS_SNAPSHOT: tuple = (
    tuple(sorted(CITY_DARK_THRESHOLDS.items())),
    DEFAULT_DARK_THRESHOLD,
    tuple(sorted(CITY_BRIGHT_THRESHOLDS.items())),
    DEFAULT_BRIGHT_THRESHOLD,
)
THRESHOLDS_HASH: str = "57e22235b40c"