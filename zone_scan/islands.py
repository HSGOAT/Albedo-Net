"""
zone_scan/islands.py — AlbedoNet App
======================================
Detection ilots de chaleur/fraicheur : seuil absolu d'albedo + clustering
spatial. Un vrai ilot de chaleur est un effet de CONCENTRATION spatiale de
toits sombres, pas un bâtiment isole. On combine donc deux criteres, les
DEUX doivent etre vrais pour flaguer un batiment :

  1. Seuil absolu d'albedo : le toit est reellement sombre en valeur
     absolue (independamment de la moyenne de la zone). Un albedo sous ~0.15
     correspond typiquement a du zinc/ardoise fonces, du bitume, etc.
  2. Clustering : le batiment a au moins MIN_DARK_NEIGHBORS autres toits
     sombres (meme seuil) dans un rayon DARK_CLUSTER_RADIUS_M.

Ce choix (ET, pas OU) evite les deux biais d'une approche purement
relative (z-score) :
  - flaguer un toit juste "un peu moins clair que ses voisins" dans une
    zone deja globalement claire (faux positif)
  - rater un vrai cluster de toits sombres dans une zone globalement
    sombre, ou le z-score ne verrait aucun ecart (faux negatif)

Contient aussi compute_zone_stats (agregats sur les albedos valides de la
zone), puisqu'elle partage les memes structures (BuildingResult, ZoneStats)
et s'execute a la meme etape du pipeline ("stats").

Extrait de l'ancien zone_scan.py (Tier 3, decoupage en sous-modules) --
aucun changement de comportement, seulement de localisation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from heat_island_calibration import get_dark_threshold, get_bright_threshold

from .models import BuildingResult, ZoneStats

# Seuil absolu d'albedo en dessous duquel un toit est considere "sombre".
#
# NOTE 09/07/2026 : ancienne valeur fixe (0.15) abandonnee -- des scans reels
# sur Lyon/Paris/Marseille ont montre que la distribution d'albedo varie
# fortement d'une ville a l'autre (min observe 0.177 a Paris vs 0.245 a
# Marseille), rendant un seuil global soit inutile soit trop permissif selon
# la ville. Le seuil est desormais resolu PAR VILLE via
# heat_island_calibration.get_dark_threshold() -- cf. ce module pour le
# detail de la calibration et ses limites (calibree sur un seul scan par
# ville pour l'instant, pas une verite terrain physique).
#
# Cette constante reste ici comme valeur de repli explicite si jamais
# flag_heat_island_suspects est appelee sans dark_threshold ET sans city_key
# (ne devrait pas arriver en usage normal, scan_zone passe toujours city_key).
DARK_ROOF_ALBEDO_THRESHOLD: float = 0.18  # = DEFAULT_DARK_THRESHOLD, garde en phase manuellement

# Rayon (m) dans lequel on cherche des voisins sombres autour d'un batiment.
DARK_CLUSTER_RADIUS_M: float = 50.0

# Nombre minimum de VOISINS sombres (le batiment lui-meme non compte) dans
# ce rayon pour que le cluster soit considere significatif.
MIN_DARK_NEIGHBORS: int = 2

# Fraction d'ombre portee (cf. BuildingResult.shadow_fraction) au-dela de
# laquelle un toit sombre est EXCLU des suspects "ilot de chaleur". Un toit
# assombri par une ombre portee (arbre, batiment voisin, etc.) au moment de
# la prise de vue n'est pas un vrai toit chaud -- c'est un faux positif
# d'ombre, pas de materiau.
SHADOW_FRACTION_MAX_HEAT: float = 0.3

# Materiau(x) exclu(s) des suspects "ilot de fraicheur" : un toit en tuiles
# peut ressortir clair en albedo sans etre un vrai toit reflechissant/froid
# au sens ou on l'entend ici (faux positif materiau).
# NOTE : la valeur exacte retournee par materials.classify_material est
# "tuile_terre_cuite" (cf. materials.MATERIALS), pas "tuile".
COOL_SUSPECT_EXCLUDED_MATERIALS: frozenset[str] = frozenset({"tuile_terre_cuite"})

# Rayon et nombre de voisins reutilises pour la symetrie clair/fonce -- meme
# logique de clustering, juste inversee (toits CLAIRS regroupes).
BRIGHT_CLUSTER_RADIUS_M: float = DARK_CLUSTER_RADIUS_M
MIN_BRIGHT_NEIGHBORS: int = MIN_DARK_NEIGHBORS


def flag_heat_island_suspects(
    buildings: list["BuildingResult"],
    city_key: Optional[str] = None,
    dark_threshold: Optional[float] = None,
    cluster_radius_m: float = DARK_CLUSTER_RADIUS_M,
    min_neighbors: int = MIN_DARK_NEIGHBORS,
) -> float:
    """Marque en place (mutation) les batiments faisant partie d'un cluster
    de toits sombres comme heat_island_suspect=True.

    Un batiment est marque uniquement si (1) son propre albedo est sous
    dark_threshold ET (2) il a au moins min_neighbors autres batiments
    egalement sous ce seuil dans un rayon de cluster_radius_m (distance
    euclidienne en coordonnees L93, donc en metres).

    Args:
        city_key: cle ville (cf. config.CITY_MODELS) utilisee pour resoudre
            le seuil calibre via heat_island_calibration.get_dark_threshold(),
            SI dark_threshold n'est pas fourni explicitement.
        dark_threshold: si fourni, prend le pas sur city_key (force un seuil
            precis, utile pour tester manuellement une valeur).

    Returns:
        Le seuil d'albedo effectivement utilise (utile pour l'affichage
        cote frontend -- l'utilisateur doit savoir quel seuil a ete applique).

    Complexite O(n^2) sur les batiments sombres uniquement (pas tous les
    batiments de la zone) -- largement suffisant vu les volumes en jeu
    (quelques centaines a quelques milliers de batiments sombres au pire).
    """
    if dark_threshold is None:
        dark_threshold = get_dark_threshold(city_key)

    ok = [b for b in buildings if b.albedo is not None]
    # Exclut les toits dont l'assombrissement vient d'une ombre portee
    # (cf. SHADOW_FRACTION_MAX_HEAT) plutot que d'un vrai materiau sombre :
    # faux positif frequent (arbre, batiment voisin, route adjacente au
    # patch, etc.), pas un vrai toit chaud.
    dark = [
        b for b in ok
        if b.albedo < dark_threshold
        and (b.shadow_fraction is None or b.shadow_fraction < SHADOW_FRACTION_MAX_HEAT)
    ]

    if len(dark) < min_neighbors + 1:
        return dark_threshold  # pas assez de toits sombres dans la zone pour un cluster

    radius_sq = cluster_radius_m ** 2
    for b in dark:
        n_dark_neighbors = 0
        for other in dark:
            if other is b:
                continue
            dx = other.centroid_x - b.centroid_x
            dy = other.centroid_y - b.centroid_y
            if (dx * dx + dy * dy) <= radius_sq:
                n_dark_neighbors += 1
                if n_dark_neighbors >= min_neighbors:
                    break
        b.heat_island_suspect = n_dark_neighbors >= min_neighbors

    return dark_threshold


def flag_cool_island_suspects(
    buildings: list["BuildingResult"],
    city_key: Optional[str] = None,
    bright_threshold: Optional[float] = None,
    cluster_radius_m: float = BRIGHT_CLUSTER_RADIUS_M,
    min_neighbors: int = MIN_BRIGHT_NEIGHBORS,
) -> float:
    """Symetrique de flag_heat_island_suspects : marque les batiments
    faisant partie d'un cluster de toits CLAIRS comme cool_island_suspect=True.

    Un toit clair reflechit davantage le rayonnement solaire -- un
    regroupement spatial de tels toits contribue moins a l'echauffement
    local que la moyenne de la zone ("ilot de fraicheur" relatif). Meme
    logique ET (seuil absolu + clustering) que pour la chaleur, cf.
    flag_heat_island_suspects pour le detail du raisonnement.

    Returns:
        Le seuil de clarte effectivement utilise (pour affichage cote frontend).
    """
    if bright_threshold is None:
        bright_threshold = get_bright_threshold(city_key)

    ok = [b for b in buildings if b.albedo is not None]
    # Exclut les toits en tuiles (cf. COOL_SUSPECT_EXCLUDED_MATERIALS) :
    # un albedo eleve sur ce materiau est un faux positif frequent, pas un
    # vrai toit reflechissant/froid.
    bright = [
        b for b in ok
        if b.albedo > bright_threshold
        and (b.material not in COOL_SUSPECT_EXCLUDED_MATERIALS)
    ]

    if len(bright) < min_neighbors + 1:
        return bright_threshold  # pas assez de toits clairs pour un cluster

    radius_sq = cluster_radius_m ** 2
    for b in bright:
        n_bright_neighbors = 0
        for other in bright:
            if other is b:
                continue
            dx = other.centroid_x - b.centroid_x
            dy = other.centroid_y - b.centroid_y
            if (dx * dx + dy * dy) <= radius_sq:
                n_bright_neighbors += 1
                if n_bright_neighbors >= min_neighbors:
                    break
        b.cool_island_suspect = n_bright_neighbors >= min_neighbors

    return bright_threshold


def compute_zone_stats(buildings: list["BuildingResult"]) -> Optional["ZoneStats"]:
    ok = [b for b in buildings if b.albedo is not None]
    if not ok:
        return None

    albedos = np.array([b.albedo for b in ok], dtype=np.float64)
    areas = np.array([max(b.area_m2, 0.0) for b in ok], dtype=np.float64)

    n = len(ok)
    mean = float(albedos.mean())
    median = float(np.median(albedos))
    std = float(albedos.std())
    amin = float(albedos.min())
    amax = float(albedos.max())

    total_area = areas.sum()
    area_weighted_mean = (
        float((albedos * areas).sum() / total_area) if total_area > 0 else mean
    )

    bins = np.linspace(0.0, 1.0, 11)  # 10 tranches de 0.1
    counts, _ = np.histogram(albedos, bins=bins)
    histogram = [
        (f"{bins[i]:.1f}-{bins[i+1]:.1f}", int(counts[i]))
        for i in range(len(counts))
    ]

    return ZoneStats(
        n=n, mean=mean, median=median, std=std, min=amin, max=amax,
        area_weighted_mean=area_weighted_mean, histogram=histogram,
    )
