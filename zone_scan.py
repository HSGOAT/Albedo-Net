"""
zone_scan.py — AlbedoNet App
==============================
Scan d'une zone (quartier, ville) au lieu d'une adresse unique : on donne un
centre (adresse) + un rayon, l'app recupere TOUS les batiments BD TOPO dans
la zone, telecharge l'orthophoto par tuiles (pas un appel WMS par batiment,
sinon le nombre de requetes explose des que la zone depasse quelques
batiments) puis calcule l'albedo de chaque batiment individuellement.

Reutilise directement (aucune duplication de logique) :
  - ign_fetch.build_http_session / compute_bbox_lambert93 / WFS_* / WMS_* /
    LAYER_ORTHO / ORTHO_RESOLUTION_M
  - patch_extraction.extract_and_normalize_patch (memes regles de decoupe
    et de normalisation min-max que le mode adresse unique / lot)
  - inference.predict_albedo (meme cache modele que le reste de l'app)
  - geocoding.geocode_address / resolve_city_key (le centre de la zone sert
    a resoudre UNE SEULE cle ville pour toute la zone -- simplification
    deliberee : une zone de quelques centaines de metres a quelques km ne
    traverse en pratique jamais deux zones climatiques/regionales, donc pas
    besoin de re-resoudre par batiment)

Limitations connues (a documenter dans le suivi projet une fois teste en
conditions reelles) :
  - Le decoupage en tuiles utilise un recouvrement (TILE_OVERLAP_M) pour que
    les batiments proches d'un bord de tuile restent malgre tout entierement
    couverts par au moins une tuile. Les batiments proches du bord EXTERIEUR
    de la zone globale (donc sans tuile voisine) peuvent etre partiellement
    hors-emprise -- meme filet de securite que ign_fetch (patch invalide ->
    resultat "erreur", pas de prediction hasardeuse).
  - Un seul modele (une seule cle ville) pour toute la zone, resolu depuis
    l'adresse centre. Pas de gestion par batiment individuel des cas
    "region ambigue" (contrairement au mode lot d'adresses).
  - Pas de filtre sur le type/la fonction du batiment (BD TOPO inclut aussi
    garages, annexes, etc.) -- a ajouter plus tard si le bruit sur les
    petites structures s'avere genant (cf. champ HAUTEUR disponible dans les
    proprietes BD TOPO si besoin d'un filtre par la suite).
"""

from __future__ import annotations

import queue
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

from config import GENERIC_FALLBACK_MODEL, get_checkpoint_path
from geocoding import geocode_address, resolve_city_key
from ign_fetch import (
    CRS_L93,
    LAYER_ORTHO,
    MAX_PIXELS_PER_AXIS,
    ORTHO_RESOLUTION_M,
    WFS_BASE_URL,
    WFS_OUTPUT_FORMAT,
    WFS_SRSNAME,
    WFS_VERSION,
    WMS_BASE_URL,
    WMS_CRS,
    WMS_VERSION,
    build_http_session,
    compute_bbox_lambert93,
)
from inference import predict_albedo_batch
from shadow_calibration import calibrate_albedo
from materials import classify_material
from patch_extraction import extract_raw_and_normalized_patch, patch_to_png_bytes, extract_display_patch
from postprocess_plausibility import (
    DEFAULT_ANCHOR_ALPHA,
    adjust_albedo_single,
    classify_plausibility_single,
)
from confidence import compute_confidence
from heat_island_calibration import get_dark_threshold, get_bright_threshold

# ──────────────────────────────────────────────────────────────────────────
# Parametres de zone
# ──────────────────────────────────────────────────────────────────────────

# Taille de chaque tuile orthophoto telechargee, en metres. A
# ORTHO_RESOLUTION_M=0.20 m/px, 800 m -> 4000 px/axe, sous la limite serveur
# MAX_PIXELS_PER_AXIS=5000 (marge de securite gardee volontairement, pour
# absorber l'arrondi largeur/hauteur).
TILE_SIDE_M: float = 800.0

# Recouvrement entre tuiles adjacentes : garantit qu'un batiment proche d'un
# bord de tuile est malgre tout entierement contenu (avec marge) dans au
# moins une des tuiles telechargees.
TILE_OVERLAP_M: float = 30.0

# Rayon maximum autorise pour un scan (garde-fou : au-dela, le nombre de
# tuiles/batiments devient deraisonnable pour un appel synchrone depuis
# Streamlit). A ajuster si besoin une fois teste en conditions reelles.
# Rayon maximum autorise pour un scan.
#
# NOTE 09/07/2026 : plafond releve a la demande de Pablo -- il ne veut plus
# etre arrete a ~3000 batiments / 2000m. Le vrai cout ici, c'est le temps de
# scan synchrone dans Streamlit (nombre de tuiles + nombre d'inferences), pas
# une limite technique dure. On garde neanmoins UNE limite (pas de "illimite"
# reel) pour eviter un scan de plusieurs dizaines de minutes qui timeout le
# navigateur/le serveur Streamlit sans jamais rendre de resultat partiel.
# A surveiller en usage reel : si un scan a 5000m devient trop lent, il
# faudra soit paralleliser davantage, soit streamer des resultats partiels
# plutot que de re-baisser ce plafond.
MAX_ZONE_RADIUS_M: float = 5000.0

# Marge minimale (m) que la tuile doit laisser autour du centroide d'un
# batiment pour que l'extraction du patch (64x64 px, soit 12.8 m de cote a
# 0.20 m/px) soit garantie valide, sans clamp au bord.
PATCH_HALF_EXTENT_M: float = 8.0  # legere marge au-dessus de 12.8/2=6.4m

WFS_LAYER_BUILDING = "BDTOPO_V3:batiment"
WFS_PAGE_SIZE = 1000

# Nombre de patches regroupes par forward pass lors de l'inference batchee
# (cf. phase 2 de scan_zone). Compromis memoire/latence CPU -- non re-mesure
# depuis la reintegration du 11/07/2026, valeur reprise telle quelle de la
# premiere implementation.
BATCH_INFERENCE_SIZE: int = 64

# Nombre de pages WFS recuperees en parallele lors de la pagination d'une
# grosse zone (cf. fetch_buildings_in_zone).
WFS_PAGE_FETCH_WORKERS: int = 4


# ──────────────────────────────────────────────────────────────────────────
# Progression "reelle" : mapping etape -> plage de pourcentage global.
# ──────────────────────────────────────────────────────────────────────────
# Chaque etape du pipeline recoit une plage [lo, hi] sur 0-100. Le callback
# recoit (i, total) POUR CETTE ETAPE UNIQUEMENT (ex: tuile 3/12) ; on projette
# lineairement i/total sur [lo, hi] pour obtenir un pourcentage global stable,
# utilisable tel quel par une barre de progression (SSE, Streamlit, etc.).
#
# Poids approximatifs bases sur le cout relatif observe en pratique :
# tuiles + extraction/inference dominent largement le temps total, geocodage
# et batiments (WFS) sont quasi-instantanes en comparaison.
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


# ──────────────────────────────────────────────────────────────────────────
# 1. Recuperation des batiments (WFS, avec pagination)
# ──────────────────────────────────────────────────────────────────────────

def fetch_buildings_in_zone(
    session,
    lon: float,
    lat: float,
    radius_m: float,
) -> gpd.GeoDataFrame:
    """Recupere TOUS les batiments BD TOPO dans le rayon, avec pagination WFS
    (contrairement a ign_fetch.fetch_buildings_near_point, prevu pour un
    petit rayon autour d'une seule adresse et donc sans pagination).

    Parallelisation : la premiere page renvoie generalement `numberMatched`
    (WFS 2.0.0) -- des qu'on le connait, les pages restantes sont recuperees
    en parallele (WFS_PAGE_FETCH_WORKERS threads) plutot qu'en serie.
    Fallback automatique sur la pagination sequentielle d'origine si le
    serveur ne renvoie pas ce champ.
    """
    xmin, ymin, xmax, ymax = compute_bbox_lambert93(lon, lat, radius_m)
    bbox_str = f"{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f},{WFS_SRSNAME}"

    def _base_params(start_index: int) -> dict:
        return {
            "SERVICE": "WFS",
            "VERSION": WFS_VERSION,
            "REQUEST": "GetFeature",
            "TYPENAMES": WFS_LAYER_BUILDING,
            "OUTPUTFORMAT": WFS_OUTPUT_FORMAT,
            "SRSNAME": WFS_SRSNAME,
            "BBOX": bbox_str,
            "COUNT": str(WFS_PAGE_SIZE),
            "STARTINDEX": str(start_index),
        }

    response = session.get(WFS_BASE_URL, params=_base_params(0), timeout=30)
    response.raise_for_status()
    geojson = response.json()
    first_features = geojson.get("features", [])
    number_matched = geojson.get("numberMatched")

    all_features: list = list(first_features)

    if len(first_features) == WFS_PAGE_SIZE and number_matched:
        n_pages_total = -(-int(number_matched) // WFS_PAGE_SIZE)  # ceil div
        remaining_starts = [
            i * WFS_PAGE_SIZE
            for i in range(1, n_pages_total)
            if i * WFS_PAGE_SIZE <= 100_000
        ]

        def _fetch_page(start_index: int) -> list:
            resp = session.get(WFS_BASE_URL, params=_base_params(start_index), timeout=30)
            resp.raise_for_status()
            return resp.json().get("features", [])

        if remaining_starts:
            with ThreadPoolExecutor(max_workers=WFS_PAGE_FETCH_WORKERS) as executor:
                futures = [executor.submit(_fetch_page, s) for s in remaining_starts]
                for future in as_completed(futures):
                    all_features.extend(future.result())

    elif len(first_features) == WFS_PAGE_SIZE:
        start_index = WFS_PAGE_SIZE
        while True:
            response = session.get(WFS_BASE_URL, params=_base_params(start_index), timeout=30)
            response.raise_for_status()
            features = response.json().get("features", [])
            all_features.extend(features)

            if len(features) < WFS_PAGE_SIZE:
                break
            start_index += WFS_PAGE_SIZE
            if start_index > 100_000:
                break

    if not all_features:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{CRS_L93}")

    return gpd.GeoDataFrame.from_features(all_features, crs=f"EPSG:{CRS_L93}")


# ──────────────────────────────────────────────────────────────────────────
# 2. Tuilage de la zone + telechargement orthophoto par tuile
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class OrthoTile:
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (L93)
    path: Path


def generate_tile_bounds(
    xmin: float, ymin: float, xmax: float, ymax: float,
    tile_side: float = TILE_SIDE_M,
    overlap: float = TILE_OVERLAP_M,
) -> list[tuple[float, float, float, float]]:
    """Decoupe une bbox en tuiles carrees de cote `tile_side`, avec
    recouvrement `overlap` entre tuiles adjacentes."""
    step = tile_side - overlap
    if step <= 0:
        raise ValueError("TILE_OVERLAP_M doit etre < TILE_SIDE_M")

    tiles = []
    y = ymin
    while y < ymax:
        x = xmin
        while x < xmax:
            tiles.append((x, y, x + tile_side, y + tile_side))
            x += step
        y += step
    return tiles


def fetch_ortho_tile(
    session,
    bounds: tuple[float, float, float, float],
    output_path: Path,
    resolution_m: float = ORTHO_RESOLUTION_M,
) -> Optional[Path]:
    """Telecharge une tuile orthophoto pour une bbox L93 donnee (variante
    "bbox directe" de ign_fetch.fetch_orthophoto_patch, qui elle prend un
    centroide + demi-etendue -- meme couche/format/CRS, meme politique
    d'erreur XML-au-lieu-de-GeoTIFF)."""
    xmin, ymin, xmax, ymax = bounds
    width_px = min(round((xmax - xmin) / resolution_m), MAX_PIXELS_PER_AXIS)
    height_px = min(round((ymax - ymin) / resolution_m), MAX_PIXELS_PER_AXIS)
    width_px, height_px = max(1, width_px), max(1, height_px)

    params = {
        "SERVICE": "WMS",
        "VERSION": WMS_VERSION,
        "REQUEST": "GetMap",
        "LAYERS": LAYER_ORTHO,
        "STYLES": "",
        "CRS": WMS_CRS,
        "BBOX": f"{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f}",
        "WIDTH": str(width_px),
        "HEIGHT": str(height_px),
        "FORMAT": "image/geotiff",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(WMS_BASE_URL, params=params, timeout=30, stream=True)

    if response.status_code == 400:
        return None
    response.raise_for_status()

    first_chunk_checked = False
    with open(output_path, "wb") as fout:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            if not first_chunk_checked:
                first_chunk_checked = True
                if chunk.lstrip()[:5].lower() in (b"<?xml", b"<serv", b"<ows:"):
                    output_path.unlink(missing_ok=True)
                    return None
            fout.write(chunk)

    return output_path


def download_all_tiles(
    session,
    zone_bounds: tuple[float, float, float, float],
    tmp_dir: Path,
    progress_callback=None,
    max_workers: int = 10,
) -> list[OrthoTile]:
    """Telecharge toutes les tuiles de la zone en parallele (I/O-bound,
    chaque tuile est un appel WMS-R independant -- meme raisonnement que
    pipeline.process_batch pour le mode lot d'adresses).
    
    max_workers augmente de 6 a 10 (09/07/2026 optim) : les tuiles sont du
    pur I/O reseau (pas de CPU-bound), donc peu de risque de rate-limit IGN
    et gain net de ~1.5-2× sur gros volumes. Surveiller les erreurs 429 si
    ca rate-limite ; revertir a 6-8 le cas echeant."""
    tile_bounds_list = generate_tile_bounds(*zone_bounds)
    total = len(tile_bounds_list)
    tiles_by_index: dict[int, OrthoTile] = {}
    progress_lock = threading.Lock()
    done = 0

    def _download_one(item: tuple[int, tuple[float, float, float, float]]):
        i, bounds = item
        path = tmp_dir / f"tile_{i:03d}.tif"
        result = fetch_ortho_tile(session, bounds, path)
        return i, bounds, result

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_download_one, (i, bounds))
            for i, bounds in enumerate(tile_bounds_list)
        ]
        for future in as_completed(futures):
            i, bounds, result = future.result()
            if result is not None:
                tiles_by_index[i] = OrthoTile(bounds=bounds, path=result)
            if progress_callback:
                with progress_lock:
                    done += 1
                    progress_callback(done, total)

    # Ordre des tuiles sans importance pour la suite (find_tile_for_point
    # les parcourt toutes), mais on garde un ordre stable par index.
    return [tiles_by_index[i] for i in sorted(tiles_by_index.keys())]


def _tile_contains_with_margin(
    bounds: tuple[float, float, float, float],
    x: float, y: float, margin: float,
) -> bool:
    xmin, ymin, xmax, ymax = bounds
    return (xmin + margin) <= x <= (xmax - margin) and (ymin + margin) <= y <= (ymax - margin)


def find_tile_for_point(tiles: list[OrthoTile], x: float, y: float) -> Optional[OrthoTile]:
    """Trouve la premiere tuile qui contient (x, y) avec une marge suffisante
    pour garantir une extraction de patch valide (pas de clamp au bord)."""
    for tile in tiles:
        if _tile_contains_with_margin(tile.bounds, x, y, PATCH_HALF_EXTENT_M):
            return tile
    # Fallback : aucune tuile ne contient le point AVEC marge (bord de zone) ;
    # on essaie quand meme la premiere tuile qui le contient sans marge --
    # extract_and_normalize_patch gere deja le clamp/rejet si trop de nodata.
    for tile in tiles:
        xmin, ymin, xmax, ymax = tile.bounds
        if xmin <= x <= xmax and ymin <= y <= ymax:
            return tile
    return None


# ──────────────────────────────────────────────────────────────────────────
# 3. Orchestration : adresse centre + rayon -> resultats par batiment
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class BuildingResult:
    building_id:  int
    centroid_lon: float
    centroid_lat: float
    centroid_x:   float  # L93
    centroid_y:   float  # L93
    area_m2:      float
    albedo:       Optional[float] = None
    albedo_brut:  Optional[float] = None  # avant calibration ombre, cf. shadow_calibration.py
    shadow_fraction: Optional[float] = None  # fraction du patch detectee en ombre
    material:     Optional[str] = None   # cf. materials.classify_material -- classifieur ML (v3.1, HistGradientBoosting)
    material_confidence: Optional[str] = None
    material_proba_max: Optional[float] = None  # cf. materials.MaterialResult.proba_max -- confiance du classifieur ML
    # OBSOLETES depuis materials.py v3.0 (passage a l'heuristique -> ML) :
    # MaterialResult n'expose plus mean_rgb ni distance_to_centroid, ces
    # champs restent definis (compat. calibrate_materials_from_osm.py qui
    # visait l'ancienne heuristique) mais ne sont PLUS alimentes -- toujours
    # None desormais. A retirer si calibrate_materials_from_osm.py est
    # lui-meme abandonne.
    mean_r:       Optional[float] = None
    mean_g:       Optional[float] = None
    mean_b:       Optional[float] = None
    distance_to_centroid: Optional[float] = None
    nodata_ratio: Optional[float] = None  # cf. patch_extraction.extract_and_normalize_patch_with_nodata_ratio
    plausibilite_materiau: Optional[str] = None  # cf. postprocess_plausibility.py
    albedo_ajuste_biblio: Optional[float] = None  # cf. postprocess_plausibility.py -- inerte tant que DEFAULT_ANCHOR_ALPHA=0.0
    confidence_score:   Optional[int] = None
    confidence_niveau:  Optional[str] = None
    confidence_raisons: Optional[str] = None  # raisons jointes par " | " pour rester une seule colonne CSV
    heat_island_suspect: bool = False    # cf. flag_heat_island_suspects()
    cool_island_suspect: bool = False    # cf. flag_cool_island_suspects()
    # Vignette RGB (PNG, en memoire) -- cf. suivi projet, galerie visuelle en
    # mode zone. Meme rationale que pipeline.AddressResult.thumbnail_png :
    # les tuiles orthophoto sont supprimees (TemporaryDirectory) avant que
    # le frontend ne puisse les afficher, donc on garde la vignette en memoire,
    # generee gratuitement a partir du patch deja extrait pour l'inference.
    thumbnail_png: Optional[bytes] = None
    # Crop plus large (16m de cote par defaut), distinct du patch modele
    # (12.8m, celui utilise pour thumbnail_png ci-dessus) -- destine
    # exclusivement a la vue plein ecran (clic sur une vignette), pour
    # eviter d'afficher en grand un patch trop serre/flou. Ne remplace pas
    # thumbnail_png (toujours utilise pour les petites vignettes de liste).
    detail_png: Optional[bytes] = None
    error:        Optional[str] = None


@dataclass
class ZoneStats:
    """Statistiques agregees sur les batiments avec prediction valide."""
    n:               int
    mean:            float
    median:          float
    std:             float
    min:             float
    max:             float
    area_weighted_mean: float  # moyenne ponderee par la surface au sol du batiment
    histogram:       list[tuple[str, int]] = field(default_factory=list)  # [("0.0-0.1", 3), ...]


# ──────────────────────────────────────────────────────────────────────────
# Detection ilots de chaleur : seuil absolu d'albedo + clustering spatial
# ──────────────────────────────────────────────────────────────────────────
# Un vrai ilot de chaleur est un effet de CONCENTRATION spatiale de toits
# sombres, pas un bâtiment isole. On combine donc deux criteres, les DEUX
# doivent etre vrais pour flaguer un batiment :
#
#   1. Seuil absolu d'albedo : le toit est reellement sombre en valeur
#      absolue (independamment de la moyenne de la zone). Un albedo sous ~0.15
#      correspond typiquement a du zinc/ardoise fonces, du bitume, etc.
#   2. Clustering : le batiment a au moins MIN_DARK_NEIGHBORS autres toits
#      sombres (meme seuil) dans un rayon DARK_CLUSTER_RADIUS_M.
#
# Ce choix (ET, pas OU) evite les deux biais d'une approche purement
# relative (z-score) :
#   - flaguer un toit juste "un peu moins clair que ses voisins" dans une
#     zone deja globalement claire (faux positif)
#   - rater un vrai cluster de toits sombres dans une zone globalement
#     sombre, ou le z-score ne verrait aucun ecart (faux negatif)

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


# Rayon et nombre de voisins reutilises pour la symetrie clair/fonce -- meme
# logique de clustering, juste inversee (toits CLAIRS regroupes).
BRIGHT_CLUSTER_RADIUS_M: float = DARK_CLUSTER_RADIUS_M
MIN_BRIGHT_NEIGHBORS: int = MIN_DARK_NEIGHBORS


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


@dataclass
class ZoneScanResult:
    center_address:   str
    center_lat:       float
    center_lon:       float
    radius_m:         float
    city_key_used:    str
    n_buildings:      int
    n_tiles:          int
    buildings:        list[BuildingResult] = field(default_factory=list)
    stats:            Optional[ZoneStats] = None
    dark_threshold_used: Optional[float] = None  # seuil ilot chaleur applique, cf. heat_island_calibration
    bright_threshold_used: Optional[float] = None  # seuil ilot fraicheur applique
    error:            Optional[str] = None


def scan_zone(
    center_address: str,
    radius_m: float,
    city_key_override: Optional[str] = None,
    progress_callback=None,
) -> ZoneScanResult:
    """Point d'entree principal pour l'endpoint /api/zone de main.py
    (section "Scan de zone" du frontend).

    Args:
        center_address: adresse texte libre servant de centre de la zone.
        radius_m: rayon de la zone en metres (borne par MAX_ZONE_RADIUS_M).
        city_key_override: force une cle ville (si None, resolue depuis
            l'adresse centre, comme pour une adresse unique).
        progress_callback: callback optionnel (etape:str, i:int, total:int)
            pour mettre a jour une barre de progression (Streamlit, SSE...).
            Etapes possibles, dans l'ordre : "geocodage", "batiments",
            "tuiles", "inference" (phase 1 : extraction patch/materiau par
            batiment), "inference_batch" (phase 2 : forward pass ViT par
            lots), "stats" (clustering ilots + agregats finaux). Utiliser
            _progress_percent(stage, i, total) pour convertir en un
            pourcentage global 0-100 coherent entre etapes.

    Returns:
        ZoneScanResult. `error` rempli si le pipeline s'arrete avant d'avoir
        pu traiter le moindre batiment (geocodage ou WFS en echec).
    """
    radius_m = min(radius_m, MAX_ZONE_RADIUS_M)

    if progress_callback:
        progress_callback("geocodage", 0, 1)
    try:
        geocoded = geocode_address(center_address)
    except Exception as exc:
        return ZoneScanResult(
            center_address=center_address, center_lat=0.0, center_lon=0.0,
            radius_m=radius_m, city_key_used="", n_buildings=0, n_tiles=0,
            error=f"Erreur reseau geocodage : {exc}",
        )

    if not geocoded.found:
        return ZoneScanResult(
            center_address=center_address, center_lat=0.0, center_lon=0.0,
            radius_m=radius_m, city_key_used="", n_buildings=0, n_tiles=0,
            error="Adresse introuvable via l'API BAN.",
        )

    if city_key_override:
        city_key = city_key_override
    else:
        city_resolution = resolve_city_key(geocoded)
        if city_resolution.city_key:
            city_key = city_resolution.city_key
        elif city_resolution.region_candidates:
            # Pas d'utilisateur pour trancher ici (meme convention que le
            # mode lot "automatique") : premier candidat par defaut.
            city_key = city_resolution.region_candidates[0]
        else:
            city_key = None  # -> fallback generique

    checkpoint_path = get_checkpoint_path(city_key) if city_key else GENERIC_FALLBACK_MODEL
    city_key_label = city_key if city_key else "generique"

    session = build_http_session()

    if progress_callback:
        progress_callback("batiments", 0, 1)
    try:
        gdf_buildings = fetch_buildings_in_zone(session, geocoded.lon, geocoded.lat, radius_m)
    except Exception as exc:
        return ZoneScanResult(
            center_address=center_address, center_lat=geocoded.lat, center_lon=geocoded.lon,
            radius_m=radius_m, city_key_used=city_key_label, n_buildings=0, n_tiles=0,
            error=f"Echec recuperation batiments (WFS) : {exc}",
        )

    if len(gdf_buildings) == 0:
        return ZoneScanResult(
            center_address=center_address, center_lat=geocoded.lat, center_lon=geocoded.lon,
            radius_m=radius_m, city_key_used=city_key_label, n_buildings=0, n_tiles=0,
            error="Aucun batiment BD TOPO trouve dans cette zone.",
        )

    zone_bounds = compute_bbox_lambert93(geocoded.lon, geocoded.lat, radius_m)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        def _tile_progress(i, total):
            if progress_callback:
                progress_callback("tuiles", i, total)

        tiles = download_all_tiles(session, zone_bounds, tmp_dir, progress_callback=_tile_progress)

        if not tiles:
            return ZoneScanResult(
                center_address=center_address, center_lat=geocoded.lat, center_lon=geocoded.lon,
                radius_m=radius_m, city_key_used=city_key_label,
                n_buildings=len(gdf_buildings), n_tiles=0,
                error="Echec telechargement de toutes les tuiles orthophoto.",
            )

        # Transformer L93 -> WGS84 pour l'affichage (centroides lat/lon)
        from pyproj import Transformer
        to_wgs84 = Transformer.from_crs(f"EPSG:{CRS_L93}", "EPSG:4326", always_xy=True)

        total_buildings = len(gdf_buildings)
        rows = list(gdf_buildings.iterrows())
        progress_lock = threading.Lock()
        done = 0

        def _extract_building(item) -> Optional[tuple["BuildingResult", Optional[np.ndarray], Optional[np.ndarray]]]:
            """Phase 1 (threadee) : localisation tuile + decoupe patch (avec
            ratio de nodata) + classification materiau. PAS d'inference ici
            -- l'albedo est rempli en phase 2, en lot.
            """
            i, row = item
            geom = row.geometry
            if geom is None or geom.is_empty:
                return None
            centroid = geom.centroid
            cx, cy = centroid.x, centroid.y
            lon_b, lat_b = to_wgs84.transform(cx, cy)
            area_m2 = float(geom.area)

            br = BuildingResult(
                building_id=i,
                centroid_lon=lon_b, centroid_lat=lat_b,
                centroid_x=cx, centroid_y=cy,
                area_m2=area_m2,
            )

            tile = find_tile_for_point(tiles, cx, cy)
            if tile is None:
                br.error = "Aucune tuile orthophoto ne couvre ce batiment."
                return br, None, None

            try:
                raw_patch, patch, nodata_ratio = extract_raw_and_normalized_patch(tile.path, cx, cy)
            except Exception as exc:
                br.error = f"Echec extraction patch : {exc}"
                return br, None, None

            if patch is None:
                br.error = "Patch invalide (hors emprise ou trop de nodata)."
                return br, None, None

            br.nodata_ratio = nodata_ratio

            # Vignette pour verification visuelle (galerie zone) -- cf.
            # note thumbnail_png plus haut. best-effort, jamais bloquant.
            try:
                br.thumbnail_png = patch_to_png_bytes(raw_patch)
            except Exception:
                pass

            # Crop large pour la vue plein ecran (cf. note detail_png plus
            # haut) -- meme tuile deja en memoire, pas d'appel reseau
            # supplementaire. best-effort, jamais bloquant : si ca echoue,
            # le frontend retombe sur thumbnail_png (cf. index.html).
            try:
                wide_patch = extract_display_patch(tile.path, cx, cy)
                if wide_patch is not None:
                    br.detail_png = patch_to_png_bytes(wide_patch)
            except Exception:
                pass

            try:
                mat = classify_material(patch)
                br.material = mat.material
                br.material_confidence = mat.confidence
                br.material_proba_max = mat.proba_max
            except Exception:
                pass

            return br, patch, raw_patch

        # ── Phase 1 : extraction des patches (threadee, I/O disque + resize) ──
        extracted: list[tuple[BuildingResult, Optional[np.ndarray], Optional[np.ndarray]]] = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(_extract_building, item) for item in rows]
            for future in as_completed(futures):
                res = future.result()
                if res is None:
                    continue
                br, patch, raw_patch = res
                extracted.append((br, patch, raw_patch))
                if progress_callback:
                    with progress_lock:
                        done += 1
                        progress_callback("inference", done, total_buildings)

        # ── Phase 2 : inference groupee par lots (un seul forward pass ViT
        # pour BATCH_INFERENCE_SIZE patches, au lieu d'un forward pass par
        # batiment -- reduit l'overhead Python/GIL sur les grosses zones) ──
        results: list[BuildingResult] = []
        to_infer = [(br, patch, raw_patch) for br, patch, raw_patch in extracted if patch is not None]
        results.extend(br for br, patch, _raw in extracted if patch is None)

        n_chunks = -(-len(to_infer) // BATCH_INFERENCE_SIZE) if to_infer else 0
        for chunk_idx, i in enumerate(range(0, len(to_infer), BATCH_INFERENCE_SIZE)):
            chunk = to_infer[i:i + BATCH_INFERENCE_SIZE]
            chunk_brs = [br for br, _, _ in chunk]
            chunk_patches = [patch for _, patch, _ in chunk]
            chunk_raw_patches = [raw_patch for _, _, raw_patch in chunk]
            try:
                albedos = predict_albedo_batch(chunk_patches, checkpoint_path)
                for br, albedo, raw_patch in zip(chunk_brs, albedos, chunk_raw_patches):
                    br.albedo = albedo
                    # Calibration post-prediction du biais d'ombre (cf.
                    # shadow_calibration.py) -- opere sur raw_patch, jamais
                    # renvoye au modele, best-effort.
                    try:
                        calib = calibrate_albedo(albedo, raw_patch, br.material)
                        br.albedo_brut = calib.albedo_brut
                        br.albedo = calib.albedo_corrige
                        br.shadow_fraction = calib.shadow_fraction
                    except Exception:
                        pass
            except Exception as exc:
                for br in chunk_brs:
                    br.error = f"Echec inference (batch) : {exc}"
            results.extend(chunk_brs)
            if progress_callback and n_chunks:
                progress_callback("inference_batch", chunk_idx + 1, n_chunks)

        # ── Phase 3 : plausibilite materiau + score de confiance -- necessite
        # l'albedo (phase 2 terminee), donc separee de l'extraction. Pas de
        # matching adresse->batiment en mode zone (batiments issus du WFS
        # directement) -- building_match_confidence reste None, cf.
        # confidence.py (signal absent = ignore, pas une degradation).
        for br in results:
            if br.albedo is None or br.material is None:
                continue
            br.plausibilite_materiau = classify_plausibility_single(br.material, br.albedo)
            br.albedo_ajuste_biblio = adjust_albedo_single(
                br.material, br.albedo, br.plausibilite_materiau,
                alpha=DEFAULT_ANCHOR_ALPHA, only_normal=True,
            )
            conf = compute_confidence(
                plausibilite_materiau=br.plausibilite_materiau,
                building_match_confidence=None,
                nodata_ratio=br.nodata_ratio,
                city_key=city_key,
                building_lat=br.centroid_lat,
                building_lon=br.centroid_lon,
                material_proba_max=br.material_proba_max,
            )
            br.confidence_score = conf.score
            br.confidence_niveau = conf.niveau
            br.confidence_raisons = " | ".join(conf.raisons)

        # Retrie par building_id pour un affichage/export stable (l'ordre
        # d'arrivee des threads est non deterministe).
        results.sort(key=lambda b: b.building_id)

    used_threshold = flag_heat_island_suspects(results, city_key=city_key)
    used_bright_threshold = flag_cool_island_suspects(results, city_key=city_key)
    stats = compute_zone_stats(results)
    if progress_callback:
        progress_callback("stats", 1, 1)

    return ZoneScanResult(
        center_address=center_address,
        center_lat=geocoded.lat, center_lon=geocoded.lon,
        radius_m=radius_m, city_key_used=city_key_label,
        n_buildings=len(gdf_buildings), n_tiles=len(tiles),
        buildings=results, stats=stats,
        dark_threshold_used=used_threshold,
        bright_threshold_used=used_bright_threshold,
    )


# ──────────────────────────────────────────────────────────────────────────
# 4. Variante generatrice (pont thread -> generateur), pour un consommateur
#    "au fil de l'eau" comme une route Flask en Server-Sent Events.
# ──────────────────────────────────────────────────────────────────────────

def scan_zone_events(
    center_address: str,
    radius_m: float,
    city_key_override: Optional[str] = None,
):
    """Variante generatrice de scan_zone(), pour brancher directement sur du
    SSE (ou tout autre consommateur qui veut la progression au fil de l'eau
    plutot qu'un unique retour bloquant).

    scan_zone() est synchrone et appelle progress_callback(stage, i, total)
    en cours de route. On la lance dans un thread separe ; le callback pousse
    chaque evenement dans une queue.Queue, et CE generateur (execute dans le
    thread appelant, ex: le thread de requete Flask) relit la queue et yield
    un dict pret a serialiser en JSON pour chaque evenement.

    Yields:
        - Des dicts de progression : {"stage", "message", "progress" (0-100),
          "n_done", "n_total"} au fur et a mesure du scan.
        - Un dernier dict {"done": True, "result": ZoneScanResult} en cas de
          succes, ou {"done": True, "error": "..."} si une exception non
          geree a echappe a scan_zone() (ne devrait normalement pas arriver,
          scan_zone() catche deja ses erreurs connues dans ZoneScanResult.error
          -- filet de securite pour ne jamais bloquer le generateur/le client
          SSE indefiniment).

    Usage typique cote Flask (voir aussi zone_scan_result_to_json) :

        def generate():
            for event in scan_zone_events(address, radius_m):
                if event.get("done"):
                    if "error" in event:
                        yield sse("error_event", {"error": event["error"]})
                    else:
                        yield sse("done", zone_scan_result_to_json(event["result"]))
                else:
                    yield sse("progress", event)
        return Response(stream_with_context(generate()), mimetype="text/event-stream")
    """
    q: "queue.Queue" = queue.Queue()
    _SENTINEL = object()
    result_holder: dict = {}

    def _callback(stage: str, i: int, total: int) -> None:
        q.put({
            "stage": stage,
            "message": _STAGE_LABELS.get(stage, stage),
            "progress": round(_progress_percent(stage, i, total), 1),
            "n_done": i,
            "n_total": total,
        })

    def _run() -> None:
        try:
            result_holder["result"] = scan_zone(
                center_address, radius_m,
                city_key_override=city_key_override,
                progress_callback=_callback,
            )
        except Exception as exc:
            result_holder["exception"] = exc
        finally:
            q.put(_SENTINEL)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    while True:
        item = q.get()
        if item is _SENTINEL:
            break
        yield item

    thread.join()

    if "exception" in result_holder:
        yield {"done": True, "error": str(result_holder["exception"])}
    else:
        yield {"done": True, "result": result_holder["result"]}


def zone_scan_result_to_json(result: "ZoneScanResult") -> dict:
    """Serialise un ZoneScanResult dans le format JSON attendu par le front
    (index.html, onglet "Scan de zone") : memes cles que l'ancienne route
    /api/zone (n_buildings, n_tiles, stats, suspects[], center_lat/lon,
    error). suspects[] ne contient QUE les batiments avec un albedo valide
    et flagges heat/cool (le front n'affiche que ceux-la dans les galeries
    et sur la carte de clusters) ; base64 encode les vignettes PNG a la
    volee, elles ne sont jamais persistees sur disque."""
    import base64

    if result.error:
        return {
            "error": result.error,
            "center_lat": result.center_lat,
            "center_lon": result.center_lon,
            "n_buildings": result.n_buildings,
            "n_tiles": result.n_tiles,
        }

    stats_json = None
    if result.stats is not None:
        s = result.stats
        stats_json = {
            "mean": s.mean, "median": s.median, "std": s.std,
            "area_weighted_mean": s.area_weighted_mean,
            "min": s.min, "max": s.max,
            "histogram": s.histogram,
        }

    suspects = []
    for b in result.buildings:
        if b.albedo is None or not (b.heat_island_suspect or b.cool_island_suspect):
            continue
        suspects.append({
            "lat": b.centroid_lat,
            "lon": b.centroid_lon,
            "albedo": b.albedo,
            "heat_suspect": b.heat_island_suspect,
            "cool_suspect": b.cool_island_suspect,
            "material": b.material,
            "material_confidence": b.material_confidence,
            "material_proba_max": b.material_proba_max,
            "confidence_niveau": b.confidence_niveau,
            "confidence_score": b.confidence_score,
            "thumbnail_png": (
                base64.b64encode(b.thumbnail_png).decode("ascii")
                if b.thumbnail_png else None
            ),
            "detail_png": (
                base64.b64encode(b.detail_png).decode("ascii")
                if b.detail_png else None
            ),
        })

    return {
        "center_lat": result.center_lat,
        "center_lon": result.center_lon,
        "n_buildings": result.n_buildings,
        "n_tiles": result.n_tiles,
        "stats": stats_json,
        "suspects": suspects,
        "dark_threshold_used": result.dark_threshold_used,
        "bright_threshold_used": result.bright_threshold_used,
        "city_key_used": result.city_key_used,
    }