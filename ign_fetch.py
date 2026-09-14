#!/usr/bin/env python3
"""
ign_fetch.py - Recuperation batiment + orthophoto IGN pour une adresse unique.

Adapte de 00a_fetch_vector.py (WFS batiments) et 00b_fetch_ign_rasters_hr.py
(WMS-R orthophoto) pour un usage temps reel par adresse (app Streamlit),
au lieu du traitement par zone/lot du pipeline d'entrainement.

Simplifications vs les scripts originaux (zone entiere -> une seule adresse) :
  - Rayon de recherche petit (par defaut 60 m) au lieu d'une zone de plusieurs km.
  - Une seule page WFS necessaire (quelques batiments max dans le rayon).
  - Une seule tuile WMS-R necessaire (patch << 5000 px / axe, pas de tuilage).
  - Pas de parallelisation (ThreadPoolExecutor) : un seul appel reseau par etape.

Logique conservee a l'identique depuis les scripts sources :
  - compute_bbox_lambert93() : meme formule que 00a (buffer circulaire L93).
  - Parametres WFS GetFeature (couche BDTOPO_V3:batiment, SRSNAME EPSG:2154) : identiques a 00a.
  - Parametres WMS GetMap (FORMAT image/geotiff) : identiques a 00b, SAUF le nom de couche
    (voir correctif 08/07/2026 ci-dessous).
  - Pas de cle API : le WMS-R IGN (data.geopf.fr/wms-r) est public, confirme par 00b
    (aucun header d'authentification, seul un User-Agent est envoye).

Correctif 08/07/2026 (bug WMS-R "LayerNotDefined"):
  - La couche ORTHOIMAGERY.ORTHOPHOTOS (utilisee par 00b) correspond desormais a la
    "pyramide mondiale" et n'est plus exposee sur l'endpoint raster data.geopf.fr/wms-r.
  - La couche BD ORTHO V3 20cm France entiere, servie sur wms-r, s'appelle
    HR.ORTHOIMAGERY.ORTHOPHOTOS -- confirme par le GetCapabilities WMS-R et par les
    retours de la communaute IGN/Geoplateforme sur ce meme type d'erreur suite a la
    bascule Geoplateforme 2023-2024. C'est ce nom qui est desormais utilise ici.
  - Le prefixe "/wms" (data.geopf.fr/wms-r/wms vs data.geopf.fr/wms-r) est sans effet
    en mode raster : WMS_BASE_URL reste donc inchange.

Selection du batiment correspondant a l'adresse (cascade, voir match_building()) :
  1. Point-in-polygon : le point geocode tombe dans un polygone -> confiance "high".
  2. Fallback distance : sinon, batiment le plus proche si distance <= BUILDING_MATCH_MAX_DISTANCE_M
     -> confiance "medium".
  3. Sinon -> confiance "none", l'app DOIT le signaler a l'utilisateur (filet de securite,
     l'image du toit + la carte permettent une verification visuelle / correction manuelle).
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import geopandas as gpd
import requests
from pyproj import Transformer
from requests.adapters import HTTPAdapter
from shapely.geometry import Point, box
from urllib3.util.retry import Retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("albedo.app.ign_fetch")

# ---------------------------------------------------------------------------
# Constantes -- identiques aux scripts sources sauf mention contraire
# ---------------------------------------------------------------------------

# --- WFS (identique a 00a_fetch_vector.py) ---
WFS_BASE_URL: str = "https://data.geopf.fr/wfs/ows"
WFS_VERSION: str = "2.0.0"
WFS_LAYER: str = "BDTOPO_V3:batiment"
WFS_OUTPUT_FORMAT: str = "application/json"
WFS_SRSNAME: str = "EPSG:2154"
WFS_PAGE_SIZE: int = 1000  # largement suffisant : un seul rayon de recherche court

# --- WMS-R (identique a 00b_fetch_ign_rasters_hr.py) ---
WMS_BASE_URL: str = "https://data.geopf.fr/wms-r"
WMS_VERSION: str = "1.3.0"
LAYER_ORTHO: str = "HR.ORTHOIMAGERY.ORTHOPHOTOS"
WMS_FORMAT_ORTHO: str = "image/geotiff"
WMS_CRS: str = "EPSG:2154"
MAX_PIXELS_PER_AXIS: int = 5_000  # limite serveur confirmee par 00b (HTTP 400 au-dela)

# --- Reseau ---
REQUEST_TIMEOUT_S: int = 30

CRS_WGS84: int = 4326
CRS_L93: int = 2154

# --- Parametres specifiques app (pas de zone/lot -> recherche locale) ---
DEFAULT_SEARCH_RADIUS_M: float = 60.0     # rayon autour du point geocode
ORTHO_PATCH_HALF_M: float = 30          # demi-cote de la tuile orthophoto telechargee
ORTHO_RESOLUTION_M: float = 0.20          # identique au pipeline d'entrainement

# --- Matching batiment <-> adresse (voir match_building()) ---
BUILDING_MATCH_MAX_DISTANCE_M: float = 15.0

# --- Selection multi-candidats (cf. suivi projet, correctifs bâtiments 2026-07-11) ---
# Rayon au dela duquel on considere un site "dense" (plusieurs batiments
# rapproches -> ambiguite type "complexe scolaire/industriel") et on durcit
# la selection automatique / on privilegie la selection manuelle.
DENSE_SITE_CANDIDATE_THRESHOLD: int = 3
# Distance max pour proposer un candidat dans la liste (plus large que le
# seuil de matching auto, pour laisser le choix visuel a l'utilisateur).
CANDIDATE_LIST_MAX_DISTANCE_M: float = 40.0
MAX_CANDIDATES_RETURNED: int = 6
# Poids de la surface dans le score composite (favorise les grands batiments
# quand plusieurs sont a distance comparable -- un Cool Roof est plus
# souvent pose sur un grand batiment que sur une dependance/garage proche).
AREA_SCORE_WEIGHT: float = 0.35


# ---------------------------------------------------------------------------
# Session HTTP (meme politique de retry que les scripts sources)
# ---------------------------------------------------------------------------

def build_http_session() -> requests.Session:
    retry_policy = Retry(
        total=3,
        backoff_factor=1.0,
        status_forcelist={500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_policy)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "albedo-app/1.0 (research)"})
    return session


# ---------------------------------------------------------------------------
# BBOX (logique identique a 00a.compute_bbox_lambert93)
# ---------------------------------------------------------------------------

def compute_bbox_lambert93(
    lon: float, lat: float, radius_m: float
) -> tuple[float, float, float, float]:
    """Point WGS84 + rayon -> BBOX carree englobante en Lambert 93 (EPSG:2154)."""
    if radius_m <= 0:
        raise ValueError(f"radius_m doit etre > 0, recu : {radius_m}")
    transformer = Transformer.from_crs(
        f"EPSG:{CRS_WGS84}", f"EPSG:{CRS_L93}", always_xy=True
    )
    x_l93, y_l93 = transformer.transform(lon, lat)
    circle = Point(x_l93, y_l93).buffer(radius_m)
    return circle.bounds  # (xmin, ymin, xmax, ymax)


def project_point_l93(lon: float, lat: float) -> Point:
    transformer = Transformer.from_crs(
        f"EPSG:{CRS_WGS84}", f"EPSG:{CRS_L93}", always_xy=True
    )
    x, y = transformer.transform(lon, lat)
    return Point(x, y)


# ---------------------------------------------------------------------------
# WFS batiments (parametres identiques a 00a, une seule page ici)
# ---------------------------------------------------------------------------

def fetch_buildings_near_point(
    session: requests.Session,
    lon: float,
    lat: float,
    radius_m: float = DEFAULT_SEARCH_RADIUS_M,
) -> gpd.GeoDataFrame:
    """Recupere les batiments BD TOPO dans un rayon autour d'un point WGS84.

    Reutilise les memes parametres WFS GetFeature que 00a_fetch_vector.py.
    Pas de pagination multi-page : un rayon de recherche de quelques dizaines
    de metres contient au plus quelques dizaines de batiments, largement
    sous WFS_PAGE_SIZE.
    """
    xmin, ymin, xmax, ymax = compute_bbox_lambert93(lon, lat, radius_m)
    bbox_str = f"{xmin:.4f},{ymin:.4f},{xmax:.4f},{ymax:.4f},{WFS_SRSNAME}"

    params = {
        "SERVICE": "WFS",
        "VERSION": WFS_VERSION,
        "REQUEST": "GetFeature",
        "TYPENAMES": WFS_LAYER,
        "OUTPUTFORMAT": WFS_OUTPUT_FORMAT,
        "SRSNAME": WFS_SRSNAME,
        "BBOX": bbox_str,
        "COUNT": str(WFS_PAGE_SIZE),
        "STARTINDEX": "0",
    }

    logger.info("WFS GetFeature autour de (lat=%.6f, lon=%.6f), rayon=%.0fm", lat, lon, radius_m)
    response = session.get(WFS_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_S)

    if 400 <= response.status_code < 500:
        logger.error(
            "Erreur WFS HTTP %d : %s", response.status_code, response.text[:500]
        )
        response.raise_for_status()
    response.raise_for_status()

    geojson = response.json()
    features = geojson.get("features", [])

    if not features:
        return gpd.GeoDataFrame(geometry=[], crs=f"EPSG:{CRS_L93}")

    gdf = gpd.GeoDataFrame.from_features(features, crs=f"EPSG:{CRS_L93}")
    logger.info("  -> %d batiment(s) trouve(s) dans le rayon.", len(gdf))
    return gdf


# ---------------------------------------------------------------------------
# Matching batiment <-> adresse geocodee (cascade + filet de securite)
# ---------------------------------------------------------------------------

@dataclass
class BuildingMatch:
    """Resultat du matching batiment pour une adresse.

    confidence:
        "high"   -> le point tombe dans le polygone du batiment retenu.
        "medium" -> pas de polygone contenant le point ; batiment le plus
                    proche retenu, sous le seuil BUILDING_MATCH_MAX_DISTANCE_M.
        "none"   -> aucun batiment fiable trouve. building/centroid sont None.
                    L'app DOIT afficher un avertissement explicite dans ce cas
                    (aucune inference automatique fiable n'est possible) et
                    laisser l'image/la carte servir de verification visuelle.
    """
    confidence: str
    building: Optional[gpd.GeoSeries]
    centroid_x: Optional[float]
    centroid_y: Optional[float]
    distance_m: Optional[float]
    n_candidates: int


def match_building(
    gdf_buildings: gpd.GeoDataFrame,
    point_l93: Point,
    max_distance_m: float = BUILDING_MATCH_MAX_DISTANCE_M,
) -> BuildingMatch:
    """Selectionne le batiment correspondant a un point geocode (cascade).

    1. Point-in-polygon (confiance "high").
    2. Batiment le plus proche si distance <= max_distance_m (confiance "medium").
    3. Sinon confiance "none" -> aucune inference automatique fiable.

    Note : ceci ne garantit jamais une correspondance parfaite (le point BAN
    peut representer la facade cote rue plutot que le centre du batiment).
    Le filet de securite reste l'affichage du toit + la carte dans l'app,
    pour verification/correction visuelle par l'utilisateur.
    """
    n_candidates = len(gdf_buildings)

    if n_candidates == 0:
        return BuildingMatch("none", None, None, None, None, 0)

    # --- 1. Point-in-polygon ---
    contains_mask = gdf_buildings.geometry.contains(point_l93)
    if contains_mask.any():
        matches = gdf_buildings[contains_mask]
        # S'il y a plusieurs polygones contenant le point (chevauchement rare),
        # on prend le plus petit (le plus specifique).
        building = matches.loc[matches.geometry.area.idxmin()]
        centroid = building.geometry.centroid
        return BuildingMatch(
            "high", building, centroid.x, centroid.y, 0.0, n_candidates
        )

    # --- 2. Fallback distance ---
    distances = gdf_buildings.geometry.distance(point_l93)
    idx_nearest = distances.idxmin()
    nearest_distance = distances.loc[idx_nearest]

    if nearest_distance <= max_distance_m:
        building = gdf_buildings.loc[idx_nearest]
        centroid = building.geometry.centroid
        return BuildingMatch(
            "medium", building, centroid.x, centroid.y,
            float(nearest_distance), n_candidates,
        )

    # --- 3. Aucune correspondance fiable ---
    return BuildingMatch("none", None, None, None, float(nearest_distance), n_candidates)


# ---------------------------------------------------------------------------
# Multi-candidats -- selection visuelle / ponderation surface (points 1,2,4,5)
# ---------------------------------------------------------------------------

@dataclass
class BuildingCandidate:
    """Un batiment candidat proche du point geocode, pour selection visuelle.

    thumb_row_off / thumb_col_off / thumb_size_px : coordonnees de la vignette
    DANS la tuile orthophoto large partagee (fetch_area_orthophoto), pas un
    fichier separe -- ca evite un appel WMS par candidat (cf. note vitesse
    dans fetch_candidates_with_thumbnails()).
    """
    building_id: int          # index dans le GeoDataFrame d'origine
    centroid_x: float
    centroid_y: float
    distance_m: float
    area_m2: float
    score: float               # plus haut = meilleur candidat (distance + surface)
    is_default_pick: bool      # True pour le candidat choisi automatiquement


def _score_candidate(distance_m: float, area_m2: float, max_distance: float, max_area: float) -> float:
    """Score composite [0, 1] : proximite ET surface comptent.

    Rationale (cf. suivi projet, point 4) : sur un site multi-batiments,
    un Cool Roof / grande toiture est plus souvent le "gros" batiment que
    la plus proche dependance. On combine donc un score de proximite
    (1 - distance normalisee) et un score de surface (surface normalisee),
    pondere par AREA_SCORE_WEIGHT.
    """
    dist_score = 1.0 - min(distance_m / max_distance, 1.0) if max_distance > 0 else 1.0
    area_score = min(area_m2 / max_area, 1.0) if max_area > 0 else 0.0
    return (1 - AREA_SCORE_WEIGHT) * dist_score + AREA_SCORE_WEIGHT * area_score


def list_building_candidates(
    gdf_buildings: gpd.GeoDataFrame,
    point_l93: Point,
    max_distance_m: float = CANDIDATE_LIST_MAX_DISTANCE_M,
    max_candidates: int = MAX_CANDIDATES_RETURNED,
) -> list[BuildingCandidate]:
    """Liste les batiments candidats proches, tries par score (proximite + surface).

    Utilise pour :
      - Point 1 (choix visuel) : le frontend affiche ces candidats en vignettes
        (donnees exposees par main.py).
      - Point 2 (site dense -> forcer selection manuelle) : voir
        is_dense_site() ci-dessous, base sur len(candidats).
      - Point 4 (ponderation surface) : le tri par score privilegie deja
        les grands batiments a distance comparable ; is_default_pick=True
        marque le meilleur candidat (utilisable en auto si pas de site dense).
      - Point 5 (correction a posteriori) : le frontend peut rouvrir cette liste
        apres coup pour permettre a l'utilisateur de changer son choix.
    """
    if len(gdf_buildings) == 0:
        return []

    distances = gdf_buildings.geometry.distance(point_l93)
    nearby_mask = distances <= max_distance_m
    if not nearby_mask.any():
        return []

    nearby = gdf_buildings[nearby_mask].copy()
    nearby["_distance_m"] = distances[nearby_mask]
    nearby["_area_m2"] = nearby.geometry.area

    max_area = nearby["_area_m2"].max()
    candidates: list[BuildingCandidate] = []
    for idx, row in nearby.iterrows():
        score = _score_candidate(row["_distance_m"], row["_area_m2"], max_distance_m, max_area)
        centroid = row.geometry.centroid
        candidates.append(BuildingCandidate(
            building_id=idx,
            centroid_x=centroid.x,
            centroid_y=centroid.y,
            distance_m=float(row["_distance_m"]),
            area_m2=float(row["_area_m2"]),
            score=score,
            is_default_pick=False,
        ))

    candidates.sort(key=lambda c: c.score, reverse=True)
    candidates = candidates[:max_candidates]
    if candidates:
        candidates[0].is_default_pick = True
    return candidates


def is_dense_site(candidates: list[BuildingCandidate]) -> bool:
    """True si le site a plusieurs batiments rapproches (point 2).

    Sur un site dense (ecole/gymnase/complexe industriel avec plusieurs
    batiments sur la meme parcelle), le matching automatique par distance
    seule est peu fiable -- le frontend DEVRAIT alors privilegier la selection
    manuelle visuelle plutot que le pick automatique.
    """
    return len(candidates) >= DENSE_SITE_CANDIDATE_THRESHOLD


# ---------------------------------------------------------------------------
# WMS-R orthophoto (parametres identiques a 00b, une seule tuile ici)
# ---------------------------------------------------------------------------

def fetch_orthophoto_patch(
    session: requests.Session,
    centroid_x: float,
    centroid_y: float,
    output_path: Path,
    half_extent_m: float = ORTHO_PATCH_HALF_M,
    resolution_m: float = ORTHO_RESOLUTION_M,
) -> Path:
    """Telecharge une tuile orthophoto unique autour d'un centroide L93.

    Reutilise les memes parametres WMS GetMap que 00b_fetch_ign_rasters_hr.py
    (FORMAT=image/geotiff, CRS=EPSG:2154), a l'exception du nom de couche : voir
    LAYER_ORTHO / correctif 08/07/2026 en tete de fichier (HR.ORTHOIMAGERY.ORTHOPHOTOS
    remplace ORTHOIMAGERY.ORTHOPHOTOS, qui n'est plus servie sur l'endpoint raster).
    Pas de tuilage : l'emprise demandee ici (quelques dizaines de metres)
    est tres largement sous la limite serveur de MAX_PIXELS_PER_AXIS px/axe.
    """
    xmin = centroid_x - half_extent_m
    xmax = centroid_x + half_extent_m
    ymin = centroid_y - half_extent_m
    ymax = centroid_y + half_extent_m

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
        "FORMAT": WMS_FORMAT_ORTHO,
    }

    logger.info(
        "WMS GetMap orthophoto : centre=(%.2f, %.2f) L93, %dx%d px",
        centroid_x, centroid_y, width_px, height_px,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = session.get(
        WMS_BASE_URL, params=params, timeout=REQUEST_TIMEOUT_S, stream=True
    )

    if response.status_code == 400:
        snippet = response.content[:500].decode("utf-8", errors="replace")
        raise RuntimeError(f"WMS-R HTTP 400 : {snippet}")
    response.raise_for_status()

    # --- Verif 1 : Content-Type de la reponse -------------------------------
    # Le moyen le plus fiable de detecter une reponse non-image AVANT de lire
    # le corps. Une reponse WMS en erreur (ou une page HTML de proxy/WAF
    # intermediaire) n'aura jamais un Content-Type image/tiff-like.
    content_type = response.headers.get("Content-Type", "").lower()
    VALID_CONTENT_TYPES = ("tiff", "geotiff", "octet-stream", "image/")
    if content_type and not any(tag in content_type for tag in VALID_CONTENT_TYPES):
        snippet = response.content[:500].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"WMS-R a renvoye un Content-Type inattendu ({content_type}) "
            f"au lieu d'un GeoTIFF. Extrait de la reponse : {snippet}"
        )

    # --- Verif 2 : signature des premiers octets ----------------------------
    # Filet de securite si le Content-Type est absent/mensonger. On elargit
    # la detection d'erreur XML/HTML (BOM UTF-8, <html>, JSON d'erreur, etc.)
    # en plus des marqueurs XML/OGC deja geres.
    ERROR_PREFIXES = (b"<?xml", b"<serv", b"<ows:", b"<html", b"<!doc", b"{\"err", b"{'err")
    TIFF_MAGIC = (b"II*\x00", b"MM\x00*")  # little/big-endian TIFF header

    first_chunk_checked = False
    with open(output_path, "wb") as fout:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            if not first_chunk_checked:
                first_chunk_checked = True
                # Ignore un eventuel BOM UTF-8 avant de comparer les prefixes.
                probe = chunk[3:] if chunk[:3] == b"\xef\xbb\xbf" else chunk
                probe_stripped = probe.lstrip()
                if probe_stripped[:5].lower() in ERROR_PREFIXES:
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"WMS-R a renvoye une erreur (non-GeoTIFF) : "
                        f"{probe[:300].decode('utf-8', errors='replace')}"
                    )
                if not probe.startswith(TIFF_MAGIC):
                    output_path.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"WMS-R a renvoye des donnees non reconnues comme TIFF "
                        f"(Content-Type={content_type or 'absent'}). "
                        f"Premiers octets : {probe[:60]!r}"
                    )
            fout.write(chunk)

    logger.info("Orthophoto ecrite : %s (%.1f Ko)", output_path, output_path.stat().st_size / 1024)
    return output_path


def fetch_area_orthophoto(
    session: requests.Session,
    lon: float,
    lat: float,
    output_path: Path,
    radius_m: float = CANDIDATE_LIST_MAX_DISTANCE_M,
    resolution_m: float = ORTHO_RESOLUTION_M,
) -> Path:
    """Telecharge UNE tuile large couvrant tout le rayon de recherche.

    Utilisee pour generer les vignettes de tous les candidats (point 1 / 5)
    en un seul appel WMS-R, au lieu d'un appel par candidat -- c'est ce qui
    garde la selection visuelle rapide meme avec plusieurs batiments a
    comparer (contrainte "vitesse fulgurante", cf. suivi projet).
    """
    point_l93 = project_point_l93(lon, lat)
    return fetch_orthophoto_patch(
        session, point_l93.x, point_l93.y, output_path,
        half_extent_m=radius_m, resolution_m=resolution_m,
    )


def crop_thumbnail(
    area_ortho_path: Path,
    centroid_x: float,
    centroid_y: float,
    half_extent_m: float = 8.0,
    thumb_px: int = 96,
) -> Optional["np.ndarray"]:  # noqa: F821
    """Decoupe une vignette RGB locale (pas d'appel reseau) depuis la tuile
    large deja telechargee par fetch_area_orthophoto().

    Reutilise rasterio directement ici (au lieu d'importer patch_extraction,
    pour eviter une dependance circulaire / le couplage a la normalisation
    min-max qui ne sert pas a l'affichage). Retourne un array (H, W, 3)
    uint8 pret pour st.image(), ou None si hors emprise.
    """
    import numpy as np
    import rasterio
    import rasterio.windows
    from rasterio.windows import from_bounds

    with rasterio.open(area_ortho_path) as src:
        win = from_bounds(
            left=centroid_x - half_extent_m,
            bottom=centroid_y - half_extent_m,
            right=centroid_x + half_extent_m,
            top=centroid_y + half_extent_m,
            transform=src.transform,
        ).round_lengths()
        try:
            win_clamped = win.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            if win_clamped.width == 0 or win_clamped.height == 0:
                return None
        except rasterio.errors.WindowError:
            return None

        data = src.read(window=win_clamped)[:3]  # (3, H, W)

    if data.shape[1] == 0 or data.shape[2] == 0:
        return None

    # HWC pour st.image(), resize simple si besoin (vignette, la qualite
    # LANCZOS complete de patch_extraction.py n'est pas necessaire ici).
    from PIL import Image
    img = Image.fromarray(np.moveaxis(data, 0, -1))
    img = img.resize((thumb_px, thumb_px), Image.BILINEAR)
    return np.array(img)


# ---------------------------------------------------------------------------
# Orchestration -- point d'entree unique utilise par main.py
# ---------------------------------------------------------------------------

@dataclass
class AddressFetchResult:
    match: BuildingMatch
    ortho_path: Optional[Path]
    point_l93: Point
    # --- Ajouts correctifs multi-batiments (cf. suivi projet 2026-07-11) ---
    candidates: list = None            # list[BuildingCandidate], vide si aucun proche
    area_ortho_path: Optional[Path] = None  # tuile large partagee pour les vignettes
    is_dense_site: bool = False        # cf. is_dense_site() -- signal pour le frontend

    def __post_init__(self):
        if self.candidates is None:
            self.candidates = []


def fetch_building_and_ortho(
    lat: float,
    lon: float,
    tmp_dir: Path,
    search_radius_m: float = DEFAULT_SEARCH_RADIUS_M,
    with_candidates: bool = True,
    session: Optional[requests.Session] = None,
) -> AddressFetchResult:
    """Point d'entree principal pour main.py : adresse geocodee -> batiment + orthophoto.

    Retourne un AddressFetchResult. Si match.confidence == "none", ortho_path
    est quand meme rempli si possible (centre sur le point geocode brut) pour
    permettre au frontend d'afficher une image de la zone malgre l'absence de
    correspondance batiment fiable -- l'utilisateur peut alors verifier
    visuellement et, si necessaire, le frontend doit clairement indiquer
    qu'aucune inference automatique n'a ete faite.

    with_candidates=True (par defaut) : calcule aussi la liste de candidats
    (point 1) et telecharge UNE tuile large partagee pour les vignettes
    (point 5), sans appel reseau supplementaire par candidat (cf.
    fetch_area_orthophoto -- contrainte vitesse). Mettre a False pour un
    traitement par lot ou la selection manuelle n'est pas necessaire
    (garde le comportement/la vitesse d'origine pour le batch automatique).

    session : permet d'injecter une requests.Session deja construite (et donc
    son pool de connexions/keep-alive) au lieu d'en creer une nouvelle a
    chaque appel. Comportement par defaut (session=None) inchange pour
    main.py : une session ephemere est creee ici, adaptee a un appel isole
    par interaction utilisateur. Pour un traitement par lot (des centaines/
    milliers d'appels, cf. build_training_dataset.py), passer une session
    partagee (idealement une par thread) evite de renegocier TLS a chaque
    ligne -- gain de vitesse notable sans changer la logique de matching.
    """
    if session is None:
        session = build_http_session()
    point_l93 = project_point_l93(lon, lat)

    gdf_buildings = fetch_buildings_near_point(session, lon, lat, search_radius_m)
    match = match_building(gdf_buildings, point_l93)

    ortho_path: Optional[Path] = None
    centroid_x = match.centroid_x if match.centroid_x is not None else point_l93.x
    centroid_y = match.centroid_y if match.centroid_y is not None else point_l93.y

    call_uid = uuid.uuid4().hex
    try:
        ortho_path = fetch_orthophoto_patch(
            session, centroid_x, centroid_y,
            output_path=tmp_dir / f"address_ortho_{call_uid}.tif",
        )
    except Exception as exc:
        logger.error("Echec telechargement orthophoto : %s", exc)
        ortho_path = None

    candidates: list = []
    area_ortho_path: Optional[Path] = None
    dense = False
    if with_candidates:
        candidates = list_building_candidates(gdf_buildings, point_l93)
        dense = is_dense_site(candidates)
        if candidates:
            try:
                area_ortho_path = fetch_area_orthophoto(
                    session, lon, lat,
                    output_path=tmp_dir / f"address_area_ortho_{call_uid}.tif",
                )
            except Exception as exc:
                logger.error("Echec telechargement tuile large (vignettes) : %s", exc)
                area_ortho_path = None
        if dense:
            logger.info(
                "Site dense detecte (%d candidats proches) : selection manuelle "
                "recommandee cote frontend.", len(candidates),
            )

    if match.confidence == "none":
        logger.warning(
            "Aucun batiment fiable trouve (candidats=%d, distance la plus proche=%s m). "
            "L'app DOIT signaler cela a l'utilisateur.",
            match.n_candidates,
            f"{match.distance_m:.1f}" if match.distance_m is not None else "n/a",
        )

    return AddressFetchResult(
        match=match, ortho_path=ortho_path, point_l93=point_l93,
        candidates=candidates, area_ortho_path=area_ortho_path, is_dense_site=dense,
    )