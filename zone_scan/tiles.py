"""
zone_scan/tiles.py — AlbedoNet App
====================================
1. Recuperation des batiments BD TOPO dans une zone (WFS, avec pagination
   parallelisee).
2. Decoupage de la zone en tuiles orthophoto + telechargement (WMS-R, une
   requete par tuile plutot qu'une requete WMS par batiment -- cf. le
   docstring d'origine de zone_scan.py pour le rationale complet).

Extrait de l'ancien zone_scan.py (Tier 3, decoupage en sous-modules) --
aucun changement de comportement, seulement de localisation.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import geopandas as gpd

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
    compute_bbox_lambert93,
)

from .models import OrthoTile

# ──────────────────────────────────────────────────────────────────────────
# Parametres de tuilage / requetes WFS
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

# Marge minimale (m) que la tuile doit laisser autour du centroide d'un
# batiment pour que l'extraction du patch (64x64 px, soit 12.8 m de cote a
# 0.20 m/px) soit garantie valide, sans clamp au bord.
PATCH_HALF_EXTENT_M: float = 8.0  # legere marge au-dessus de 12.8/2=6.4m

WFS_LAYER_BUILDING = "BDTOPO_V3:batiment"
WFS_PAGE_SIZE = 1000

# Nombre de pages WFS recuperees en parallele lors de la pagination d'une
# grosse zone (cf. fetch_buildings_in_zone).
WFS_PAGE_FETCH_WORKERS: int = 4


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
