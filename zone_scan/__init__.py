"""
zone_scan — AlbedoNet App
===========================
Scan d'une zone (quartier, ville) au lieu d'une adresse unique : on donne un
centre (adresse) + un rayon, l'app recupere TOUS les batiments BD TOPO dans
la zone, telecharge l'orthophoto par tuiles (pas un appel WMS par batiment,
sinon le nombre de requetes explose des que la zone depasse quelques
batiments) puis calcule l'albedo de chaque batiment individuellement.

Package decoupe en sous-modules (Tier 3, dette structurelle -- ancien
zone_scan.py monolithique de 1122 lignes) :
  - models.py         : dataclasses partagees (OrthoTile, BuildingResult,
                         ZoneStats, ZoneScanResult)
  - tiles.py           : recuperation des batiments WFS + tuilage/telecharge-
                         ment orthophoto
  - islands.py         : detection ilots de chaleur/fraicheur + stats de zone
  - progress.py         : mapping etape -> pourcentage global
  - scan.py             : orchestration principale (scan_zone)
  - serialization.py    : pont SSE (scan_zone_events) + serialisation JSON
                         (zone_scan_result_to_json)

Ce fichier ne fait que ré-exporter l'API publique de l'ancien module unique,
pour que tout code existant (`from zone_scan import scan_zone`, etc.,
notamment main.py) continue de fonctionner sans aucune modification.
"""

from .islands import (
    BRIGHT_CLUSTER_RADIUS_M,
    COOL_SUSPECT_EXCLUDED_MATERIALS,
    DARK_CLUSTER_RADIUS_M,
    DARK_ROOF_ALBEDO_THRESHOLD,
    MIN_BRIGHT_NEIGHBORS,
    MIN_DARK_NEIGHBORS,
    SHADOW_FRACTION_MAX_HEAT,
    compute_zone_stats,
    flag_cool_island_suspects,
    flag_heat_island_suspects,
)
from .models import BuildingResult, OrthoTile, ZoneScanResult, ZoneStats
from .progress import _progress_percent, _STAGE_LABELS, _STAGE_WEIGHTS
from .scan import BATCH_INFERENCE_SIZE, MAX_ZONE_RADIUS_M, scan_zone
from .serialization import scan_zone_events, zone_scan_result_to_json
from .tiles import (
    PATCH_HALF_EXTENT_M,
    TILE_OVERLAP_M,
    TILE_SIDE_M,
    WFS_LAYER_BUILDING,
    WFS_PAGE_FETCH_WORKERS,
    WFS_PAGE_SIZE,
    _tile_contains_with_margin,
    download_all_tiles,
    fetch_buildings_in_zone,
    fetch_ortho_tile,
    find_tile_for_point,
    generate_tile_bounds,
)

__all__ = [
    # models
    "OrthoTile", "BuildingResult", "ZoneStats", "ZoneScanResult",
    # tiles
    "TILE_SIDE_M", "TILE_OVERLAP_M", "PATCH_HALF_EXTENT_M",
    "WFS_LAYER_BUILDING", "WFS_PAGE_SIZE", "WFS_PAGE_FETCH_WORKERS",
    "fetch_buildings_in_zone", "generate_tile_bounds", "fetch_ortho_tile",
    "download_all_tiles", "find_tile_for_point",
    # islands
    "DARK_ROOF_ALBEDO_THRESHOLD", "DARK_CLUSTER_RADIUS_M", "MIN_DARK_NEIGHBORS",
    "SHADOW_FRACTION_MAX_HEAT", "COOL_SUSPECT_EXCLUDED_MATERIALS",
    "BRIGHT_CLUSTER_RADIUS_M", "MIN_BRIGHT_NEIGHBORS",
    "flag_heat_island_suspects", "flag_cool_island_suspects", "compute_zone_stats",
    # scan
    "MAX_ZONE_RADIUS_M", "BATCH_INFERENCE_SIZE", "scan_zone",
    # serialization
    "scan_zone_events", "zone_scan_result_to_json",
]
