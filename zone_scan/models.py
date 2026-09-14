"""
zone_scan/models.py — AlbedoNet App
====================================
Dataclasses partagees entre les sous-modules de zone_scan : une tuile
orthophoto telechargee (OrthoTile), le resultat par batiment
(BuildingResult), les statistiques agregees (ZoneStats) et le resultat
global d'un scan (ZoneScanResult).

Extrait de l'ancien zone_scan.py (Tier 3, decoupage en sous-modules) --
aucun changement de comportement, seulement de localisation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class OrthoTile:
    bounds: tuple[float, float, float, float]  # xmin, ymin, xmax, ymax (L93)
    path: Path


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
    heat_island_suspect: bool = False    # cf. islands.flag_heat_island_suspects()
    cool_island_suspect: bool = False    # cf. islands.flag_cool_island_suspects()
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
