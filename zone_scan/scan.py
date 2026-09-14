"""
zone_scan/scan.py — AlbedoNet App
===================================
Orchestration : adresse centre + rayon -> resultats par batiment.
Point d'entree principal (scan_zone), utilise par l'endpoint /api/zone et
par scan_zone_events (cf. serialization.py) pour la variante SSE.

Reutilise directement (aucune duplication de logique) :
  - ign_fetch.build_http_session / compute_bbox_lambert93
  - patch_extraction.extract_and_normalize_patch (memes regles de decoupe
    et de normalisation min-max que le mode adresse unique / lot)
  - inference.predict_albedo_batch (meme cache modele que le reste de l'app)
  - geocoding.geocode_address / resolve_city_key (le centre de la zone sert
    a resoudre UNE SEULE cle ville pour toute la zone -- simplification
    deliberee : une zone de quelques centaines de metres a quelques km ne
    traverse en pratique jamais deux zones climatiques/regionales, donc pas
    besoin de re-resoudre par batiment)

Limitations connues (a documenter dans le suivi projet une fois teste en
conditions reelles) :
  - Le decoupage en tuiles utilise un recouvrement (tiles.TILE_OVERLAP_M)
    pour que les batiments proches d'un bord de tuile restent malgre tout
    entierement couverts par au moins une tuile. Les batiments proches du
    bord EXTERIEUR de la zone globale (donc sans tuile voisine) peuvent
    etre partiellement hors-emprise -- meme filet de securite que ign_fetch
    (patch invalide -> resultat "erreur", pas de prediction hasardeuse).
  - Un seul modele (une seule cle ville) pour toute la zone, resolu depuis
    l'adresse centre. Pas de gestion par batiment individuel des cas
    "region ambigue" (contrairement au mode lot d'adresses).
  - Pas de filtre sur le type/la fonction du batiment (BD TOPO inclut aussi
    garages, annexes, etc.) -- a ajouter plus tard si le bruit sur les
    petites structures s'avere genant (cf. champ HAUTEUR disponible dans les
    proprietes BD TOPO si besoin d'un filtre par la suite).

Extrait de l'ancien zone_scan.py (Tier 3, decoupage en sous-modules) --
aucun changement de comportement, seulement de localisation.
"""

from __future__ import annotations

import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from config import GENERIC_FALLBACK_MODEL, get_checkpoint_path
from geocoding import geocode_address, resolve_city_key
from ign_fetch import CRS_L93, build_http_session, compute_bbox_lambert93
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

from .islands import compute_zone_stats, flag_cool_island_suspects, flag_heat_island_suspects
from .models import BuildingResult, ZoneScanResult
from .tiles import download_all_tiles, fetch_buildings_in_zone, find_tile_for_point

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

# Nombre de patches regroupes par forward pass lors de l'inference batchee
# (cf. phase 2 de scan_zone). Compromis memoire/latence CPU -- non re-mesure
# depuis la reintegration du 11/07/2026, valeur reprise telle quelle de la
# premiere implementation.
BATCH_INFERENCE_SIZE: int = 64


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
            progress._progress_percent(stage, i, total) pour convertir en un
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
            # note thumbnail_png dans models.py. best-effort, jamais bloquant.
            try:
                br.thumbnail_png = patch_to_png_bytes(raw_patch)
            except Exception:
                pass

            # Crop large pour la vue plein ecran (cf. note detail_png dans
            # models.py) -- meme tuile deja en memoire, pas d'appel reseau
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
