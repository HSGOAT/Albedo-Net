"""
pipeline.py — AlbedoNet App
=============================
Pipeline unique adresse -> albedo, factorise pour etre appele a l'identique
depuis le mode "adresse unique" et le mode "lot d'adresses" de main.py.

En mode lot, il n'y a personne pour repondre a un menu "quelle ville
choisir parmi ces 3 candidats region ?" -- resolve_batch() applique donc une
regle de decision automatique explicite (premier candidat de la liste,
qui est la ville "grande metropole" de la region par convention de
config.REGION_TO_CITY_KEYS) et le signale clairement dans le resultat
(champ `ambiguous=True`) plutot que de choisir silencieusement.
"""

from __future__ import annotations

import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from config import CITY_MODELS, GENERIC_FALLBACK_MODEL, get_checkpoint_path
from confidence import compute_confidence
from geocoding import geocode_address, resolve_city_key
from ign_fetch import fetch_building_and_ortho
from inference import predict_albedo
from materials import classify_material
from patch_extraction import extract_raw_and_normalized_patch, patch_to_png_bytes
from postprocess_plausibility import (
    DEFAULT_ANCHOR_ALPHA,
    adjust_albedo_single,
    classify_plausibility_single,
)
from shadow_calibration import calibrate_albedo
from versioning import build_run_metadata


@dataclass
class CityPreResolution:
    """Resultat du pre-scan (geocodage + resolution ville, SANS appel IGN).

    Utilise en mode lot "manuel" pour lister les adresses ambigues avant de
    lancer le traitement complet (evite de telecharger IGN pour rien si
    l'utilisateur veut d'abord choisir).
    """
    index:             int
    address:           str
    found:             bool
    address_label:     Optional[str] = None
    auto_city_key:     Optional[str] = None   # None si "generique" ou si ambigu
    region:            Optional[str] = None
    candidates:        list = field(default_factory=list)  # options a proposer si ambigu
    is_ambiguous:      bool = False
    error:             Optional[str] = None


def pre_resolve_addresses(addresses: list[str]) -> list[CityPreResolution]:
    """Geocode et resout la ville pour chaque adresse, sans toucher a l'IGN.

    Rapide (un seul appel API BAN par adresse) : sert a construire l'ecran
    de confirmation du mode lot "manuel", avant de lancer le traitement
    complet (potentiellement long) sur toutes les adresses.
    """
    results: list[CityPreResolution] = []

    for i, addr in enumerate(addresses):
        addr = addr.strip()
        if not addr:
            continue

        try:
            geocoded = geocode_address(addr)
        except Exception as exc:
            results.append(CityPreResolution(
                index=i, address=addr, found=False,
                error=f"Erreur reseau geocodage : {exc}",
            ))
            continue

        if not geocoded.found:
            results.append(CityPreResolution(
                index=i, address=addr, found=False,
                error="Adresse introuvable via l'API BAN.",
            ))
            continue

        city_resolution = resolve_city_key(geocoded)

        if city_resolution.city_key:
            # Match direct fiable -> rien a demander
            results.append(CityPreResolution(
                index=i, address=addr, found=True,
                address_label=geocoded.address_label,
                auto_city_key=city_resolution.city_key,
            ))
        elif len(city_resolution.region_candidates) > 1:
            # Region deduite, plusieurs modeles candidats -> ambigu
            results.append(CityPreResolution(
                index=i, address=addr, found=True,
                address_label=geocoded.address_label,
                region=city_resolution.region,
                candidates=city_resolution.region_candidates,
                is_ambiguous=True,
            ))
        elif len(city_resolution.region_candidates) == 1:
            # Region deduite, un seul candidat -> pas besoin de demander
            results.append(CityPreResolution(
                index=i, address=addr, found=True,
                address_label=geocoded.address_label,
                auto_city_key=city_resolution.region_candidates[0],
                region=city_resolution.region,
            ))
        elif city_resolution.region:
            # Region deduite mais aucun modele dedie -> generique, pas ambigu
            results.append(CityPreResolution(
                index=i, address=addr, found=True,
                address_label=geocoded.address_label,
                region=city_resolution.region,
            ))
        else:
            # Rien d'exploitable (needs_user_region) -> ambigu, propose TOUS
            # les modeles disponibles (pas de region pour restreindre le choix)
            results.append(CityPreResolution(
                index=i, address=addr, found=True,
                address_label=geocoded.address_label,
                candidates=sorted(CITY_MODELS.keys()),
                is_ambiguous=True,
            ))

    return results


@dataclass
class AddressResult:
    address_input:   str
    found:            bool = False
    address_label:    Optional[str] = None
    lat:              Optional[float] = None
    lon:              Optional[float] = None
    city_key_used:    Optional[str] = None
    ambiguous_region: bool = False           # True si choix auto parmi plusieurs candidats region
    region_candidates: list = field(default_factory=list)
    confidence:       Optional[str] = None   # "high" / "medium" / "none"
    distance_m:       Optional[float] = None
    albedo:           Optional[float] = None
    albedo_brut:      Optional[float] = None  # avant calibration ombre, cf. shadow_calibration.py
    shadow_fraction:  Optional[float] = None  # fraction du patch detectee en ombre
    material:         Optional[str] = None   # cf. materials.classify_material -- classifieur ML (v3.0+)
    material_confidence: Optional[str] = None  # "ml_classifieur_v1_hgb" ou "invalide" (materials.py v3.0+)
    material_proba_max: Optional[float] = None  # cf. materials.MaterialResult.proba_max (classifieur ML)
    # cf. postprocess_plausibility.py -- cf. commentaire equivalent dans
    # zone_scan.BuildingResult (detection uniquement, jamais de correction
    # silencieuse d'albedo/material).
    plausibilite_materiau: Optional[str] = None
    albedo_ajuste_biblio: Optional[float] = None
    nodata_ratio:     Optional[float] = None  # cf. patch_extraction.extract_and_normalize_patch_with_nodata_ratio
    # Score de confiance composite (cf. confidence.py) -- PAS une marge
    # d'erreur calibree, une heuristique de transparence basee sur des
    # signaux internes (matching batiment, plausibilite materiau, nodata).
    confidence_score:   Optional[int] = None
    confidence_niveau:  Optional[str] = None   # "haute" / "moyenne" / "faible"
    confidence_raisons: Optional[str] = None   # raisons concatenees, lisibles
    checkpoint_hash:  Optional[str] = None    # cf. versioning.py -- tracabilite du modele utilise
    ortho_path:       Optional[Path] = None
    # Vignette RGB (PNG, en memoire) du patch decoupe -- cf. suivi projet,
    # points 1/5 (verification visuelle en lot). Stockee en bytes plutot
    # qu'un chemin car ortho_path pointe dans un TemporaryDirectory qui est
    # supprime avant que main.py ne puisse l'afficher en fin de traitement
    # par lot -- la vignette, elle, survit (quelques Ko en memoire).
    thumbnail_png:    Optional[bytes] = None
    error:            Optional[str] = None   # message d'erreur lisible, si echec a une etape


def process_address(
    address: str,
    tmp_dir: Path,
    city_key_override: Optional[str] = None,
    force_generic: bool = False,
) -> AddressResult:
    """Execute le pipeline complet pour une adresse et retourne un resultat structure.

    Args:
        address: Adresse en texte libre.
        tmp_dir: Dossier temporaire pour l'orthophoto telechargee.
        city_key_override: Force une cle ville precise (mode "adresse unique"
            avec choix manuel, ou mode lot "manuel" apres pre-scan).
        force_generic: Si True, force explicitement le fallback generique
            (l'utilisateur a choisi "generique" dans le pre-scan manuel),
            plutot que de laisser la resolution automatique determiner le
            fallback. Ignore si city_key_override est aussi fourni.

    Returns:
        AddressResult, avec `error` rempli des que le pipeline s'arrete avant
        la prediction (aucune exception ne remonte a l'appelant).
    """
    res = AddressResult(address_input=address)

    try:
        geocoded = geocode_address(address)
    except Exception as exc:
        res.error = f"Erreur reseau geocodage : {exc}"
        return res

    if not geocoded.found:
        res.error = "Adresse introuvable via l'API BAN."
        return res

    res.found = True
    res.address_label = geocoded.address_label
    res.lat = geocoded.lat
    res.lon = geocoded.lon

    if city_key_override:
        city_key = city_key_override
    elif force_generic:
        city_key = None
    else:
        city_resolution = resolve_city_key(geocoded)
        if city_resolution.city_key:
            city_key = city_resolution.city_key
        elif city_resolution.region_candidates:
            # Mode lot "automatique" (pas d'utilisateur pour trancher) : on
            # prend le premier candidat par convention, et on le signale.
            city_key = city_resolution.region_candidates[0]
            res.ambiguous_region = True
            res.region_candidates = city_resolution.region_candidates
        else:
            city_key = None  # -> fallback generique

    res.city_key_used = city_key if city_key else "generique"
    checkpoint_path = get_checkpoint_path(city_key) if city_key else GENERIC_FALLBACK_MODEL

    try:
        fetch_result = fetch_building_and_ortho(
            lat=res.lat, lon=res.lon, tmp_dir=tmp_dir, with_candidates=False
        )
    except Exception as exc:
        res.error = f"Echec recuperation IGN (WFS/WMS) : {exc}"
        return res

    match = fetch_result.match
    res.confidence = match.confidence
    res.distance_m = match.distance_m
    res.ortho_path = fetch_result.ortho_path

    if match.confidence == "none":
        res.error = "Aucune correspondance batiment fiable (filet de securite)."
        return res

    if fetch_result.ortho_path is None:
        res.error = "Echec telechargement orthophoto."
        return res

    raw_patch, patch, nodata_ratio = extract_raw_and_normalized_patch(
        fetch_result.ortho_path, match.centroid_x, match.centroid_y
    )
    res.nodata_ratio = nodata_ratio
    if patch is None:
        res.error = "Patch invalide (hors emprise ou trop de nodata)."
        return res

    # Vignette pour verification visuelle (mode lot -- cf. suivi projet).
    # best-effort : ne bloque jamais le reste du pipeline si l'encodage
    # echoue pour une raison quelconque.
    try:
        res.thumbnail_png = patch_to_png_bytes(raw_patch)
    except Exception:
        pass

    # Classification materiau (classifieur ML, cf. materials.py v3.0+) --
    # best-effort, jamais bloquant pour la prediction albedo.
    try:
        mat = classify_material(patch)
        res.material = mat.material
        res.material_confidence = mat.confidence
        res.material_proba_max = mat.proba_max
    except Exception:
        pass

    try:
        res.albedo = predict_albedo(patch, checkpoint_path)
    except Exception as exc:
        res.error = f"Echec inference : {exc}\n{traceback.format_exc(limit=1)}"

    # Calibration post-prediction du biais d'ombre (cf. shadow_calibration.py) :
    # corrige res.albedo AVANT la plausibilite materiau ci-dessous, pour que
    # celle-ci travaille sur un albedo deja debiaise de l'ombre plutot que
    # sur la valeur brute du modele. Opere uniquement sur raw_patch (jamais
    # renvoye au modele) -- best-effort, jamais bloquant.
    if res.albedo is not None:
        try:
            calib = calibrate_albedo(res.albedo, raw_patch, res.material)
            res.albedo_brut = calib.albedo_brut
            res.albedo = calib.albedo_corrige
            res.shadow_fraction = calib.shadow_fraction
        except Exception:
            pass

    # Plausibilite materiau (cf. postprocess_plausibility.py) : DETECTION
    # uniquement, calculee des que material+albedo sont disponibles tous les
    # deux. best-effort, jamais bloquant (comme la classification materiau
    # ci-dessus).
    if res.albedo is not None:
        try:
            res.plausibilite_materiau = classify_plausibility_single(res.material, res.albedo)
            res.albedo_ajuste_biblio = adjust_albedo_single(
                res.material, res.albedo, res.plausibilite_materiau,
                alpha=DEFAULT_ANCHOR_ALPHA,
            )
        except Exception:
            pass

        # Score de confiance composite (cf. confidence.py) -- best-effort,
        # jamais bloquant, comme les etapes de detection ci-dessus.
        try:
            conf = compute_confidence(
                plausibilite_materiau=res.plausibilite_materiau,
                building_match_confidence=res.confidence,
                nodata_ratio=res.nodata_ratio,
                city_key=city_key,
                building_lat=res.lat,
                building_lon=res.lon,
                material_proba_max=res.material_proba_max,
            )
            res.confidence_score = conf.score
            res.confidence_niveau = conf.niveau
            res.confidence_raisons = "; ".join(conf.raisons)
        except Exception:
            pass

    # Tracabilite : hash du checkpoint reellement utilise (cf. versioning.py).
    # best-effort -- n'empeche jamais de retourner un resultat si le fichier
    # est introuvable ou le hash echoue pour une raison quelconque.
    try:
        res.checkpoint_hash = build_run_metadata(
            checkpoint_path, res.city_key_used or "generique", mode="adresse_unique_ou_lot"
        ).checkpoint_hash
    except Exception:
        pass

    return res


def process_batch(
    addresses: list[str],
    overrides: Optional[dict[int, Optional[str]]] = None,
    progress_callback=None,
    max_workers: int = 6,
) -> list[AddressResult]:
    """Traite une liste d'adresses en parallele (threads).

    Chaque adresse fait des appels reseau independants (geocodage BAN, WFS/
    WMS-R IGN) -- I/O-bound, donc paralleliser avec des threads accelere
    nettement sans toucher a la logique metier (le GIL n'est de toute facon
    pas un probleme ici : l'essentiel du temps est passe en attente reseau,
    pas en calcul Python pur ; seule l'inference torch est CPU-bound, mais
    reste rapide par rapport aux appels reseau et le cache modele est
    thread-safe, cf. inference.get_predictor).

    Args:
        addresses: Liste d'adresses en texte libre (une par element).
        overrides: Mapping optionnel {index dans `addresses` (avant filtrage
            des lignes vides) -> cle ville forcee}. Utilise par le mode lot
            "manuel", ou l'utilisateur a tranche les cas ambigus lors du
            pre-scan (cf. pre_resolve_addresses). Une valeur None dans ce
            dict force explicitement le fallback generique (choix
            "generique" fait par l'utilisateur), a distinguer de l'absence
            de cle dans le dict (= pas d'override, resolution automatique).
        progress_callback: Fonction optionnelle appelee (i, total, address)
            apres chaque adresse traitee, pour mettre a jour une barre de
            progression Streamlit. Appelee depuis plusieurs threads : le
            compteur `i` est protege par un lock, mais l'ORDRE d'arrivee des
            callbacks ne correspond plus a l'ordre des adresses (normal en
            parallele) -- ne pas s'y fier pour autre chose qu'une barre de
            progression globale.
        max_workers: Nombre de threads concurrents (defaut 6 -- au-dela, les
            API publiques BAN/IGN peuvent rate-limiter ou timeout davantage
            sans gain de vitesse supplementaire).

    Returns:
        Liste d'AddressResult, DANS LE MEME ORDRE que `addresses` (lignes
        vides ignorees) -- l'ordre est garanti malgre le traitement
        parallele (resultats collectes par index puis retries dans l'ordre).
    """
    overrides = overrides or {}

    # (index original dans `addresses`, adresse) pour les lignes non vides
    indexed_addresses = [(i, a.strip()) for i, a in enumerate(addresses) if a.strip()]
    total = len(indexed_addresses)

    results_by_index: dict[int, AddressResult] = {}
    processed = 0
    progress_lock = threading.Lock()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        def _process_one(item: tuple[int, str]) -> tuple[int, AddressResult]:
            i, addr = item
            has_override = i in overrides
            override_value = overrides.get(i)
            result = process_address(
                addr, tmp_dir,
                city_key_override=override_value if has_override else None,
                force_generic=has_override and override_value is None,
            )
            return i, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_one, item) for item in indexed_addresses]
            for future in as_completed(futures):
                i, result = future.result()
                results_by_index[i] = result
                if progress_callback:
                    with progress_lock:
                        processed += 1
                        progress_callback(processed, total, result.address_input)

    # Retrie dans l'ordre original des adresses (indispensable : l'ordre
    # d'arrivee des threads est non deterministe).
    ordered_indices = [i for i, _ in indexed_addresses]
    return [results_by_index[i] for i in ordered_indices]
