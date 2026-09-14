# main.py
from __future__ import annotations

import base64
import json
import logging
from PIL import Image
import io
import csv
import io
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger("albedonet.ortho")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import GENERIC_FALLBACK_MODEL, get_checkpoint_path
from geocoding import geocode_address, resolve_city_key
from ign_fetch import fetch_building_and_ortho
from inference import predict_albedo
from materials import classify_material, material_confidence_label
from patch_extraction import extract_raw_and_normalized_patch
from pipeline import process_batch
from shadow_calibration import calibrate_albedo
from zone_scan import scan_zone, scan_zone_events, zone_scan_result_to_json

import checkpoint_integrity

app = FastAPI(title="AlbedoNet API")


@app.on_event("startup")
def _verify_checkpoint_integrity() -> None:
    # Refus au demarrage si un checkpoint present ne correspond pas au hash
    # attendu (corruption / mauvaise version). Un checkpoint absent est gere
    # ailleurs (cf. test_city_model_smoke.py) -- ce n'est pas la meme erreur.
    checkpoint_integrity.verify()

# Origines autorisées : liste explicite en prod (via env var), fallback dev
# permissif uniquement en local. Ne jamais repasser à ["*"] en prod.
import os
_allowed_origins_env = os.environ.get("ALBEDONET_ALLOWED_ORIGINS", "")
ALLOWED_ORIGINS = (
    [o.strip() for o in _allowed_origins_env.split(",") if o.strip()]
    if _allowed_origins_env
    else ["http://localhost:8000", "http://127.0.0.1:8000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class BatchRequest(BaseModel):
    addresses: list[str]


def _resolve_city_key_auto(geocode_result) -> Optional[str]:
    city_resolution = resolve_city_key(geocode_result)
    if city_resolution.city_key:
        return city_resolution.city_key
    if city_resolution.region_candidates:
        return city_resolution.region_candidates[0]
    return None


def _ortho_to_data_url(ortho_path: Optional[Path]) -> Optional[str]:
    if ortho_path is None:
        return None
    try:
        raw = Path(ortho_path).read_bytes()
    except OSError:
        logger.exception("Echec de lecture de l'orthophoto (path=%s)", ortho_path)
        return None
    return "data:image/tiff;base64," + base64.b64encode(raw).decode("ascii")



def _resize_thumbnail(thumb_bytes, size=(256, 256)):
    """Redimensionne une vignette PNG à la taille spécifiée."""
    try:
        img = Image.open(io.BytesIO(thumb_bytes))
        img = img.resize(size, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()
    except Exception:
        return thumb_bytes  # fallback

@app.get("/api/single")
def api_single(address: str):
    if not address.strip():
        raise HTTPException(400, "Adresse vide.")

    geocode_result = geocode_address(address)
    if not geocode_result.found:
        return {"found": False, "error": "Adresse introuvable via l'API BAN."}

    city_key = _resolve_city_key_auto(geocode_result)
    checkpoint_path = get_checkpoint_path(city_key) if city_key else GENERIC_FALLBACK_MODEL
    model_label = city_key if city_key else "generique (paris)"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        try:
            fetch_result = fetch_building_and_ortho(
                lat=geocode_result.lat, lon=geocode_result.lon, tmp_dir=tmp_dir,
                with_candidates=False,
            )
        except Exception as exc:
            logger.exception("Echec de la recuperation IGN (WFS/WMS) pour address=%r", address)
            return {
                "found": True, "address_label": geocode_result.address_label,
                "lat": geocode_result.lat, "lon": geocode_result.lon,
                "albedo": None,
                "error": f"Echec de la recuperation IGN (WFS/WMS) : {exc}",
            }

        match = fetch_result.match
        base = {
            "found": True,
            "address_label": geocode_result.address_label,
            "lat": geocode_result.lat,
            "lon": geocode_result.lon,
            "confidence": match.confidence,
            "distance_m": match.distance_m,
            "model_used": model_label,
        }

        if match.confidence == "none":
            base["albedo"] = None
            base["error"] = "Aucune correspondance batiment fiable."
            base["ortho_image"] = _ortho_to_data_url(fetch_result.ortho_path)
            return base

        if fetch_result.ortho_path is None:
            base["albedo"] = None
            base["error"] = "Echec du telechargement de l'orthophoto IGN."
            return base

        eff_x = match.centroid_x if match.centroid_x is not None else fetch_result.point_l93.x
        eff_y = match.centroid_y if match.centroid_y is not None else fetch_result.point_l93.y

        raw_patch, patch, _ratio = extract_raw_and_normalized_patch(
            fetch_result.ortho_path, eff_x, eff_y
        )
        if patch is None:
            base["albedo"] = None
            base["error"] = "Patch invalide (hors emprise orthophoto)."
            return base

        try:
            albedo = predict_albedo(patch, checkpoint_path)
        except Exception as exc:
            logger.exception(
                "Echec de l'inference (checkpoint=%s, model_label=%s, address=%r)",
                checkpoint_path, model_label, address,
            )
            base["albedo"] = None
            base["error"] = f"Echec de l'inference : {exc}"
            return base

        material = None
        material_proba_max = None
        try:
            mat_result = classify_material(patch)
            material = mat_result.material
            material_proba_max = mat_result.proba_max
        except Exception:
            logger.exception("Echec de la classification materiau")

        # --- Calibration post-prediction du biais d'ombre (cf. shadow_calibration.py) ---
        # Applique sur l'albedo predit, jamais sur le patch envoye au modele.
        try:
            calib = calibrate_albedo(albedo, raw_patch, material)
            base["albedo"] = calib.albedo_corrige
            base["albedo_brut"] = calib.albedo_brut
            base["shadow_fraction"] = calib.shadow_fraction
        except Exception:
            logger.exception(
                "Echec de la calibration d'ombre (material=%s, address=%r) -- fallback sur l'albedo brut",
                material, address,
            )
            base["albedo"] = albedo
            base["albedo_brut"] = albedo
            base["shadow_fraction"] = None

        base["material"] = material
        base["material_proba_max"] = material_proba_max
        # Transparence sur la fiabilite de l'estimation materiau (cf. discussion
        # "etre pro" du 22/07/2026) -- pas seulement en docstring interne, un
        # client externe voit directement la probabilite du classifieur pour
        # ce materiau.
        base["material_is_heuristic"] = False  # classifieur ML entraine depuis le 22/07/2026, cf. materials.py v3.0
        base["material_confidence_label"] = (
            material_confidence_label(material, material_proba_max) if material else None
        )
        base["ortho_image"] = _ortho_to_data_url(fetch_result.ortho_path)
        base["error"] = None
        return base


@app.post("/api/batch")
def api_batch(req: BatchRequest):
    if not req.addresses:
        raise HTTPException(400, "Liste d'adresses vide.")
    results = process_batch(req.addresses)
    return {
        "results": [
            {
                "address_input": r.address_input,
                "address_label": r.address_label,
                "lat": r.lat,
                "lon": r.lon,
                "model_used": r.city_key_used,
                "ambiguous_region": bool(r.ambiguous_region),
                "confidence": r.confidence,
                "distance_m": r.distance_m,
                "albedo": r.albedo,
                "material": r.material,
                "material_proba_max": r.material_proba_max,
                "error": r.error,
                "thumbnail_png": base64.b64encode(_resize_thumbnail(r.thumbnail_png, (128,128))).decode("ascii") if r.thumbnail_png else None,
            }
            for r in results
        ]
    }


@app.post("/api/batch.csv")
def api_batch_csv(req: BatchRequest):
    if not req.addresses:
        raise HTTPException(400, "Liste d'adresses vide.")
    results = process_batch(req.addresses)
    buf = io.StringIO()
    fieldnames = [
        "adresse_saisie", "adresse_geocodee", "lat", "lon", "modele_utilise",
        "region_ambigue", "confiance_batiment", "distance_batiment_m",
        "albedo", "albedo_brut", "shadow_fraction", "materiau_estime",
        "material_proba_max", "erreur",
    ]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
        writer.writerow({
            "adresse_saisie": r.address_input,
            "adresse_geocodee": r.address_label or "",
            "lat": r.lat, "lon": r.lon,
            "modele_utilise": r.city_key_used or "",
            "region_ambigue": "oui" if r.ambiguous_region else "",
            "confiance_batiment": r.confidence or "",
            "distance_batiment_m": round(r.distance_m, 1) if r.distance_m is not None else "",
            "albedo": round(r.albedo, 3) if r.albedo is not None else "",
            # albedo_brut/shadow_fraction : necessaires pour la calibration
            # empirique de shadow_calibration.SHADOW_CORRECTION_FACTOR (cf.
            # validate_thresholds.py) -- absents avant ce changement, ce qui
            # rendait cette calibration impossible meme avec des scans reels.
            "albedo_brut": round(r.albedo_brut, 3) if getattr(r, "albedo_brut", None) is not None else "",
            "shadow_fraction": round(r.shadow_fraction, 4) if getattr(r, "shadow_fraction", None) is not None else "",
            "materiau_estime": r.material or "",
            "material_proba_max": round(r.material_proba_max, 3) if getattr(r, "material_proba_max", None) is not None else "",
            "erreur": r.error or "",
        })
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=albedo_resultats_lot.csv"},
    )


@app.get("/api/zone")
def api_zone(address: str, radius_m: float = 250.0):
    if not address.strip():
        raise HTTPException(400, "Adresse vide.")
    zone_result = scan_zone(address.strip(), radius_m)
    if zone_result.error and not zone_result.buildings:
        return {"error": zone_result.error}

    stats = zone_result.stats
    return {
        "error": None,
        "n_buildings": zone_result.n_buildings,
        "n_tiles": zone_result.n_tiles,
        "model_used": zone_result.city_key_used,
        "center_lat": zone_result.center_lat,
        "center_lon": zone_result.center_lon,
        "stats": None if stats is None else {
            "mean": stats.mean, "median": stats.median, "std": stats.std,
            "area_weighted_mean": stats.area_weighted_mean,
            "min": stats.min, "max": stats.max, "n": stats.n,
            "histogram": stats.histogram,
        },
        "n_heat_suspects": sum(1 for b in zone_result.buildings if b.heat_island_suspect),
        "n_cool_suspects": sum(1 for b in zone_result.buildings if b.cool_island_suspect),
        "suspects": [
            {
                "lat": b.centroid_lat,
                "lon": b.centroid_lon,
                "albedo": b.albedo,
                "material": b.material or "",
                "material_proba_max": getattr(b, "material_proba_max", None),
                "heat_suspect": b.heat_island_suspect,
                "cool_suspect": b.cool_island_suspect,
                "thumbnail_png": base64.b64encode(_resize_thumbnail(b.thumbnail_png, (256,256))).decode("ascii") if b.thumbnail_png else None,
            }
            for b in zone_result.buildings
            if b.heat_island_suspect or b.cool_island_suspect
        ],
    }


@app.get("/api/zone.csv")
def api_zone_csv(address: str, radius_m: float = 250.0):
    if not address.strip():
        raise HTTPException(400, "Adresse vide.")
    zone_result = scan_zone(address.strip(), radius_m)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "building_id", "centroid_lat", "centroid_lon", "area_m2", "albedo",
        "albedo_brut", "shadow_fraction", "materiau_estime", "material_proba_max",
        "ilot_chaleur_suspect", "ilot_fraicheur_suspect", "erreur",
    ])
    for b in zone_result.buildings:
        writer.writerow([
            b.building_id, b.centroid_lat, b.centroid_lon, round(b.area_m2, 1),
            round(b.albedo, 4) if b.albedo is not None else "",
            # albedo_brut/shadow_fraction : cf. commentaire equivalent dans
            # api_batch_csv -- necessaires pour valider empiriquement
            # shadow_calibration.SHADOW_CORRECTION_FACTOR.
            round(b.albedo_brut, 4) if getattr(b, "albedo_brut", None) is not None else "",
            round(b.shadow_fraction, 4) if getattr(b, "shadow_fraction", None) is not None else "",
            b.material or "",
            round(b.material_proba_max, 3) if getattr(b, "material_proba_max", None) is not None else "",
            "oui" if b.heat_island_suspect else "",
            "oui" if b.cool_island_suspect else "",
            b.error or "",
        ])
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=albedo_resultats_zone.csv"},
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/zone/stream")
def api_zone_stream(address: str, radius_m: float = 250.0):
    if not address.strip():
        raise HTTPException(400, "Adresse vide.")

    def generate():
        for event in scan_zone_events(address.strip(), radius_m):
            if event.get("done"):
                if "error" in event:
                    yield _sse("error_event", {"error": event["error"]})
                else:
                    yield _sse("done", zone_scan_result_to_json(event["result"]))
            else:
                yield _sse("progress", event)

    return StreamingResponse(generate(), media_type="text/event-stream")


# Sert le dossier static (contient index.html)
app.mount("/", StaticFiles(directory="static", html=True), name="static")