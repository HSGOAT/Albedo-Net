"""
zone_scan/serialization.py — AlbedoNet App
============================================
1. scan_zone_events : pont thread -> generateur, pour brancher scan_zone()
   (synchrone) sur un consommateur "au fil de l'eau" comme une route en
   Server-Sent Events.
2. zone_scan_result_to_json : serialisation d'un ZoneScanResult dans le
   format JSON attendu par le frontend (memes cles que l'ancienne route
   /api/zone).

Extrait de l'ancien zone_scan.py (Tier 3, decoupage en sous-modules) --
aucun changement de comportement, seulement de localisation.
"""

from __future__ import annotations

import base64
import queue
import threading
from typing import Optional

from .models import ZoneScanResult
from .progress import _STAGE_LABELS, _progress_percent
from .scan import scan_zone

# ──────────────────────────────────────────────────────────────────────────
# Variante generatrice (pont thread -> generateur), pour un consommateur
# "au fil de l'eau" comme une route Flask en Server-Sent Events.
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
