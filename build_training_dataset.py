"""Fusion OSM + annotations manuelles, extraction de patches, calcul de features,
export du dataset d'entraînement.

Points clés vs. la version précédente :
- dédoublonnage spatial entre sources OSM et manuelles (évite la fuite train/test) ;
- group_id (cluster spatial ~50m) attaché à chaque échantillon, pour permettre un
  split GroupShuffleSplit en aval (cf. train_material_classifier.py) ;
- feature_set_version enregistrée dans le dataset pour détecter toute incompatibilité
  avec un modèle entraîné sur un schéma de features différent ;
- logging structuré + reprise possible (les échecs individuels n'interrompent pas le run) ;
- PARALLÉLISATION (ThreadPoolExecutor, I/O-bound) avec :
    * détection auto de throttling (si le taux d'erreurs réseau grimpe, on te le dit
      dans les logs plutôt que de foncer aveuglément) ;
    * retry avec backoff exponentiel sur erreurs réseau transitoires, pour pouvoir
      tenir une concurrence plus haute sans qu'un pic ponctuel ne fasse échouer
      des lignes qui auraient réussi en réessayant.
"""
from __future__ import annotations

import logging
import random
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# features.py vit dans training/ alors que ign_fetch.py / patch_extraction.py
# sont a la racine du projet (layout mixte constate le 23/07/2026) -- on
# ajoute training/ au sys.path pour que l'import fonctionne quel que soit le
# repertoire depuis lequel ce script est lance, sans deplacer les modules.
_TRAINING_DIR = Path(__file__).resolve().parent / "training"
if _TRAINING_DIR.is_dir() and str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from features import FEATURE_NAMES, FEATURE_SET_VERSION, InvalidPatchError, extract_features
from ign_fetch import build_http_session, fetch_building_and_ortho
from patch_extraction import extract_and_normalize_patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OSM_CSV = Path("datasets/osm_materials.csv")
ANNOT_CSV = Path("datasets/annotations_manuelles.csv")
OUTPUT_PARQUET = Path("datasets/materials_v1.parquet")

# Nombre de requêtes IGN en parallèle. On part à 16 (au lieu de 8) : le WMS-R public
# IGN n'a pas de rate limit documenté et encaisse en général bien ce niveau de charge.
# Le run surveille lui-même le taux d'erreurs réseau (cf. _NetworkErrorMonitor) et te
# préviendra dans les logs s'il détecte un throttling -- baisse alors cette valeur.
MAX_WORKERS = 16

# Retry sur erreurs réseau transitoires (timeout, connection reset, 429, 503...).
MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 0.5  # backoff exponentiel : 0.5s, 1s, 2s (+ jitter)

# Classes cibles validées en Phase 0 (cf. material_classifier_spec.md).
# "indetermine" n'en fait PAS partie : c'est une option de saisie pour
# l'annotateur humain (annotation_tool.py) quand un bâtiment est ambigu à
# l'œil, mais ce n'est pas une classe de sortie du modèle -- ces lignes
# doivent être exclues du dataset d'entraînement, pas apprises comme une
# 5e classe.
ALLOWED_MATERIALS = {"zinc", "ardoise", "tuile_terre_cuite", "beton"}

# Tolérance de dédoublonnage spatial : deux points à moins de ~10m sont considérés
# comme le même bâtiment. 1e-4 degré ≈ 11m en latitude.
DEDUP_TOLERANCE_DEG = 1e-4

# Taille de cluster utilisée pour le group_id du split (évite la fuite spatiale) :
# ~500m, suffisant pour séparer des toits mitoyens du même pâté de maisons.
GROUP_CLUSTER_DEG = 0.005

# Erreurs considérées comme transitoires -> retry. Tout le reste (InvalidPatchError,
# KeyError sur données malformées, etc.) est fatal pour la ligne et n'est PAS retenté.
_TRANSIENT_ERROR_MARKERS = ("timeout", "connection", "reset", "429", "503", "502", "temporarily")


class _NetworkErrorMonitor:
    """Compte les erreurs probablement liées à du throttling / rate limiting, pour
    prévenir l'utilisateur plutôt que de le laisser découvrir un taux d'échec élevé
    à la fin d'un run de 2400 lignes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._transient_errors = 0
        self._total_attempts = 0

    def record_attempt(self) -> None:
        with self._lock:
            self._total_attempts += 1

    def record_transient_error(self) -> None:
        with self._lock:
            self._transient_errors += 1

    def maybe_warn(self, n_done: int, total: int) -> None:
        with self._lock:
            errors, attempts = self._transient_errors, self._total_attempts
        if attempts < 50:
            return
        rate = errors / attempts
        if rate > 0.15:
            logger.warning(
                "Taux d'erreurs réseau transitoires élevé (%.1f%% sur %d tentatives) -- "
                "possible throttling IGN. Envisage de baisser MAX_WORKERS (actuellement %d).",
                100 * rate, attempts, MAX_WORKERS,
            )


_monitor = _NetworkErrorMonitor()

# Une requests.Session par thread (keep-alive/pool de connexions), au lieu
# d'une session neuve a chaque ligne (cf. fetch_building_and_ortho -- avant
# ce fix, une session ephemere etait creee a chaque appel, ce qui renegociait
# TLS a chaque requete meme vers le meme host). Les Session ne sont pas
# garanties thread-safe pour un usage concurrent partage -- une session par
# thread (et non une session globale) est le pattern recommande par requests.
_thread_local = threading.local()


def _get_thread_session():
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = build_http_session()
        _thread_local.session = session
    return session



def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS)


def _load_source_rows() -> list[dict]:
    rows: list[dict] = []
    if OSM_CSV.exists():
        osm_df = pd.read_csv(OSM_CSV)
        for _, r in osm_df.iterrows():
            rows.append({"source": "osm", "city": r["city_key"], "lat": float(r["lat"]), "lon": float(r["lon"]), "material": r["material"]})
    else:
        logger.warning("%s introuvable, aucune donnée OSM chargée.", OSM_CSV)

    if ANNOT_CSV.exists():
        annot_df = pd.read_csv(ANNOT_CSV)
        for _, r in annot_df.iterrows():
            rows.append({"source": "manual", "city": "unknown", "lat": float(r["lat"]), "lon": float(r["lon"]), "material": r["material"]})
    else:
        logger.warning("%s introuvable, aucune annotation manuelle chargée.", ANNOT_CSV)

    return rows


def _dedup_spatial(rows: list[dict]) -> list[dict]:
    """Supprime les doublons cross-source (même bâtiment annoté à la fois via OSM et
    manuellement). En cas de conflit, priorité à l'annotation manuelle (jugée plus fiable).
    """
    rows_sorted = sorted(rows, key=lambda r: 0 if r["source"] == "manual" else 1)
    kept: list[dict] = []
    for row in rows_sorted:
        is_dup = any(
            abs(row["lat"] - k["lat"]) < DEDUP_TOLERANCE_DEG and abs(row["lon"] - k["lon"]) < DEDUP_TOLERANCE_DEG
            for k in kept
        )
        if is_dup:
            continue
        kept.append(row)
    n_removed = len(rows) - len(kept)
    if n_removed:
        logger.info("Dédoublonnage spatial : %d échantillons supprimés (doublons cross-source).", n_removed)
    return kept


def _group_id(lat: float, lon: float) -> str:
    """Identifiant de cluster spatial grossier, pour un split train/val/test qui ne
    fuit pas d'information entre bâtiments géographiquement proches."""
    lat_bin = round(lat / GROUP_CLUSTER_DEG)
    lon_bin = round(lon / GROUP_CLUSTER_DEG)
    return f"{lat_bin}_{lon_bin}"


def _filter_allowed_materials(rows: list[dict]) -> list[dict]:
    kept = [r for r in rows if r["material"] in ALLOWED_MATERIALS]
    n_removed = len(rows) - len(kept)
    if n_removed:
        rejected = sorted({r["material"] for r in rows if r["material"] not in ALLOWED_MATERIALS})
        logger.info(
            "%d échantillons exclus (matériau hors classes cibles %s) : valeurs rejetées = %s",
            n_removed, sorted(ALLOWED_MATERIALS), rejected,
        )
    return kept


def _fetch_with_retry(lat: float, lon: float, tmp_dir: Path):
    """fetch_building_and_ortho avec retry + backoff exponentiel + jitter sur erreurs
    transitoires uniquement. Les erreurs non-transitoires (données invalides, bug logique)
    remontent immédiatement sans attendre -- pas la peine de retenter une erreur déterministe.
    """
    last_exc: Exception | None = None
    session = _get_thread_session()
    for attempt in range(MAX_RETRIES + 1):
        _monitor.record_attempt()
        try:
            return fetch_building_and_ortho(lat, lon, tmp_dir, with_candidates=False, session=session)
        except Exception as exc:  # noqa: BLE001 - on filtre juste après
            if not _is_transient(exc):
                raise
            _monitor.record_transient_error()
            last_exc = exc
            if attempt < MAX_RETRIES:
                delay = RETRY_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 0.3)
                time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _process_row(row: dict) -> dict | None:
    """Traite une ligne : fetch IGN + patch + features. Retourne None en cas d'échec
    (chaque échec est loggé mais n'interrompt pas le run global). Thread-safe : chaque
    appel crée son propre TemporaryDirectory, aucun état partagé muté hors du monitor.
    """
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            fetch_result = _fetch_with_retry(row["lat"], row["lon"], tmp_dir)
            if fetch_result.match.confidence == "none":
                logger.debug("Aucun bâtiment matché pour %s, échantillon ignoré.", row)
                return None
            eff_x = fetch_result.match.centroid_x or fetch_result.point_l93.x
            eff_y = fetch_result.match.centroid_y or fetch_result.point_l93.y
            patch = extract_and_normalize_patch(fetch_result.ortho_path, eff_x, eff_y)
            features = extract_features(patch)  # lève InvalidPatchError si patch invalide
            return {
                "features": features,
                "material": row["material"],
                "city": row["city"],
                "source": row["source"],
                "group_id": _group_id(row["lat"], row["lon"]),
                "lat": row["lat"],
                "lon": row["lon"],
            }
    except InvalidPatchError as exc:
        logger.warning("Patch invalide pour %s : %s", row, exc)
        return None
    except Exception:
        logger.exception("Échec inattendu pour %s (après retries si applicable)", row)
        return None


def build_dataset() -> None:
    rows = _filter_allowed_materials(_load_source_rows())
    rows = _dedup_spatial(rows)
    if not rows:
        logger.error("Aucune donnée source trouvée (ou tout exclu par le filtre de classes), arrêt.")
        return

    all_data: list[dict] = []
    n_failed = 0
    n_done = 0
    lock = threading.Lock()
    total = len(rows)
    t0 = time.monotonic()

    logger.info("Lancement du traitement parallèle (%d workers) sur %d échantillons.", MAX_WORKERS, total)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_process_row, row): row for row in rows}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                n_done += 1
                if result is None:
                    n_failed += 1
                else:
                    all_data.append(result)
                if n_done % 50 == 0 or n_done == total:
                    elapsed = time.monotonic() - t0
                    rate = n_done / elapsed if elapsed > 0 else 0.0
                    eta_s = (total - n_done) / rate if rate > 0 else float("nan")
                    logger.info(
                        "Progression : %d/%d traités (%d échecs) -- %.1f lignes/s, ETA %.0fs",
                        n_done, total, n_failed, rate, eta_s,
                    )
                    _monitor.maybe_warn(n_done, total)

    if not all_data:
        logger.error("Aucun patch extrait avec succès, arrêt.")
        return

    logger.info("%d échantillons extraits, %d échecs (%.1f%%) en %.0fs",
                len(all_data), n_failed, 100 * n_failed / total, time.monotonic() - t0)

    feature_matrix = np.stack([d["features"] for d in all_data])
    df = pd.DataFrame(feature_matrix, columns=FEATURE_NAMES)
    df["material"] = [d["material"] for d in all_data]
    df["city"] = [d["city"] for d in all_data]
    df["source"] = [d["source"] for d in all_data]
    df["group_id"] = [d["group_id"] for d in all_data]
    df["lat"] = [d["lat"] for d in all_data]
    df["lon"] = [d["lon"] for d in all_data]
    df.attrs["feature_set_version"] = FEATURE_SET_VERSION

    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET)
    # pandas n'enregistre pas df.attrs dans le parquet -> on stocke la version à côté.
    (OUTPUT_PARQUET.parent / f"{OUTPUT_PARQUET.stem}.feature_version.txt").write_text(FEATURE_SET_VERSION)

    logger.info("Dataset sauvegardé : %s (%d échantillons, feature_set_version=%s)",
                OUTPUT_PARQUET, len(df), FEATURE_SET_VERSION)

    class_counts = df["material"].value_counts()
    logger.info("Distribution des classes :\n%s", class_counts.to_string())
    rare_classes = class_counts[class_counts < 20]
    if not rare_classes.empty:
        logger.warning("Classes sous-représentées (<20 échantillons) : %s", list(rare_classes.index))


if __name__ == "__main__":
    build_dataset()