#!/usr/bin/env python3
"""
patch_extraction.py - Decoupe patch 64x64 + normalisation min-max pour l'app.

Adapte de extract_patch() dans 01_finetune_patch_extractor.py, avec la meme
logique de fenetre rasterio centree + clamp aux bords + resize LANCZOS.

Difference vs le script d'entrainement :
  - 01_finetune_patch_extractor.py normalise en /255.0 (l'ortho source y est
    deja en uint8 propre, sans nodata a l'interieur du patch car filtre au
    niveau bati/zone en amont).
  - Ici (app temps reel), on suit plutot la meme normalisation min-max PAR
    BANDE que 00_patch_extractor.py (extract_patch_fast), car la tuile
    orthophoto telechargee pour une seule adresse peut contenir du nodata
    en bordure (facade de tuile WMS) -- le min-max sur pixels valides gere
    ce cas, la simple division par 255 non.
  - Rappel de l'ordre de normalisation (note technique du suivi projet) :
    min-max par bande (pixels valides) PUIS z-score via BandNormalizer
    (stats stockees dans le checkpoint) -- cette derniere etape reste faite
    par inference.py, pas ici.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rasterio
import rasterio.windows
from rasterio.windows import from_bounds
from PIL import Image

logger = logging.getLogger("albedo.app.patch_extraction")

PATCH_SIZE: int = 64
N_BANDS: int = 3  # RGB uniquement -- pas les bandes Sentinel-2 (cf. notes techniques)
MAX_NODATA_RATIO: float = 0.10  # coherent avec 00_patch_extractor.py (extract_patch_fast)


def attenuate_shadows(rgb_patch: np.ndarray, shadow_threshold: int = 60, boost_factor: float = 1.4) -> np.ndarray:
    """
    Corrige les ombres dans un patch RGB (uint8, (3, H, W)) en eclaircissant
    les pixels dont la valeur (HSV) est inferieure a shadow_threshold.
    """
    img = np.moveaxis(rgb_patch, 0, -1).astype(np.uint8)  # (H, W, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)

    shadow_mask = v < shadow_threshold
    v[shadow_mask] = np.clip(v[shadow_mask] * boost_factor, 0, 255)

    hsv[:, :, 2] = v.astype(np.uint8)
    corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return np.moveaxis(corrected, -1, 0)


def extract_patch(
    ortho_path: Path,
    centroid_x: float,
    centroid_y: float,
    patch_size: int = PATCH_SIZE,
    resolution: float = 0.20,
) -> Optional[np.ndarray]:
    """Decoupe un patch RGB centre sur un centroide, avec gestion des bords."""
    with rasterio.open(ortho_path) as src:
        if src.count < N_BANDS:
            logger.error(
                "Ortho a %d bande(s), %d attendues -- fichier invalide.",
                src.count, N_BANDS,
            )
            return None

        half = (patch_size * resolution) / 2.0
        win = from_bounds(
            left=centroid_x - half,
            bottom=centroid_y - half,
            right=centroid_x + half,
            top=centroid_y + half,
            transform=src.transform,
        ).round_lengths()

        try:
            win_clamped = win.intersection(
                rasterio.windows.Window(0, 0, src.width, src.height)
            )
            if win_clamped.width == 0 or win_clamped.height == 0:
                logger.warning("Centroide hors emprise de l'orthophoto telechargee.")
                return None
        except rasterio.errors.WindowError:
            logger.warning("Fenetre de decoupe hors emprise (WindowError).")
            return None

        data = src.read(window=win_clamped)[:N_BANDS]  # (3, H, W)

        if data.shape[1:] != (patch_size, patch_size):
            channels = []
            for c in range(data.shape[0]):
                img = Image.fromarray(data[c]).resize(
                    (patch_size, patch_size), Image.LANCZOS
                )
                channels.append(np.array(img))
            data = np.stack(channels)

    return data


def normalize_minmax_per_band(
    patch: np.ndarray,
    max_nodata_ratio: float = MAX_NODATA_RATIO,
) -> Optional[np.ndarray]:
    """Normalisation min-max vectorisee par bande, sur les pixels valides."""
    result = _normalize_minmax_per_band_with_ratio(patch, max_nodata_ratio)
    if result is None:
        return None
    data, _ratio = result
    return data


def _normalize_minmax_per_band_with_ratio(
    patch: np.ndarray,
    max_nodata_ratio: float = MAX_NODATA_RATIO,
) -> Optional[tuple[np.ndarray, float]]:
    """Coeur de normalize_minmax_per_band(), exposant le ratio nodata."""
    data = patch.astype(np.float32)
    nodata_mask = (data == 0).all(axis=0)
    ratio = float(nodata_mask.mean())

    if ratio > max_nodata_ratio:
        logger.warning(
            "Patch rejete : %.1f%% de nodata (seuil %.1f%%).",
            100 * ratio, 100 * max_nodata_ratio,
        )
        return None

    valid_mask = ~nodata_mask
    for b in range(data.shape[0]):
        band = data[b]
        valid_vals = band[valid_mask]
        if valid_vals.size == 0:
            return None
        bmin = valid_vals.min()
        bmax = valid_vals.max()
        rng = bmax - bmin
        if rng < 1e-6:
            data[b] = 0.0
        else:
            data[b] = np.clip((band - bmin) / rng, 0.0, 1.0)

    data[:, nodata_mask] = 0.0
    return data, ratio


def extract_and_normalize_patch(
    ortho_path: Path,
    centroid_x: float,
    centroid_y: float,
    patch_size: int = PATCH_SIZE,
    resolution: float = 0.20,
) -> Optional[np.ndarray]:
    """Point d'entree unique pour app.py : decoupe + normalisation min-max."""
    raw_patch = extract_patch(ortho_path, centroid_x, centroid_y, patch_size, resolution)
    if raw_patch is None:
        return None
    return normalize_minmax_per_band(raw_patch)


def extract_and_normalize_patch_with_nodata_ratio(
    ortho_path: Path,
    centroid_x: float,
    centroid_y: float,
    patch_size: int = PATCH_SIZE,
    resolution: float = 0.20,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """Variante exposant le ratio nodata."""
    raw_patch = extract_patch(ortho_path, centroid_x, centroid_y, patch_size, resolution)
    if raw_patch is None:
        return None, None
    result = _normalize_minmax_per_band_with_ratio(raw_patch)
    if result is None:
        return None, None
    return result


def extract_raw_and_normalized_patch(
    ortho_path: Path,
    centroid_x: float,
    centroid_y: float,
    patch_size: int = PATCH_SIZE,
    resolution: float = 0.20,
    max_nodata_ratio: float = MAX_NODATA_RATIO,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
    """Variante retournant le patch RAW + normalise + ratio nodata.

    NOTE (correctif biais ombre, cf. suivi projet) : attenuate_shadows() a
    ete retire de ce chemin. La fonction modifiait le patch RAW envoye au
    modele (boost du canal V en HSV) uniquement cote app, alors que
    00_patch_extractor.py et 01_finetune_patch_extractor.py (entrainement)
    ne l'appliquent PAS. Ca creait un decalage train/inference : le modele
    n'a jamais vu, pendant l'entrainement, de patches avec ce type de
    correction gamma-like sur les pixels sombres -- ses predictions dans
    cette zone du domaine d'entree n'etaient donc pas fiables.
    La correction du biais d'ombre se fait desormais en POST-PREDICTION,
    sur l'albedo en sortie du modele, via shadow_calibration.py -- ce qui
    laisse le patch d'entree strictement identique a ce que le modele a
    vu a l'entrainement. attenuate_shadows() est conservee ci-dessus (non
    appelee ici) au cas ou un futur re-entrainement l'integrerait cote
    00_/01_patch_extractor -- dans ce cas seulement, elle redeviendrait
    valide a l'inference aussi.
    """
    raw_patch = extract_patch(ortho_path, centroid_x, centroid_y, patch_size, resolution)
    if raw_patch is None:
        return None, None, None

    result = _normalize_minmax_per_band_with_ratio(raw_patch, max_nodata_ratio)
    if result is None:
        return None, None, None
    normalized, ratio = result
    return raw_patch, normalized, ratio


def patch_to_png_bytes(raw_patch: np.ndarray) -> bytes:
    """Convertit un patch RAW (3, H, W) uint8 en PNG encode (bytes)."""
    img = Image.fromarray(np.moveaxis(raw_patch, 0, -1))  # (H, W, 3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def extract_display_patch(
    ortho_path: Path,
    centroid_x: float,
    centroid_y: float,
    half_extent_m: float = 8.0,
    display_px: int = 300,
    resolution: float = 0.20,
) -> Optional[np.ndarray]:
    """Decoupe un crop RAW plus large que le patch modele, destine
    exclusivement a l'AFFICHAGE (vignette agrandie / vue plein ecran) --
    JAMAIS envoye au classifieur ni au modele d'albedo.

    Le patch modele (extract_patch(), PATCH_SIZE=64 px a resolution=0.20m,
    soit 12.8m de cote) est volontairement etroit -- c'est ce que le modele
    a vu a l'entrainement, on n'y touche pas (meme principe que
    shadow_calibration.py : ne jamais modifier l'input du modele). Mais ce
    cadrage est trop serre pour un affichage humain -- on ne voit qu'un
    petit bout de toiture flou, pas le batiment. Ce crop-ci utilise un
    half_extent_m plus large (defaut 8m -> 16m de cote, coherent avec
    ign_fetch.crop_thumbnail utilise en mode adresse unique) et un
    redimensionnement LANCZOS vers display_px pour un rendu net a l'ecran.

    Retourne None si le centroide est hors emprise de la tuile fournie.
    """
    with rasterio.open(ortho_path) as src:
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

        data = src.read(window=win_clamped)[:N_BANDS]

    if data.shape[1] == 0 or data.shape[2] == 0:
        return None

    if data.shape[1:] != (display_px, display_px):
        channels = []
        for c in range(data.shape[0]):
            img = Image.fromarray(data[c]).resize(
                (display_px, display_px), Image.LANCZOS
            )
            channels.append(np.array(img))
        data = np.stack(channels)

    return data