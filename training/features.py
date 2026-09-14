"""Extraction de features tabulaires pour le classifieur de matériau de toiture.

Toute modification de la logique d'extraction DOIT s'accompagner d'un incrément
de FEATURE_SET_VERSION. Le dataset et le modèle enregistrent cette version afin
de détecter automatiquement toute incompatibilité (cf. train_material_classifier.py,
qui calcule et enregistre les metriques d'evaluation directement -- il n'existe
pas de script d'evaluation separe pour l'instant).

v3.0 (23/07/2026) -- Diagnostic post-entrainement v2.0 : F1 beton=0.40 (val) /
0.33 (test), nettement le plus faible des 4 classes, cf. rapport
train_material_classifier.py. Hypothese retenue : le jeu de features v2.0 ne
capture ni la teinte (uniquement saturation -- un gris neutre beton et un gris
legerement bleute zinc peuvent avoir la meme saturation moyenne mais une teinte
differente), ni assez de direction en texture (GLCM moyenne sur 2 angles
seulement, 0/90 -- perd le signal directionnel des joints/plis du zinc en
feuilles vs la texture plus isotrope du beton), ni de signal de regularite
structurelle (densite de bords -- toit zinc a joints debout = motif lineaire
regulier, toiture-terrasse beton = surface plus continue).
CHANGEMENTS v2.0 -> v3.0 (ajouts uniquement, aucune feature retiree -- backward
compatible en lecture mais PAS en shape, d'ou l'incrementation de version) :
  - hue_mean, hue_std (teinte HSV, calculee a la main pour rester coherente avec
    le calcul de saturation deja existant, plutot que de dependre d'une conversion
    RGB->HSV externe qui dupliquerait un calcul quasi identique) ;
  - glcm_contrast/homogeneity/energy recalcules sur 4 angles (0, 45, 90, 135)
    au lieu de 2 (0, 90) -- capture le signal directionnel manque en v2.0 ;
  - edge_density : proportion de pixels a fort gradient (Sobel), feature de
    regularite structurelle absente de v2.0.
A RE-EVALUER apres ce changement : si le F1 beton ne s'ameliore pas
significativement, le probleme est probablement plus profond que le jeu de
features tabulaire (ex. qualite/taille des patches, ou necessite d'un modele a
plus grande capacite type CNN sur le patch brut plutot que des features a la main).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import sobel
from skimage.feature import graycomatrix, graycoprops

# Incrémenter à chaque changement de la liste ou de la définition des features.
FEATURE_SET_VERSION = "3.0"

FEATURE_NAMES = [
    "r_mean", "r_std",
    "g_mean", "g_std",
    "b_mean", "b_std",
    "sat_mean", "sat_std",
    "hue_mean", "hue_std",
    "glcm_contrast",
    "glcm_homogeneity",
    "glcm_energy",
    "edge_density",
]

# Seuil (sur gradient Sobel normalise [0,1]) au-dessus duquel un pixel est
# considere comme un "bord". Choix empirique modere -- pas calibre sur verite
# terrain (meme statut de prudence que les autres seuils du pipeline, cf.
# confidence.py) : a ajuster si edge_density s'avere peu discriminante une
# fois le modele re-entraine.
EDGE_GRADIENT_THRESHOLD = 0.15


class InvalidPatchError(ValueError):
    """Levée quand un patch ne peut pas être utilisé pour l'extraction de features."""


def _validate_patch(patch: np.ndarray) -> None:
    if patch is None:
        raise InvalidPatchError("Le patch est None.")
    if patch.ndim != 3 or patch.shape[0] != 3:
        raise InvalidPatchError(f"Forme de patch invalide, attendu (3, H, W), reçu {patch.shape}.")
    if patch.shape[1] < 2 or patch.shape[2] < 2:
        raise InvalidPatchError(f"Patch trop petit pour la texture GLCM : {patch.shape}.")
    if not np.isfinite(patch).all():
        raise InvalidPatchError("Le patch contient des valeurs NaN/Inf.")
    if patch.min() < -1e-6 or patch.max() > 1 + 1e-6:
        raise InvalidPatchError(f"Patch hors de l'intervalle attendu [0,1] (min={patch.min()}, max={patch.max()}).")


def _rgb_to_hue(r: np.ndarray, g: np.ndarray, b: np.ndarray, max_val: np.ndarray, min_val: np.ndarray) -> np.ndarray:
    """Teinte HSV en degres [0, 360), calculee directement depuis les bandes deja
    disponibles (max_val/min_val partages avec le calcul de saturation existant,
    pas de conversion colorimetrique externe redondante).

    Pixels achromatiques (max_val == min_val, gris pur) -> hue = 0 par convention ;
    ce sont exactement les pixels a saturation ~0, deja downweightes par hue_std
    qui restera faible dans les zones tres desaturees (beton clair typiquement).
    """
    delta = max_val - min_val
    delta_safe = np.where(delta < 1e-8, 1.0, delta)  # evite division par zero, corrige juste apres

    hue = np.zeros_like(r)

    is_r_max = (max_val == r) & (delta > 1e-8)
    is_g_max = (max_val == g) & (delta > 1e-8) & ~is_r_max
    is_b_max = (max_val == b) & (delta > 1e-8) & ~is_r_max & ~is_g_max

    hue = np.where(is_r_max, (60 * ((g - b) / delta_safe) + 360) % 360, hue)
    hue = np.where(is_g_max, (60 * ((b - r) / delta_safe) + 120) % 360, hue)
    hue = np.where(is_b_max, (60 * ((r - g) / delta_safe) + 240) % 360, hue)

    return hue


def extract_features(patch: np.ndarray) -> np.ndarray:
    """Calcule le vecteur de features pour un patch (3, H, W) normalisé dans [0, 1].

    Lève InvalidPatchError si le patch est None, mal formé, ou contient des valeurs
    non finies — pour éviter de silencieusement corrompre le dataset avec des NaN.

    Retourne un vecteur 1D de taille len(FEATURE_NAMES), dtype float32.
    """
    _validate_patch(patch)
    r, g, b = patch[0], patch[1], patch[2]
    feats: list[float] = []

    for band in (r, g, b):
        feats.append(float(np.mean(band)))
        feats.append(float(np.std(band)))

    max_val = np.maximum(np.maximum(r, g), b)
    min_val = np.minimum(np.minimum(r, g), b)
    sat = (max_val - min_val) / (max_val + 1e-8)
    feats.append(float(np.mean(sat)))
    feats.append(float(np.std(sat)))

    # Teinte (nouveau en v3.0) -- cf. docstring module. Circularite de la teinte
    # (0 et 360 sont le meme point) ignoree ici : sur des toitures, les teintes
    # observees restent dans une plage limitee (gris/roux/brun/vert-cuivre), le
    # wraparound proche de 0/360 est un cas marginal, pas un blocage pour un
    # premier ajout de cette feature.
    hue = _rgb_to_hue(r, g, b, max_val, min_val)
    feats.append(float(np.mean(hue)))
    feats.append(float(np.std(hue)))

    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gray_uint8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    # 4 angles (0, 45, 90, 135) au lieu de 2 en v2.0 -- capture le signal
    # directionnel de texture manque precedemment (cf. docstring module).
    glcm = graycomatrix(
        gray_uint8, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=256,
        symmetric=True, normed=True,
    )
    feats.append(float(graycoprops(glcm, "contrast").mean()))
    feats.append(float(graycoprops(glcm, "homogeneity").mean()))
    feats.append(float(graycoprops(glcm, "energy").mean()))

    # Densite de bords (nouveau en v3.0) -- proportion de pixels a fort gradient,
    # signal de regularite structurelle absent de v2.0. Gradient Sobel combine
    # (magnitude), normalise par sa valeur max sur le patch pour rester comparable
    # d'un patch a l'autre malgre des contrastes globaux differents.
    sobel_x = sobel(gray, axis=0)
    sobel_y = sobel(gray, axis=1)
    gradient_mag = np.hypot(sobel_x, sobel_y)
    grad_max = gradient_mag.max()
    gradient_norm = gradient_mag / grad_max if grad_max > 1e-8 else gradient_mag
    edge_density = float(np.mean(gradient_norm > EDGE_GRADIENT_THRESHOLD))
    feats.append(edge_density)

    result = np.array(feats, dtype=np.float32)
    assert len(result) == len(FEATURE_NAMES), "Mismatch entre feats calculées et FEATURE_NAMES."
    return result