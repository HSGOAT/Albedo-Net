"""
shadow_calibration.py — AlbedoNet App
========================================
Correction du biais d'ombre en POST-PREDICTION (sur l'albedo en sortie du
modele), et non sur le patch en entree.

Pourquoi ici et pas dans patch_extraction.py :
  - Les modeles ont ete entraines (00_/01_patch_extractor.py) sur des patches
    RGB bruts, sans aucune correction d'ombre. Toute modification du patch
    AVANT l'encodeur (ex. l'ancien attenuate_shadows()) cree un decalage
    train/inference : le modele n'a jamais vu ce type de pixel a
    l'entrainement, sa reponse dans cette zone du domaine d'entree n'est pas
    fiable (cf. suivi projet -- retrait de attenuate_shadows() du chemin
    d'inference).
  - Corriger l'albedo PREDIT, apres coup, ne touche jamais a ce que le
    modele voit : le forward pass est strictement identique a l'entrainement.
    Le biais du modele sur les pixels sombres (il a appris "sombre -> albedo
    bas", meme quand le materiau reel est clair et juste dans l'ombre) est
    donc corrige a la sortie, de facon explicite et auditable, plutot que
    fondu dans une modification opaque de l'input.

Methode :
  1. detect_shadow_fraction() estime la proportion de pixels ombragés dans
     le patch RAW, via un critere HSV (valeur basse + faible saturation --
     les ombres restent bleutees/peu contrastees, cf. lumiere diffuse du
     ciel plutot que solaire directe).
  2. calibrate_albedo() applique un facteur de correction multiplicatif,
     PONDERE par cette fraction d'ombre (un patch a 10% d'ombre est presque
     pas touche, un patch a 90% d'ombre recoit (quasi) tout le facteur) --
     ca evite les discontinuites brutales entre patches voisins legerement/
     fortement ombrages.
  3. Le facteur de correction est parametre par materiau (SHADOW_CORRECTION_FACTOR)
     quand classify_material() a pu determiner un materiau, sinon un facteur
     generique DEFAULT_SHADOW_CORRECTION_FACTOR est utilise.

Calibration des facteurs (SHADOW_CORRECTION_FACTOR) :
  Cles alignees sur le vocabulaire du classifieur ML actuel (materials.py
  v3.0) : "zinc", "ardoise", "tuile_terre_cuite", "beton", "indetermine".
  Ce sont des valeurs de DEPART raisonnables (ardoise/tuile proches de
  1.0-1.15 -- l'ombre change peu la lecture relative d'un materiau deja
  sombre --, zinc plus eleve -- l'ombre y masque davantage l'ecart reel avec
  un pixel eclaire, car le zinc est tres reflechissant). A REMPLACER des que
  possible par une vraie calibration empirique (meme methode que
  heat_island_calibration.py : comparer, par ville/materiau, l'albedo moyen
  predit sur pixels ombrages vs non-ombrages sur un scan de zone connu),
  plutot que ces valeurs a dire d'expert.
  ATTENTION : si classify_material() change de vocabulaire, mettre a jour ce
  dict ET EXPECTED_MATERIAL_KEYS en meme temps -- sinon .get() retombe
  silencieusement sur DEFAULT_SHADOW_CORRECTION_FACTOR pour toutes les cles
  (bug deja rencontre une fois sur ce module, cf. warning de calibrate_albedo).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from materials import MATERIALS

logger = logging.getLogger("albedo.app.shadow_calibration")

# --- Detection d'ombre (sur le patch RAW, jamais sur l'input du modele) ---
SHADOW_VALUE_THRESHOLD: int = 80        # HSV V < seuil -> candidat ombre
SHADOW_SATURATION_MAX: float = 0.45     # faible saturation -> lumiere diffuse (ombre)
                                          # plutot que materiau sombre mais eclaire
                                          # (asphalte/toiture sombre = souvent plus
                                          # sature qu'une ombre bleutee)

# --- Facteurs de correction multiplicative de l'albedo, par materiau ---
# facteur > 1.0 : on remonte l'albedo predit (le modele l'a sous-estime a
# cause de l'ombre). Volontairement conservateur (proche de 1) tant que non
# calibre empiriquement -- mieux vaut sous-corriger que sur-corriger.
# NOTE : ces cles doivent correspondre EXACTEMENT au vocabulaire retourne par
# le classifieur ML actuel (materials.py v3.0 / classify_material()), pas a
# l'ancienne heuristique RGB. Vocabulaire actuel :
#   {"zinc", "ardoise", "tuile_terre_cuite", "beton", "indetermine"}
# Si ce set change, mettre a jour EXPECTED_MATERIAL_KEYS en meme temps que ce
# dict, sinon le garde-fou de calibrate_albedo() devient muet.
SHADOW_CORRECTION_FACTOR: dict[str, float] = {
    "zinc":              1.35,  # tres reflechissant : l'ombre ecrase fortement
                                 # la luminance mesuree relativement a l'albedo
                                 # reel -> correction la plus forte du lot
    "ardoise":           1.10,  # sombre par nature, l'ombre change peu la
                                 # lecture relative (equivalent de l'ancien
                                 # "toiture_sombre")
    "tuile_terre_cuite": 1.15,  # intermediaire ; coherent avec le biais +0.104
                                 # deja identifie sur ce materiau en calibration
    "beton":             1.25,  # inchange (seul overlap avec l'ancien vocabulaire)
    "indetermine":       1.15,  # valeur neutre explicite, volontairement distincte
                                 # du defaut generique pour ne pas coincider par
                                 # accident avec "beton"
}
DEFAULT_SHADOW_CORRECTION_FACTOR: float = 1.15  # applique seulement si material
                                                  # est None ou hors vocabulaire connu
# Source unique de verite : le vocabulaire de materials.MATERIALS, pas une
# copie codee en dur ici (c'est exactement cette duplication qui a cause le
# bug initial de desynchronisation asphalte/beton/... vs zinc/ardoise/...).
EXPECTED_MATERIAL_KEYS: frozenset[str] = frozenset(MATERIALS)

_missing = EXPECTED_MATERIAL_KEYS - SHADOW_CORRECTION_FACTOR.keys()
if _missing:
    logger.warning(
        "shadow_calibration: materials.MATERIALS contient des classes absentes "
        "de SHADOW_CORRECTION_FACTOR (%s) -- ces materiaux retomberont sur "
        "DEFAULT_SHADOW_CORRECTION_FACTOR=%.2f faute de valeur dediee. "
        "Ajouter une entree calibree pour chacun.",
        sorted(_missing), DEFAULT_SHADOW_CORRECTION_FACTOR,
    )

# Fraction d'ombre en-dessous de laquelle on ne corrige pas du tout (bruit /
# ombres ponctuelles negligeables, ex. cheminee, antenne).
MIN_SHADOW_FRACTION_TO_CORRECT: float = 0.05


@dataclass
class ShadowCalibrationResult:
    albedo_corrige: float
    albedo_brut: float
    shadow_fraction: float
    facteur_applique: float
    materiau_utilise: Optional[str]


def detect_shadow_fraction(raw_patch: np.ndarray) -> float:
    """Estime la fraction de pixels ombrages dans un patch RGB (3, H, W) uint8.

    Critere : value HSV basse ET saturation basse (les ombres portees restent
    bleutees/peu contrastees car eclairees par la lumiere diffuse du ciel,
    pas par le soleil direct -- contrairement a un materiau intrinsequement
    sombre mais eclaire, qui garde en general plus de saturation/contraste).
    """
    if raw_patch.ndim != 3 or raw_patch.shape[0] != 3:
        raise ValueError(f"raw_patch attendu (3, H, W), recu {raw_patch.shape}")

    img = np.moveaxis(raw_patch, 0, -1).astype(np.uint8)  # (H, W, 3)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32) / 255.0

    shadow_mask = (v < SHADOW_VALUE_THRESHOLD) & (s < SHADOW_SATURATION_MAX)
    return float(shadow_mask.mean())


def calibrate_albedo(
    albedo: float,
    raw_patch: np.ndarray,
    material: Optional[str] = None,
) -> ShadowCalibrationResult:
    """Point d'entree principal : corrige l'albedo PREDIT selon la fraction
    d'ombre detectee dans le patch RAW correspondant.

    A appeler juste apres predict_albedo() / predict_albedo_batch(), jamais
    avant (le patch envoye au modele ne doit pas etre modifie -- cf. docstring
    du module).
    """
    shadow_fraction = detect_shadow_fraction(raw_patch)

    if shadow_fraction < MIN_SHADOW_FRACTION_TO_CORRECT:
        return ShadowCalibrationResult(
            albedo_corrige=albedo,
            albedo_brut=albedo,
            shadow_fraction=shadow_fraction,
            facteur_applique=1.0,
            materiau_utilise=material,
        )

    if material and material not in EXPECTED_MATERIAL_KEYS:
        logger.warning(
            "calibrate_albedo: materiau '%s' absent de SHADOW_CORRECTION_FACTOR "
            "(cles attendues : %s) -- repli sur DEFAULT_SHADOW_CORRECTION_FACTOR=%.2f. "
            "Verifier une eventuelle desynchronisation avec le vocabulaire de "
            "classify_material() dans materials.py.",
            material, sorted(EXPECTED_MATERIAL_KEYS), DEFAULT_SHADOW_CORRECTION_FACTOR,
        )

    facteur_max = SHADOW_CORRECTION_FACTOR.get(material, DEFAULT_SHADOW_CORRECTION_FACTOR) \
        if material else DEFAULT_SHADOW_CORRECTION_FACTOR

    # Ponderation lineaire par la fraction d'ombre : un patch a 100% d'ombre
    # recoit tout le facteur, un patch partiellement ombrage n'en recoit
    # qu'une fraction proportionnelle -- evite les sauts brusques entre
    # patches voisins.
    facteur_applique = 1.0 + (facteur_max - 1.0) * shadow_fraction

    albedo_corrige = float(np.clip(albedo * facteur_applique, 0.0, 1.0))

    return ShadowCalibrationResult(
        albedo_corrige=albedo_corrige,
        albedo_brut=albedo,
        shadow_fraction=shadow_fraction,
        facteur_applique=facteur_applique,
        materiau_utilise=material,
    )
    