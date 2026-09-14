"""
materials.py — AlbedoNet App
==============================
Classification du materiau de toiture a partir du patch RGB deja extrait
et normalise (patch_extraction.extract_and_normalize_patch).

HISTORIQUE :
- v1 -> v2.1 : heuristique couleur (distance aux centroides RGB), voir
  git history pour le detail des avertissements (09-11/07/2026).
- v3.0 (22/07/2026) : PHASE 3 DU PLAN "material_classifier" -- remplacement
  de l'heuristique par le vrai classifieur ML entraine (RandomForest sur
  features tabulaires GLCM + couleur, cf. training/features.py,
  training/train_material_classifier.py, training/model_registry.json).
  Le modele est charge une seule fois (cache module-level, verrouille par
  un threading.Lock -- meme pattern que inference.py::_predictor_cache
  pour les checkpoints .pt) et son hash SHA256 est verifie contre
  training/model_registry.json au chargement : demarrage refuse si le
  fichier .joblib ne correspond pas au hash declare (garde-fou equivalent
  a threshold_guard.py pour les anciens centroides).
- v3.1 (23/07/2026) : swap RandomForest -> HistGradientBoostingClassifier
  (training/train_material_classifier_hgb.py), suite a un comparatif sur
  le meme dataset (feature_set_version=3.0, cf. features.py) montrant un
  meilleur F1 sur la classe beton (0.36 -> 0.42 sur le split test), la
  classe la plus problematique jusque-la. Gain modere -- pas une rupture --
  mais retenu car beton etait specifiquement le point faible vise. Modele
  et registre desormais dans des fichiers dedies
  (models/material_classifier_v1_hgb.joblib,
  training/model_registry_hgb.json) pour ne pas ecraser le RandomForest
  precedent, garde disponible pour comparaison ulterieure si besoin.

CE QUE CE MODULE NE FAIT PLUS :
- Il ne retourne plus confidence="heuristique_couleur" pour un patch
  valide -- confidence vaut desormais "ml_classifieur_v1_hgb".
- "indetermine" reste reserve aux patchs invalides (None / mauvaise
  forme), comme avant.

NOUVEAU CHAMP : MaterialResult.proba_max -- probabilite (predict_proba)
associee a la classe predite. Utilise par confidence.py pour penaliser les
predictions peu sures (proba_max faible) au lieu de l'ancienne penalite
"distance au centroide le plus proche" qui n'a plus de sens avec un
classifieur ML.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

# features.py vit dans training/ alors que materials.py est a la racine du
# projet (importe par main.py au demarrage de l'app) -- meme fix que
# build_training_dataset.py / train_material_classifier.py.
_TRAINING_DIR = Path(__file__).resolve().parent / "training"
if _TRAINING_DIR.is_dir() and str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from features import FEATURE_NAMES, FEATURE_SET_VERSION, InvalidPatchError, extract_features

logger = logging.getLogger("albedonet.materials")

MATERIALS: tuple[str, ...] = ("zinc", "ardoise", "tuile_terre_cuite", "beton", "indetermine")
CLASSIFIABLE_MATERIALS: tuple[str, ...] = ("zinc", "ardoise", "tuile_terre_cuite", "beton")

# v3.1 (23/07/2026) -- HistGradientBoostingClassifier retenu apres comparatif
# avec le RandomForest v1 (meilleur F1 beton), cf. HISTORIQUE ci-dessus.
# Fichiers separes de l'ancien RandomForest (pas d'ecrasement) :
# l'ancien modele reste disponible sous models/material_classifier_v1.joblib
# + training/model_registry.json si besoin de revert ou de comparaison.
MODEL_FILE = Path("models/material_classifier_v1_hgb.joblib")
REGISTRY_FILE = Path("training/model_registry_hgb.json")

# En dessous de ce proba_max, la prediction est consideree peu fiable meme
# si techniquement tranchee -- utilise par confidence.py. Calibre a l'oeil
# (meme reserve que le reste du pipeline) : a affiner une fois que
# analyze_annotations.py permet de correler proba_max a l'erreur reelle.
MIN_RELIABLE_PROBA: float = 0.6


@dataclass
class MaterialResult:
    material: str                  # une valeur de MATERIALS
    confidence: str                # "ml_classifieur_v1_hgb" (ou "invalide" si patch invalide)
    proba_max: float | None = None  # probabilite du modele pour la classe predite, None si patch invalide


class _ModelCache:
    """Cache module-level du classifieur, charge une seule fois et verifie
    contre le registre (hash + feature_set_version), avec le meme pattern
    que inference.py::_predictor_cache (threading.Lock pour la concurrence
    en mode batch/zone)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clf = None
        self._classes: list[str] | None = None

    def get(self):
        if self._clf is not None:
            return self._clf, self._classes
        with self._lock:
            if self._clf is not None:
                return self._clf, self._classes
            self._clf, self._classes = self._load()
            return self._clf, self._classes

    def _load(self):
        if not MODEL_FILE.exists():
            raise RuntimeError(
                f"Modele materiau introuvable : {MODEL_FILE}. "
                "Lancer training/train_material_classifier_hgb.py avant de demarrer l'app."
            )
        if not REGISTRY_FILE.exists():
            raise RuntimeError(
                f"Registre modele introuvable : {REGISTRY_FILE}. "
                "Impossible de verifier l'integrite du modele charge."
            )

        registry = json.loads(REGISTRY_FILE.read_text())

        expected_hash = registry.get("model_hash")
        actual_hash = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()[:16]
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(
                f"REFUS DE DEMARRAGE : hash du modele materiau ({actual_hash}) "
                f"ne correspond pas au hash declare dans {REGISTRY_FILE} "
                f"({expected_hash}). Le fichier .joblib a peut-etre ete remplace "
                "sans mise a jour du registre -- regenerer le registre via "
                "train_material_classifier_hgb.py si ce changement est intentionnel."
            )

        registry_feat_version = registry.get("feature_set_version")
        if registry_feat_version and registry_feat_version != FEATURE_SET_VERSION:
            raise RuntimeError(
                f"REFUS DE DEMARRAGE : modele entraine avec "
                f"feature_set_version={registry_feat_version}, mais le code "
                f"actuel de features.py utilise FEATURE_SET_VERSION="
                f"{FEATURE_SET_VERSION}. Featurisation train/inference "
                "incompatible -- reentrainer ou revert features.py."
            )

        clf = joblib.load(MODEL_FILE)
        classes = list(clf.classes_)
        logger.info(
            "Classifieur materiau charge (%s, hash=%s, feature_set_version=%s, classes=%s).",
            MODEL_FILE, actual_hash, FEATURE_SET_VERSION, classes,
        )
        return clf, classes


_model_cache = _ModelCache()


def material_confidence_label(material: str, proba_max: float | None = None) -> str:
    """Etiquette lisible sur la fiabilite d'une prediction donnee, destinee
    a etre exposee directement cote API/UI (cf. main.py)."""
    if proba_max is None:
        return "non calibre (materiau invalide ou hors reference)"
    if proba_max >= MIN_RELIABLE_PROBA:
        return f"classifieur ML (probabilite={proba_max:.0%})"
    return f"classifieur ML (probabilite={proba_max:.0%}, prediction peu sure -- a confirmer)"


def classify_material(patch: np.ndarray) -> MaterialResult:
    """
    patch : ndarray (3, H, W), normalise min-max par bande dans [0, 1]
            (sortie directe de patch_extraction.extract_and_normalize_patch).

    Retourne la prediction du classifieur ML entraine
    (training/train_material_classifier_hgb.py), avec sa probabilite
    associee (proba_max). "indetermine" est retourne UNIQUEMENT si le
    patch est invalide -- jamais comme sortie du modele (le modele ne
    connait pas cette classe, cf. Phase 0 du plan : classe de rejet non
    retenue pour l'entrainement).
    """
    try:
        feats = extract_features(patch)
    except InvalidPatchError:
        return MaterialResult(material="indetermine", confidence="invalide", proba_max=None)

    clf, classes = _model_cache.get()
    X = feats.reshape(1, -1)
    proba = clf.predict_proba(X)[0]
    best_idx = int(np.argmax(proba))
    material = classes[best_idx]
    proba_max = float(proba[best_idx])

    return MaterialResult(material=material, confidence="ml_classifieur_v1_hgb", proba_max=proba_max)