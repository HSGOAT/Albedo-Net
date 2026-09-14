"""Test de non-régression : le modèle doit conserver un recall >= seuil par classe
sur un jeu de patches réels et figés (fixtures versionnées avec le code).

IMPORTANT : contrairement à une version précédente qui utilisait des patches
aléatoires (np.random.rand), ce test s'appuie sur de vrais patches sauvegardés
au format .npy sous training/tests/fixtures/<materiau>/*.npy. Un test sur du bruit
aléatoire ne détecte aucune régression réelle : le recall serait uniforme et
indépendant de la qualité du modèle.

Pour peupler les fixtures : exporter quelques patches normalisés (3, H, W) du
dataset d'entraînement, connus et vérifiés manuellement, avec
`np.save(f"training/tests/fixtures/{material}/sample_{i}.npy", patch)`.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from features import extract_features

MODEL_FILE = Path("models/material_classifier_v1.joblib")
FIXTURES_DIR = Path(__file__).parent / "fixtures"
MIN_RECALL_PER_CLASS = 0.80
MIN_SAMPLES_PER_CLASS = 3  # en dessous, le recall n'est pas statistiquement significatif


def _load_fixtures() -> dict[str, list[np.ndarray]]:
    fixtures: dict[str, list[np.ndarray]] = {}
    if not FIXTURES_DIR.exists():
        return fixtures
    for material_dir in FIXTURES_DIR.iterdir():
        if not material_dir.is_dir():
            continue
        patches = [np.load(p) for p in sorted(material_dir.glob("*.npy"))]
        if patches:
            fixtures[material_dir.name] = patches
    return fixtures


@pytest.mark.skipif(not MODEL_FILE.exists(), reason="Modèle non trouvé, entraînement pas encore lancé.")
def test_recall_per_class():
    fixtures = _load_fixtures()
    if not fixtures:
        pytest.skip(
            f"Aucune fixture trouvée sous {FIXTURES_DIR}. "
            "Ce test ne peut pas garantir l'absence de régression sans vrais patches figés — "
            "voir le docstring du module pour les peupler."
        )

    clf = joblib.load(MODEL_FILE)
    failures = []

    for material, patches in fixtures.items():
        if len(patches) < MIN_SAMPLES_PER_CLASS:
            failures.append(
                f"{material}: seulement {len(patches)} fixtures (< {MIN_SAMPLES_PER_CLASS}), "
                "résultat non significatif — ajouter des échantillons."
            )
            continue

        correct = 0
        for patch in patches:
            feats = extract_features(patch).reshape(1, -1)
            pred = clf.predict(feats)[0]
            if pred == material:
                correct += 1
        recall = correct / len(patches)
        if recall < MIN_RECALL_PER_CLASS:
            failures.append(f"{material}: recall={recall:.2f} (< {MIN_RECALL_PER_CLASS}, n={len(patches)})")

    assert not failures, "Régressions détectées :\n" + "\n".join(failures)
