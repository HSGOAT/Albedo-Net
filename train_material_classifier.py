"""Entraînement du classifieur de matériau de toiture.

Changements clés vs. la version précédente :
- split train/val/test par GroupShuffleSplit sur `group_id` (cluster spatial), pour
  éviter qu'un toit mitoyen du même pâté de maisons ne se retrouve à la fois en train
  et en test (fuite spatiale qui gonflait artificiellement l'accuracy) ;
- vérification que le dataset a été produit avec la version de features attendue ;
- le registre inclut désormais feature_set_version, la taille des splits, et les
  hyperparamètres retenus.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit

# features.py vit dans training/ alors que ce script est pense pour tourner
# depuis la racine du projet (DATASET et REGISTRY_FILE ci-dessous sont des
# chemins relatifs a la racine) -- meme fix que build_training_dataset.py :
# on ajoute training/ au sys.path pour que l'import fonctionne sans deplacer
# features.py ni changer les chemins relatifs existants.
_TRAINING_DIR = Path(__file__).resolve().parent / "training"
if _TRAINING_DIR.is_dir() and str(_TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAINING_DIR))

from features import FEATURE_NAMES, FEATURE_SET_VERSION

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATASET = Path("datasets/materials_v1.parquet")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "material_classifier_v1_hgb.joblib"
REGISTRY_FILE = Path("training/model_registry_hgb.json")

RANDOM_STATE = 42


def _check_feature_version(dataset_path: Path) -> None:
    version_file = dataset_path.parent / f"{dataset_path.stem}.feature_version.txt"
    if not version_file.exists():
        logger.warning("Aucun fichier de version de features trouvé à côté de %s ; "
                        "impossible de vérifier la compatibilité.", dataset_path)
        return
    dataset_version = version_file.read_text().strip()
    if dataset_version != FEATURE_SET_VERSION:
        raise RuntimeError(
            f"Le dataset a été construit avec feature_set_version={dataset_version}, "
            f"mais le code actuel utilise FEATURE_SET_VERSION={FEATURE_SET_VERSION}. "
            "Reconstruire le dataset (build_training_dataset.py) avant d'entraîner."
        )


def _group_split(df: pd.DataFrame, group_col: str, test_size: float, random_state: int):
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    idx_a, idx_b = next(splitter.split(df, groups=df[group_col]))
    return df.iloc[idx_a], df.iloc[idx_b]


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Dataset introuvable : {DATASET}. Lancer build_training_dataset.py d'abord.")

    _check_feature_version(DATASET)
    df = pd.read_parquet(DATASET)

    missing_cols = set(FEATURE_NAMES + ["material", "group_id"]) - set(df.columns)
    if missing_cols:
        raise RuntimeError(f"Colonnes manquantes dans le dataset : {missing_cols}")

    X = df[FEATURE_NAMES].values
    y = df["material"].values

    if "group_id" not in df.columns:
        raise RuntimeError("Colonne group_id absente : régénérer le dataset avec la version à jour de build_training_dataset.py.")

    # Split groupé : aucun group_id ne se retrouve dans plus d'un split.
    train_df, temp_df = _group_split(df, "group_id", test_size=0.3, random_state=RANDOM_STATE)
    val_df, test_df = _group_split(temp_df, "group_id", test_size=0.5, random_state=RANDOM_STATE)

    # .to_numpy(dtype=object) plutot que .values : le parquet charge "material" avec
    # un dtype backend pyarrow (ArrowExtensionArray), que sklearn/joblib ne savent pas
    # indexer par fancy indexing lors du decoupage en folds de cross-validation
    # (TypeError: only integer scalar arrays can be converted to a scalar index).
    # Un numpy array standard evite le probleme sans toucher au contenu des donnees.
    X_train, y_train = train_df[FEATURE_NAMES].values, train_df["material"].to_numpy(dtype=object)
    X_val, y_val = val_df[FEATURE_NAMES].values, val_df["material"].to_numpy(dtype=object)
    X_test, y_test = test_df[FEATURE_NAMES].values, test_df["material"].to_numpy(dtype=object)

    logger.info("Split (groupé par cluster spatial) : train=%d val=%d test=%d",
                len(train_df), len(val_df), len(test_df))

    class_counts = pd.Series(y_train).value_counts()
    min_class_count = class_counts.min()
    cv_folds = min(5, min_class_count)
    if cv_folds < 5:
        logger.warning("Classe la plus rare n'a que %d échantillons en train ; cv réduit à %d folds.",
                        min_class_count, cv_folds)
    if cv_folds < 2:
        raise RuntimeError(f"Pas assez d'échantillons pour la classe minoritaire (train counts:\n{class_counts})")

    param_grid = {
        "max_iter": [100, 200],
        "max_depth": [None, 10, 20],
        "learning_rate": [0.05, 0.1],
        "min_samples_leaf": [5, 10, 20],
    }
    # HistGradientBoostingClassifier n'a pas de class_weight="balanced" natif
    # (contrairement a RandomForestClassifier) -- on reproduit l'equivalent via
    # sample_weight, calcule sur y_train uniquement pour ne pas fuiter
    # d'information des autres splits dans la ponderation.
    class_counts_arr = pd.Series(y_train).value_counts()
    weight_per_class = {cls: len(y_train) / (len(class_counts_arr) * count) for cls, count in class_counts_arr.items()}
    sample_weight = np.array([weight_per_class[label] for label in y_train])

    search = GridSearchCV(
        HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        param_grid, cv=cv_folds, scoring="f1_macro", n_jobs=-1,
    )
    search.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info("Meilleurs hyperparamètres : %s", search.best_params_)

    # Sélection de modèle sur la validation (le grid search cv est déjà fait sur train ;
    # on garde val comme vérification indépendante avant le verdict final sur test).
    val_pred = search.predict(X_val)
    logger.info("Rapport sur validation :\n%s", classification_report(y_val, val_pred, zero_division=0))

    y_pred = search.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    logger.info("Rapport sur test :\n%s", classification_report(y_test, y_pred, zero_division=0))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, MODEL_FILE)

    model_hash = hashlib.sha256(MODEL_FILE.read_bytes()).hexdigest()[:16]

    registry = {
        "model_version": "1.1-hgb",
        "dataset_version": "v1",
        "feature_set_version": FEATURE_SET_VERSION,
        "date": str(date.today()),
        "model_file": str(MODEL_FILE.name),
        "model_hash": model_hash,
        "hyperparameters": search.best_params_,
        "split_sizes": {"train": len(train_df), "val": len(val_df), "test": len(test_df)},
        "split_method": "GroupShuffleSplit on group_id (spatial cluster ~500m)",
        "metrics": {
            "accuracy": report["accuracy"],
            "macro avg": report["macro avg"],
            "per_class": {
                cls: report[cls] for cls in report
                if cls not in ("accuracy", "macro avg", "weighted avg")
            },
        },
    }
    REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)
    logger.info("Modèle et registre sauvegardés (%s, hash=%s).", MODEL_FILE, model_hash)


if __name__ == "__main__":
    main()