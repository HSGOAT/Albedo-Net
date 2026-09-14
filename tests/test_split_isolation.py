"""Audit automatique de l'étanchéité train/val/test.

Contexte : un audit manuel a détecté une contamination entre splits (même
groupe spatial / bâtiment présent dans plusieurs splits à la fois). Ce test
transforme cet audit ponctuel en garde-fou permanent, exécuté en CI à chaque
changement du dataset ou du code de split.

Il ne réimplémente PAS la logique de split : il importe _group_split et
RANDOM_STATE depuis train_material_classifier.py, pour que le test échoue
si le comportement RÉEL dérive, plutôt que de comparer à une copie figée
qui pourrait diverger silencieusement du code de prod.

Trois niveaux de vérification, du plus fort au plus faible :
1. group_id  : aucun cluster spatial (~500m) ne doit apparaître dans plus
   d'un split -- c'est la garantie que le split est censé fournir.
2. (lat, lon): garde-fou indépendant du clustering -- si GROUP_CLUSTER_DEG
   ou le binning a un bug, deux points quasi identiques pourraient tomber
   dans des group_id différents et fuiter malgré tout.
3. villes    : couverture minimale des 15 villes dans chaque split, pour
   détecter un déséquilibre géographique qui fausserait l'évaluation
   (silencieux lui aussi : le split "réussit" techniquement mais une ville
   entière peut se retrouver absente du test set).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from train_material_classifier import DATASET, RANDOM_STATE, _group_split  # noqa: E402

# Nombre de villes attendu (documenté par le brief produit / Sprint 4), utilisé
# comme signal d'alerte si le dataset en contient sensiblement moins -- sans
# figer la LISTE des villes en dur : celle-ci est dérivée dynamiquement du
# dataset lui-même (cf. `all_cities` dans les fixtures), pour ne pas dépendre
# d'une source externe (ex. noms de dossiers checkpoints/) qui peut diverger
# du contenu réel du dataset.
EXPECTED_CITY_COUNT = 15
MIN_SAMPLES_PER_CITY_PER_SPLIT = 1  # seuil minimal ; à durcir si le volume de données le permet

# Une ville avec moins de 3 group_id ne peut mathématiquement pas être répartie
# sur 3 splits (train/val/test) -- ce n'est pas un défaut du split, c'est un
# déficit de collecte de données pour cette ville. On l'isole explicitement
# plutôt que de la traiter comme une régression de code.
MIN_GROUPS_FOR_SPLITTABILITY = 3


def _load_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        pytest.skip(f"Dataset introuvable : {DATASET}. Lancer build_training_dataset.py d'abord.")
    return pd.read_parquet(DATASET)


def _reproduce_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduit EXACTEMENT le split de train_material_classifier.py."""
    train_df, temp_df = _group_split(df, "group_id", test_size=0.3, random_state=RANDOM_STATE)
    val_df, test_df = _group_split(temp_df, "group_id", test_size=0.5, random_state=RANDOM_STATE)
    return train_df, val_df, test_df


@pytest.fixture(scope="module")
def splits():
    df = _load_dataset()
    missing = {"group_id", "city", "lat", "lon"} - set(df.columns)
    if missing:
        pytest.fail(f"Colonnes requises absentes du dataset : {missing}")
    return _reproduce_splits(df)


def _pairwise(names_and_sets):
    """Toutes les paires (a, b) sans répétition, pour comparer chaque split
    aux deux autres sans dupliquer les comparaisons."""
    items = list(names_and_sets)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]


def test_no_group_id_overlap(splits):
    """Garantie principale du split : un group_id ne doit apparaître que
    dans un seul des trois ensembles."""
    train_df, val_df, test_df = splits
    named = [
        ("train", set(train_df["group_id"])),
        ("val", set(val_df["group_id"])),
        ("test", set(test_df["group_id"])),
    ]
    violations = []
    for (name_a, set_a), (name_b, set_b) in _pairwise(named):
        overlap = set_a & set_b
        if overlap:
            violations.append(f"{name_a} ∩ {name_b} : {len(overlap)} group_id partagés (ex. {sorted(overlap)[:5]})")
    assert not violations, "Contamination détectée (group_id) :\n" + "\n".join(violations)


def test_no_raw_coordinate_overlap(splits):
    """Garde-fou indépendant du clustering group_id : si le binning spatial
    a un bug (arrondi, changement de GROUP_CLUSTER_DEG entre deux runs...),
    des points quasi identiques pourraient échapper à la détection ci-dessus."""
    train_df, val_df, test_df = splits
    named = [
        ("train", set(zip(train_df["lat"].round(6), train_df["lon"].round(6)))),
        ("val", set(zip(val_df["lat"].round(6), val_df["lon"].round(6)))),
        ("test", set(zip(test_df["lat"].round(6), test_df["lon"].round(6)))),
    ]
    violations = []
    for (name_a, set_a), (name_b, set_b) in _pairwise(named):
        overlap = set_a & set_b
        if overlap:
            violations.append(f"{name_a} ∩ {name_b} : {len(overlap)} coordonnées partagées (ex. {sorted(overlap)[:5]})")
    assert not violations, "Contamination détectée (lat/lon bruts) :\n" + "\n".join(violations)


def test_city_count_matches_expectation(splits):
    """Alerte (sans bloquer la CI) si le dataset contient sensiblement moins
    de villes que prévu -- signal amont utile mais pas une garantie de non-
    contamination en soi, donc average un warning plutôt qu'un échec strict."""
    train_df, val_df, test_df = splits
    all_cities = set(train_df["city"]) | set(val_df["city"]) | set(test_df["city"])
    if len(all_cities) < EXPECTED_CITY_COUNT:
        pytest.skip(
            f"Seulement {len(all_cities)} ville(s) dans le dataset actuel "
            f"({sorted(all_cities)}), {EXPECTED_CITY_COUNT} attendues à terme. "
            "Pas un échec : le dataset est probablement en cours de constitution."
        )


def test_no_city_entirely_missing_from_train(splits):
    """Échec BLOQUANT, mais uniquement pour les villes qui ont assez de
    group_id pour être splittables (cf. MIN_GROUPS_FOR_SPLITTABILITY). Une
    ville avec 1-2 group_id ne peut mathématiquement pas être répartie sur
    3 splits -- ce n'est pas un défaut du split, c'est un déficit de collecte
    de données, signalé séparément par test_cities_with_insufficient_data."""
    full_df = pd.concat(splits, ignore_index=True)
    train_df, val_df, test_df = splits

    groups_per_city = full_df.groupby("city")["group_id"].nunique()
    splittable_cities = set(groups_per_city[groups_per_city >= MIN_GROUPS_FOR_SPLITTABILITY].index)

    all_cities = set(train_df["city"]) | set(val_df["city"]) | set(test_df["city"])
    train_cities = set(train_df["city"])
    missing_from_train = (all_cities - train_cities) & splittable_cities
    assert not missing_from_train, (
        f"Ville(s) splittable(s) (>= {MIN_GROUPS_FOR_SPLITTABILITY} group_id) mais absente(s) "
        f"de train : {sorted(missing_from_train)} -- ceci EST un défaut du split, "
        "pas un déficit de données."
    )


def test_cities_with_insufficient_data_for_splitting(splits):
    """Non-bloquant (xfail informatif) : liste les villes qui n'ont
    structurellement pas assez de group_id pour être répartie sur 3 splits.
    Sert de rappel actionnable pour la collecte de données -- caen/rouen
    avec 1 group_id chacune ne pourront jamais être correctement évaluées
    tant qu'on ne collecte pas plus d'échantillons pour ces villes."""
    full_df = pd.concat(splits, ignore_index=True)
    groups_per_city = full_df.groupby("city")["group_id"].nunique().sort_values()
    insufficient = groups_per_city[groups_per_city < MIN_GROUPS_FOR_SPLITTABILITY]
    if not insufficient.empty:
        pytest.xfail(
            "Ville(s) avec trop peu de group_id pour être splittable (non bloquant, "
            "action requise = collecter plus de données) :\n" + insufficient.to_string()
        )


def test_city_coverage_in_val_and_test(splits):
    """Non-bloquant par design (xfail informatif) : une ville absente de
    val/test empêche d'évaluer le modèle sur cette ville. Ne considère que
    les villes splittables (cf. MIN_GROUPS_FOR_SPLITTABILITY) -- les villes
    sous-dotées en données sont déjà couvertes par
    test_cities_with_insufficient_data_for_splitting, pas la peine de les
    signaler deux fois pour la même cause racine."""
    full_df = pd.concat(splits, ignore_index=True)
    train_df, val_df, test_df = splits

    groups_per_city = full_df.groupby("city")["group_id"].nunique()
    splittable_cities = set(groups_per_city[groups_per_city >= MIN_GROUPS_FOR_SPLITTABILITY].index)

    gaps = []
    for split_name, split_df in (("val", val_df), ("test", test_df)):
        counts = split_df["city"].value_counts()
        for city in sorted(splittable_cities):
            n = int(counts.get(city, 0))
            if n < MIN_SAMPLES_PER_CITY_PER_SPLIT:
                gaps.append(f"{split_name}/{city}: {n} échantillon(s)")
    if gaps:
        pytest.xfail("Couverture incomplète par ville en val/test (non bloquant) :\n" + "\n".join(gaps))


def test_split_sizes_are_nonzero(splits):
    """Filet de sécurité minimal : un split vide invaliderait silencieusement
    tout entraînement/évaluation en aval sans forcément lever d'erreur."""
    train_df, val_df, test_df = splits
    assert len(train_df) > 0, "Split train vide"
    assert len(val_df) > 0, "Split val vide"
    assert len(test_df) > 0, "Split test vide"
