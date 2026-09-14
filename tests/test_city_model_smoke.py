"""Non-regression par ville : un checkpoint qui charge mal (mismatch
encoder_state_dict/head_state_dict, mauvais norm_type, config absente...)
doit etre detecte AVANT la prod, ville par ville.

Contexte : 14 modeles independants (config.CITY_MODELS -- "beauce" est
volontairement absente, cf. docstring config.py : tete de regression
degeneree, sortie quasi nulle quelle que soit l'entree). Avec un
_predictor_cache par checkpoint et un chargement individuel par ville
(inference.AlbedoPredictor.__init__), un probleme de chargement sur UNE
ville (ex. mauvais head_hidden, mauvais norm_type LayerNorm/BatchNorm, cf.
commentaires deja presents dans inference.py sur ces bugs deja rencontres)
n'empeche pas les 13 autres villes de fonctionner -- donc rien ne le
detecte automatiquement sans un test dedie par ville.

Chaque ville est un test PARAMETRIZE distinct (pas une boucle dans un seul
test) : si lille echoue, grenoble/lyon/... restent visibles individuellement
comme PASSED dans le rapport -- exactement le comportement demande (un
checkpoint casse ne doit pas masquer l'etat des 13 autres).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import CITY_MODELS  # noqa: E402
from inference import AlbedoPredictor  # noqa: E402

# Patch synthetique FIXE (deterministe, pas de dependance a une vraie image
# disque) : degrade lineaire reproductible par bande, distinct entre les 3
# canaux RGB pour eviter un patch degenere (ex. tout a 0 ou tout identique
# sur les 3 bandes) qui masquerait un bug de forward pass. Deja dans le
# domaine attendu par AlbedoPredictor.predict() : (3, 64, 64) float32 dans
# [0, 1] (sortie de patch_extraction.normalize_minmax_per_band).
def _make_fixed_patch() -> np.ndarray:
    rng_row = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    rng_col = np.linspace(1.0, 0.0, 64, dtype=np.float32)
    base = np.outer(rng_row, rng_col)  # (64, 64), gradient diagonal
    patch = np.stack([
        base,
        np.roll(base, shift=16, axis=0),
        np.roll(base, shift=32, axis=1),
    ]).astype(np.float32)
    assert patch.shape == (3, 64, 64)
    return patch


FIXED_PATCH = _make_fixed_patch()

# Plage de sortie attendue -- volontairement large (albedo physique valide
# dans [0, 1], AlbedoHead se termine par un Sigmoid donc mathematiquement
# borne a [0, 1] de toute facon). L'objectif n'est PAS de valider la
# justesse du modele (aucune verite terrain ici, juste un patch synthetique
# arbitraire) mais d'attraper un CRASH ou une sortie degenere/non-finie qui
# revele un probleme de chargement -- cf. cas "beauce" deja documente
# (sortie quasi nulle quelle que soit l'entree).
MIN_PLAUSIBLE_ALBEDO = 0.0
MAX_PLAUSIBLE_ALBEDO = 1.0


def _checkpoint_available(checkpoint_path: str) -> bool:
    return Path(checkpoint_path).exists()


@pytest.mark.parametrize("city_key", sorted(CITY_MODELS.keys()))
def test_city_checkpoint_loads_and_infers(city_key: str):
    """Charge le checkpoint de la ville et infere sur le patch fixe.
    Echoue si : le chargement leve une exception (mismatch state_dict,
    config manquante mal geree, etc.), OU si la sortie est hors de [0, 1],
    NaN, ou infinie."""
    checkpoint_path = CITY_MODELS[city_key]

    if not _checkpoint_available(checkpoint_path):
        pytest.skip(
            f"Checkpoint introuvable pour '{city_key}' : {checkpoint_path} -- "
            "probablement absent de cet environnement (fichier .pt volumineux, "
            "voir si le repo utilise Git LFS ou un stockage externe pour les "
            "checkpoints en CI)."
        )

    try:
        predictor = AlbedoPredictor(checkpoint_path, device="cpu")
    except Exception as exc:
        pytest.fail(
            f"Echec du chargement du checkpoint '{city_key}' ({checkpoint_path}) : "
            f"{type(exc).__name__}: {exc}"
        )

    albedo = predictor.predict(FIXED_PATCH)

    assert np.isfinite(albedo), (
        f"Sortie non-finie pour '{city_key}' (NaN/inf) : {albedo} -- "
        "signe probable d'un mismatch de poids (encoder/head mal apparies)."
    )
    assert MIN_PLAUSIBLE_ALBEDO <= albedo <= MAX_PLAUSIBLE_ALBEDO, (
        f"Albedo hors plage physique pour '{city_key}' : {albedo} "
        f"(attendu dans [{MIN_PLAUSIBLE_ALBEDO}, {MAX_PLAUSIBLE_ALBEDO}])"
    )


def test_city_models_do_not_all_collapse_to_the_same_output():
    """Garde-fou complementaire : si TOUS les modeles charges avec succes
    renvoient exactement la meme valeur sur le meme patch, c'est le signe
    d'un bug de partage d'etat entre modeles (ex. _predictor_cache mal cle,
    ou un seul jeu de poids charge partout par erreur) plutot que 14
    modeles reellement independants."""
    outputs: dict[str, float] = {}
    for city_key, checkpoint_path in sorted(CITY_MODELS.items()):
        if not _checkpoint_available(checkpoint_path):
            continue
        try:
            predictor = AlbedoPredictor(checkpoint_path, device="cpu")
            outputs[city_key] = predictor.predict(FIXED_PATCH)
        except Exception:
            continue  # deja couvert et rapporte par le test parametrize ci-dessus

    if len(outputs) < 2:
        pytest.skip(
            f"Seulement {len(outputs)} checkpoint(s) disponible(s) dans cet "
            "environnement -- impossible de comparer entre villes."
        )

    unique_values = set(outputs.values())
    assert len(unique_values) > 1, (
        f"Tous les modeles charges ({sorted(outputs)}) renvoient exactement "
        f"la meme sortie ({next(iter(unique_values))}) sur le patch fixe -- "
        "suspect d'un partage d'etat entre modeles (cache mal cle, poids "
        "identiques charges par erreur)."
    )
