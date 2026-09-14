"""
config.py — AlbedoNet App
==========================
Mapping ville -> checkpoint .pt, + structure de fallback par region
quand le geocodage ne matche aucune ville directement (cf. geocoding.py).

IMPORTANT : liste des checkpoints basee sur le contenu reel de
albedo_data/results/checkpoints/ (screenshot fourni par Pablo, 08/07/2026).
A tenir a jour si de nouveaux dossiers de checkpoints sont ajoutes/renommes.

NOTE 09/07/2026 : "beauce" retire de CITY_MODELS et de tous les fallbacks
region -- diagnostic (diagnose_checkpoint.py) a confirme une tete de
regression degeneree (sortie quasi nulle quelle que soit l'entree, cf.
suivi projet). Le dossier checkpoints/beauce/ peut rester sur disque, il
n'est simplement plus reference nulle part ici. A reintegrer uniquement
apres reentrainement (cf. pistes : aligner phase1_epochs/head_hidden/
phase2_head_dropout/llrd_decay sur la config provence_rurale, qui elle est
saine).
"""

from pathlib import Path

CHECKPOINTS_DIR = Path("checkpoints")

# ──────────────────────────────────────────────────────────────────────────
# 1. Mapping ville (cle normalisee) -> chemin du checkpoint
# ──────────────────────────────────────────────────────────────────────────
# Cle = nom normalise (minuscules, sans accents/espaces) tel que produit par
# geocoding.normalize_city_name(). Doit correspondre aux noms de dossiers.

CITY_MODELS: dict[str, str] = {
    "clermont":        str(CHECKPOINTS_DIR / "clermont" / "best_finetune.pt"),
    "grenoble":        str(CHECKPOINTS_DIR / "grenoble" / "best_finetune.pt"),
    "lille":           str(CHECKPOINTS_DIR / "lille" / "best_finetune.pt"),
    "lyon":            str(CHECKPOINTS_DIR / "lyon" / "best_finetune.pt"),
    "marseille":       str(CHECKPOINTS_DIR / "marseille" / "best_finetune.pt"),
    "metz":            str(CHECKPOINTS_DIR / "metz" / "best_finetune.pt"),
    "montpellier":     str(CHECKPOINTS_DIR / "montpellier" / "best_finetune.pt"),
    "nantes":          str(CHECKPOINTS_DIR / "nantes" / "best_finetune.pt"),
    "nice":            str(CHECKPOINTS_DIR / "nice" / "best_finetune.pt"),
    "paris":           str(CHECKPOINTS_DIR / "paris" / "best_finetune.pt"),
    "provence_rurale": str(CHECKPOINTS_DIR / "provence_rurale" / "best_finetune.pt"),
    "strasbourg":      str(CHECKPOINTS_DIR / "strasbourg" / "best_finetune.pt"),
    "toulouse":        str(CHECKPOINTS_DIR / "toulouse" / "best_finetune.pt"),
    "tours":           str(CHECKPOINTS_DIR / "tours" / "best_finetune.pt"),
}

# "provence_rurale" n'est pas un nom de commune : c'est un modele entraine
# sur une zone rurale. Il ne doit jamais etre matche par nom de ville depuis
# le geocodage BAN (aucune commune ne s'appelle "Provence rurale") -- il ne
# sert que via le fallback region.
RURAL_MODELS = {"provence_rurale"}

# Modele generique de secours si vraiment rien ne matche (ni ville, ni region
# choisie par l'utilisateur). A definir si Pablo en a un ; sinon on utilise
# la ville la plus peuplee comme generique par defaut.
GENERIC_FALLBACK_MODEL = CITY_MODELS["paris"]

# ──────────────────────────────────────────────────────────────────────────
# 2. Communes couvertes directement (nom normalise -> cle CITY_MODELS)
# ──────────────────────────────────────────────────────────────────────────
# Utilise en priorite 1 par geocoding.py (matching direct par nom de commune
# renvoye par l'API BAN). Ajouter ici des alias si necessaire (ex. anciennes
# orthographes, arrondissements).

COMMUNE_TO_CITY_KEY: dict[str, str] = {
    "paris": "paris",
    "lyon": "lyon",
    "marseille": "marseille",
    "grenoble": "grenoble",
    "clermontferrand": "clermont",
    "lille": "lille",
    "metz": "metz",
    "montpellier": "montpellier",
    "nantes": "nantes",
    "nice": "nice",
    "strasbourg": "strasbourg",
    "toulouse": "toulouse",
    "tours": "tours",
    # Paris/Lyon/Marseille : arrondissements (l'API BAN renvoie parfois
    # "Paris 15e Arrondissement" etc. -> normalise vers la ville mere)
}

# ──────────────────────────────────────────────────────────────────────────
# 3. Regions administratives -> villes couvertes (pour le fallback)
# ──────────────────────────────────────────────────────────────────────────
# Utilise en priorite 2 quand le nom de commune ne matche rien : on se rabat
# sur le code INSEE / departement -> region (voir geocoding.INSEE_DEPT_TO_REGION),
# puis sur cette table region -> ville(s) couverte(s).
#
# Quand une region a plusieurs villes couvertes, la premiere de la liste est
# la ville "grande metropole" proposee par defaut ; les suivantes sont
# proposees comme options plus precises dans le menu de secours cote
# frontend (cf. geocoding.py / main.py).
#
# "beauce" retire des listes ci-dessous (Centre-Val de Loire, Ile-de-France)
# suite au diagnostic du 09/07/2026 -- ces regions retombent donc sur un
# seul candidat (Centre-Val de Loire) ou sur "paris" seul (Ile-de-France),
# ce qui les fait sortir du cas "ambigu" (plus qu'un seul candidat).

REGION_TO_CITY_KEYS: dict[str, list[str]] = {
    "Auvergne-Rhone-Alpes":        ["lyon", "grenoble", "clermont"],
    "Bourgogne-Franche-Comte":     [],  # non couverte -> generique
    "Bretagne":                    [],  # non couverte -> generique
    "Centre-Val de Loire":         ["tours"],
    "Corse":                       [],  # non couverte -> generique
    "Grand Est":                   ["strasbourg", "metz"],
    "Hauts-de-France":             ["lille"],
    "Ile-de-France":               ["paris"],
    "Normandie":                   [],  # non couverte -> generique
    "Nouvelle-Aquitaine":          [],  # non couverte -> generique
    "Occitanie":                   ["toulouse", "montpellier"],
    "Pays de la Loire":            ["nantes"],
    "Provence-Alpes-Cote d'Azur":  ["marseille", "nice", "provence_rurale"],
}

# Liste ordonnee des regions a proposer dans le menu de secours cote
# frontend (region administrative francaise standard, metropole uniquement).
REGIONS_FR: list[str] = list(REGION_TO_CITY_KEYS.keys())


def get_checkpoint_path(city_key: str) -> str:
    """Retourne le chemin du checkpoint pour une cle ville donnee, ou fallback generique."""
    return CITY_MODELS.get(city_key, GENERIC_FALLBACK_MODEL)
