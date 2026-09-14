# AlbedoNet

Estimation de l'albédo de toitures à partir d'orthophotos IGN, via des
modèles MAE (Masked Autoencoder) fine-tunés par ville, avec classification
du matériau de toiture (zinc / ardoise / tuile terre cuite / béton) pour
calibrer l'estimation. Détail complet de l'architecture et des choix de
conception : [`METHODOLOGIE.md`](METHODOLOGIE.md).

## Installation

Python 3.11 recommandé (version utilisée en CI).

```bash
python -m venv .venv
source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` couvre le produit officiel (backend API + pipeline
d'inférence). Pour les outils annexes (app Streamlit archivée, scraping
OSM, diagnostics d'entraînement), voir
[`requirements-legacy.txt`](requirements-legacy.txt).

## Checkpoints modèles (obligatoires, non versionnés)

Les 14 checkpoints `.pt` par ville (~260 Mo chacun) ne sont **pas** dans
le dépôt git (cf. `.gitignore`). Ils doivent être placés dans :

```
checkpoints/<cle_ville>/best_finetune.pt
```

où `<cle_ville>` correspond aux clés de `config.CITY_MODELS` (ex.
`checkpoints/paris/best_finetune.pt`, `checkpoints/lyon/best_finetune.pt`,
...). Sans ces fichiers, l'app démarre mais toute requête d'inférence
échouera au chargement du checkpoint.

> Il n'existe pas encore de procédure documentée pour récupérer ces
> checkpoints (stockage externe à définir, cf. `METHODOLOGIE.md` § "Ce qui
> reste ouvert"). Se rapprocher de l'équipe projet pour se les procurer.

Le classifieur de matériau (`models/material_classifier_v1_hgb.joblib`),
lui, **est** versionné dans le dépôt (~1.5 Mo).

## Lancer l'application

```bash
uvicorn main:app --reload --port 8000
```

Puis ouvrir `http://localhost:8000` — `main.py` sert `static/index.html`
(le frontend) et expose l'API REST (`/api/single`, `/api/batch`,
`/api/batch.csv`, `/api/zone`, `/api/zone.csv`, `/api/zone/stream`).

## Lancer les tests

```bash
pytest tests/ -v
```

Ces tests sont des **audits** (étanchéité train/val/test, cohérence des
clés ville entre modules, non-régression par ville sur les checkpoints,
synchro confidence/materials, régression du classifieur matériau) — pas
des tests unitaires classiques. Ils tournent aussi automatiquement en CI
sur tout changement touchant les données, seuils, ou config (cf.
`.github/workflows/data-and-thresholds-audit.yml`). Certains dépendent des
checkpoints (`test_city_model_smoke.py`) : sans eux, ces tests échouent
plutôt que d'être silencieusement ignorés — c'est voulu, un checkpoint
manquant doit être visible.

## Structure du dépôt

```
main.py                  Backend FastAPI — point d'entree officiel
static/index.html         Frontend (servi par main.py)
legacy/                   App Streamlit archivee (non maintenue, cf. METHODOLOGIE.md §2)

config.py                 CITY_MODELS, fallback regional
city_registry.py          Source de verite des cles ville (anti-desynchro)
threshold_guard.py        Garde-fou anti-changement-silencieux de seuils
geocoding.py               Adresse -> lat/lon -> cle ville (API BAN)
ign_fetch.py               Matching batiment + orthophoto IGN
patch_extraction.py        Decoupe/normalisation du patch 64x64
inference.py                Chargement checkpoint + prediction MAE
materials.py                Classification materiau (ML, HistGradientBoosting)
shadow_calibration.py       Correction d'albedo par materiau (ombre)
confidence.py                Score de confiance de la prediction
postprocess_plausibility.py  Plages de plausibilite bibliographiques
heat_island_calibration.py   Seuils de detection d'ilots de chaleur
zone_scan.py                  Scan de zone (agregation multi-batiments)
pipeline.py                   Orchestration adresse unique / lot d'adresses
versioning.py                  Metadonnees de tracabilite (hash checkpoint)

tests/                     Audits (cf. section "Lancer les tests")
training/                  Entrainement du classifieur materiau
data_collection/           Outils de collecte de donnees (annotation, scraping OSM)
datasets/                  Dataset materiaux versionne (parquet + CSV)
models/                    Architecture des modeles (ViT, MAE encoder) + classifieur entraine
utils/                     Utilitaires partages (masking, normalisation de bandes)
```

## Prochaines étapes

Suivi à jour dans [`METHODOLOGIE.md`](METHODOLOGIE.md), section "Ce qui
reste ouvert". Points notables au moment de la rédaction de ce README :
vérification d'intégrité des checkpoints `.pt` (actuellement seul le
classifieur matériau a ce garde-fou), collecte du jeu de données
stratifié nécessaire à `validate_thresholds.py`, décision sur le statut
des villes "scraping only" (`city_registry.EXTRA_SCRAPING_ONLY_CITY_KEYS`).
