# METHODOLOGIE.md — AlbedoNet

> Dernière mise à jour : 14/09/2026 (frontend tranché). Ce document décrit l'état **réel** du
> dépôt, pas l'état visé. Quand quelque chose est en cours ou non résolu,
> c'est dit explicitement — l'objectif est d'éviter qu'une future session
> (agent ou humain) reparte d'hypothèses fausses sur l'architecture.

## 1. Ce que fait le projet

Estimation de l'albédo de toitures à partir d'orthophotos IGN, via des
modèles MAE (Masked Autoencoder) fine-tunés **par ville** (14 checkpoints
indépendants, `config.CITY_MODELS`), plus une classification du matériau de
toiture (zinc / ardoise / tuile terre cuite / béton / indéterminé) qui sert
à calibrer l'albédo estimé (correction d'ombre par matériau, cf.
`shadow_calibration.py`).

Pipeline bout-en-bout (`pipeline.py`, `zone_scan.py`) :

1. `geocoding.py` — adresse → lat/lon + commune/citycode (API BAN), puis
   résolution de la clé ville (`resolve_city_key`), avec fallback région
   (`config.REGIONS_FR`) si la commune n'a pas de modèle dédié.
2. `ign_fetch.py` — récupération du bâtiment matché (cascade de stratégies)
   + orthophoto IGN.
3. `patch_extraction.py` — découpe et normalisation min-max [0,1] d'un
   patch 64×64 sur le toit.
4. `inference.py` — `AlbedoPredictor` charge le checkpoint `.pt` de la
   ville (encoder MAE + `AlbedoHead`) et prédit l'albédo brut.
5. `materials.py` — classification du matériau (HistGradientBoosting, cf.
   §4) sur le même patch.
6. `shadow_calibration.py` — correction de l'albédo selon le matériau
   détecté (facteur multiplicatif, `SHADOW_CORRECTION_FACTOR`).
7. `confidence.py` — score de confiance combinant qualité du matching
   bâtiment, proba du classifieur matériau, distance au centre-ville.
8. `zone_scan.py` — variante "scan de zone" : agrège les bâtiments autour
   d'une adresse, détecte les îlots de chaleur (toits sombres isolés parmi
   des toits clairs et inversement) via `DARK_ROOF_ALBEDO_THRESHOLD` /
   `MIN_DARK_NEIGHBORS` / `MIN_BRIGHT_NEIGHBORS`.

## 2. Frontend — **tranché le 14/09/2026 : FastAPI + HTML est le produit**

`main.py` (**FastAPI**) + `static/index.html` est désormais l'unique
frontend maintenu :

- API REST : `/api/single`, `/api/batch`, `/api/batch.csv`, `/api/zone`,
  `/api/zone.csv`, `/api/zone/stream` (Server-Sent Events pour le retour
  progressif d'un scan de zone).
- `static/index.html` — frontend JS autonome qui consomme cette API,
  servi par FastAPI lui-même (`StaticFiles(..., html=True)`, monté en
  `/`).

L'ancienne app **Streamlit** (`app.py` — geocodage → carte `pydeck` →
image du toit → albédo, réimportant directement les mêmes modules Python
que le backend : `inference`, `materials`, `pipeline`, `zone_scan`, etc.)
est **archivée dans `legacy/app.py`**, pas supprimée (cf.
`legacy/README.md` pour le détail du choix et comment la relancer si
besoin ponctuel). Elle n'est plus maintenue : toute nouvelle feature UI
(score de confiance, filtre par matériau, etc.) ne doit être écrite que
côté `static/index.html` + `main.py`.

**Ne pas confondre avec `data_collection/annotation_tool.py`** : cet
outil utilise aussi Streamlit, mais c'est un outil interne d'annotation
manuelle de données (labellisation matériaux pour construire le dataset
d'entraînement), pas un frontend produit. Il n'est pas concerné par cet
archivage et continue de vivre à la racine de `data_collection/`.

Les nombreux commentaires dans les modules partagés
(`geocoding.py`, `ign_fetch.py`, `zone_scan.py`, `pipeline.py`,
`patch_extraction.py`, `inference.py`, ...) qui décrivent encore un point
d'entrée comme "utilisé depuis `app.py`" désignent l'**usage historique**
documenté au moment de l'écriture ; ces modules sont désormais consommés
par `main.py`, pas par `app.py`. Le comportement du code n'a pas changé
par ce déplacement (aucun import cassé — `main.py` n'a jamais dépendu de
`app.py`), seuls les commentaires n'ont pas été réécrits systématiquement.

## 3. Étanchéité des splits train/val/test — audité en CI

`tests/test_split_isolation.py` tourne en CI
(`.github/workflows/data-and-thresholds-audit.yml`, déclenché sur tout
changement touchant `datasets/materials_v1.parquet`,
`build_training_dataset.py`, `train_material_classifier.py`,
`config.py`, `heat_island_calibration.py`, `confidence.py`,
`materials.py`, `shadow_calibration.py`, `city_registry.py`,
`tests/**`). Il vérifie notamment :

- absence de recouvrement de `group_id` entre train/val/test (fuite de
  données) ;
- couverture des villes (`EXPECTED_CITY_COUNT = 15` — dataset matériaux,
  différent des 14 `CITY_MODELS` : voir §5 sur l'écart de vocabulaire) ;
- qu'aucune ville "splittable" (assez de groupes) ne soit totalement
  absente de train ;
- un signal non-bloquant (xfail informatif) sur les villes trop pauvres en
  échantillons pour être correctement splittées.

Ceci remplace l'ancien audit manuel ponctuel qui avait détecté la
contamination du held-out set — ce n'est plus une relecture à refaire à
chaque fois, la CI casse automatiquement en cas de régression.

## 4. Classification du matériau de toiture

- **v1 → v2.1** : heuristique couleur (distance aux centroïdes RGB),
  abandonnée (comportement dégradé documenté dans l'historique git de
  `materials.py`).
- **v3.0** (22/07/2026) : remplacement par un vrai classifieur ML
  (RandomForest sur features tabulaires GLCM + couleur,
  `training/features.py`, `training/train_material_classifier.py`,
  `training/model_registry.json`). Chargement mis en cache
  module-level (verrou `threading.Lock`, même pattern que
  `inference.py::_predictor_cache`), avec vérification du hash SHA256 du
  `.joblib` contre le registre au démarrage — refus de démarrer si
  incohérent.
- **v3.1** (23/07/2026) : swap vers `HistGradientBoostingClassifier`
  (`training/train_material_classifier_hgb.py`), meilleur F1 sur la
  classe béton (0.36 → 0.42) sur le même split test. Modèle et registre
  dans des fichiers dédiés
  (`models/material_classifier_v1_hgb.joblib`,
  `training/model_registry_hgb.json`) pour garder le RandomForest
  disponible en comparaison.
- Le module expose `MaterialResult.proba_max`, utilisé par `confidence.py`
  pour pénaliser les prédictions peu sûres.

## 5. Centralisation des clés/seuils par ville et par matériau

Historique du problème : plusieurs bugs distincts (calibrateur isotonique
mal routé, vocabulaire de clés désynchronisé entre `shadow_calibration.py`
et `materials.py`, seuils îlot de chaleur mal appliqués) venaient tous du
même symptôme — des constantes dupliquées dans plusieurs modules, sans
mécanisme forçant leur synchronisation. Trois mécanismes distincts
adressent chacun un aspect du problème, **ils ne se recouvrent pas** :

- **`EXPECTED_MATERIAL_KEYS`** (`shadow_calibration.py`) : source de
  vérité pour le vocabulaire des matériaux utilisé dans
  `SHADOW_CORRECTION_FACTOR`. Un `.get(cle, DEFAULT)` mal synchronisé
  retombe silencieusement sur une valeur par défaut sans lever d'erreur —
  c'est exactement le bug déjà rencontré une fois sur ce module.
- **`city_registry.py`** : étend le même principe aux **clés de ville**,
  dupliquées à travers `config.py` (source canonique de facto —
  `CITY_MODELS`, la liste des checkpoints réellement entraînés),
  `heat_island_calibration.py` et `confidence.py`. Ce module ne redéfinit
  aucune clé : il importe `CANONICAL_CITY_KEYS` depuis `config.CITY_MODELS`
  et documente les écarts **volontaires** entre modules via des exceptions
  explicites (`CITIES_WITHOUT_HEAT_ISLAND_CALIBRATION`,
  `EXTRA_SCRAPING_ONLY_CITY_KEYS`, `CITIES_WITHOUT_APPROX_CENTER`) —
  tout écart non documenté ici fait échouer
  `tests/test_city_key_consistency.py` en CI.
- **`threshold_guard.py`** : garde-fou orthogonal, pour les **valeurs** de
  seuils métier (pas les clés). Chaque module de seuils déclare
  `THRESHOLDS_VERSION` + `THRESHOLDS_HASH` (hash figé de ses constantes) ;
  un test recalcule le hash et échoue si quelqu'un change un seuil sans
  mise à jour explicite de la version. Ça ne valide pas qu'un seuil est
  correct, ça garantit juste qu'un changement est toujours visible en
  review, jamais un effet de bord silencieux.
- **`tests/test_confidence_materials_sync.py`** : audit bloquant en CI de
  la synchro `confidence.py` ↔ `materials.py` (remplace un ancien
  `logger.warning` au chargement, invisible tant que personne ne
  regardait les logs).

**Point d'attention pour toute nouvelle ville ou tout nouveau matériau** :
il faut mettre à jour `CITY_MODELS` (ou `EXPECTED_MATERIAL_KEYS`) **et**
documenter explicitement toute exception dans `city_registry.py`, sinon la
CI casse (`test_city_key_consistency.py` / `test_confidence_materials_sync.py`).
C'est le comportement voulu, pas un bug de la CI.

## 6. Non-régression par ville (checkpoints)

`tests/test_city_model_smoke.py` : un test **paramétrisé par ville** (pas
une boucle dans un seul test) — charge le checkpoint, infère sur un patch
fixe, vérifie une sortie dans une plage attendue. Un checkpoint qui charge
mal sur une ville (mismatch `encoder_state_dict`/`head_state_dict`,
mauvais `norm_type` LayerNorm/BatchNorm, `head_hidden` incorrect — bugs
déjà rencontrés, cf. commentaires dans `inference.py`) est détecté
individuellement : les 13 autres villes restent visibles comme PASSED
dans le rapport CI, au lieu qu'un échec masque l'état de tout le lot.

`training/model_registry.json` (et son `.bak`) / `model_registry_hgb.json`
suivent séparément la version du classifieur matériau (hash `.joblib`,
cf. §4) — ce n'est pas le même mécanisme que les checkpoints `.pt` par
ville.

## 7. Points explicitement obsolètes / corrigés depuis

- La **HeatmapLayer** dans `static/index.html`, précédemment documentée
  comme "abandonnée mais restée active" (code mort exécuté malgré tout),
  a été retirée — aucune référence à `HeatmapLayer`/`Heatmap` ne subsiste
  dans `static/index.html` au moment de la rédaction de ce document. Si un
  futur diff la réintroduit, vérifier qu'elle est réellement utilisée
  avant de merger.
- "beauce" a été retirée de `config.CITY_MODELS` et de tous les fallbacks
  (09/07/2026) : tête de régression dégénérée, sortie quasi nulle quelle
  que soit l'entrée. Ne pas la réintroduire sans ré-entraînement.
- L'heuristique couleur pour la classification du matériau (v1→v2.1) est
  remplacée par un classifieur ML depuis la v3.0 (§4) — toute
  documentation ou code faisant encore référence à
  `confidence="heuristique_couleur"` est obsolète : la valeur actuelle est
  `"ml_classifieur_v1_hgb"`.

## 8. Ce qui reste ouvert (par ordre de coût/bénéfice décroissant, cf. discussion produit du 14/09/2026)

1. ~~Trancher le frontend unique~~ — fait le 14/09/2026 : FastAPI +
   `static/index.html` retenu, `app.py` (Streamlit) archivé dans
   `legacy/` (§2). Point de vigilance résiduel : les commentaires
   "utilisé depuis app.py" dans les modules partagés ne sont pas tous
   réécrits — à corriger au fil de l'eau si on touche ces zones, pas
   urgent.
2. ~~Retrouver/committer le backend FastAPI manquant~~ — fait, `main.py`
   est dans le dépôt et sert bien `/api/zone/stream` + `static/`.
3. ~~Audit d'étanchéité des splits en CI~~ — fait (§3).
4. ~~Centralisation clés/seuils~~ — fait pour villes et matériaux (§5) ;
   à étendre si de nouvelles familles de constantes dupliquées
   apparaissent (le principe — une source canonique + des exceptions
   documentées + un test qui casse — est réutilisable tel quel).
5. ~~Tests de non-régression par ville~~ — fait (§6).
6. Ce document — créé pour la première fois le 14/09/2026 (n'existait pas
   avant dans le dépôt versionné). À rouvrir dès qu'une décision de §8.1
   est prise, ou qu'un mécanisme de §5 est étendu à une nouvelle classe de
   constantes.
