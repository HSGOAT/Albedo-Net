"""
data_collection/osm_material_scraper.py — AlbedoNet App
=========================================================
v2 (23/07/2026) — RÉÉCRITURE : abandon de l'API Overpass en direct.

CONTEXTE DU CHANGEMENT : la v1 de ce script (basée sur des requêtes HTTP
successives vers overpass-api.de / overpass.kumi.systems, une par ville)
se heurtait systématiquement à un 429 Too Many Requests, y compris dès la
toute première ville et malgré un délai de 15s entre requêtes. Le vrai
problème n'était pas le rythme mais le NOMBRE de requêtes HTTP distinctes
(13-14, une par ville) contre un service public à quota global par IP.

NOUVELLE APPROCHE : téléchargement préalable (par toi, en local) d'un
extrait OSM au format .osm.pbf depuis Geofabrik (https://download.geofabrik.de/,
ex. "france-latest.osm.pbf" ou un extrait régional plus léger), puis lecture
et filtrage 100% LOCAL via la librairie `pyrosm` -- aucune requête réseau
répétée, tout se passe en mémoire après le chargement initial du fichier.

PRÉREQUIS :
    pip install pyrosm pandas
    Télécharger un ou plusieurs extraits .osm.pbf depuis
    https://download.geofabrik.de/europe/france.html.

    IMPORTANT si tes villes sont dispersées sur toute la France (pas
    concentrées dans une seule région) : NE PAS utiliser le fichier national
    "france-latest.osm.pbf" pour tout traiter en un coup -- même avec la
    bounding box automatique de ce script (cf. plus bas), un run couvrant
    des villes aux deux bouts du pays produit une bbox qui englobe presque
    toute la France, ce qui redonne le même MemoryError qu'un run sans bbox.
    Procédure recommandée : télécharger un extrait RÉGIONAL par région
    couverte (ex. "ile-de-france-latest.osm.pbf",
    "provence-alpes-cote-d-azur-latest.osm.pbf",
    "auvergne-rhone-alpes-latest.osm.pbf", etc.), et lancer ce script UNE
    FOIS PAR RÉGION avec --villes limité aux villes de cette région et
    --out pointant vers un CSV différent à chaque fois. Concaténer les CSV
    obtenus à la fin (pandas.concat ou simple copier-coller des lignes hors
    en-tête).

USAGE :
    python data_collection/osm_material_scraper.py \
        --pbf chemin/vers/france-latest.osm.pbf \
        --out data_collection/osm_materials_raw.csv \
        --villes paris marseille lyon toulouse ...
    (par defaut : toutes les villes de confidence.CITY_CENTERS_APPROX)

SORTIE : CSV colonnes
    osm_id, osm_type, city_key, lat, lon, material_raw_tag, material
(même format qu'avant -- build_training_dataset.py n'a pas besoin de changer.)
Dedoublonnage : sur (osm_type, osm_id).

Tags OSM lus : `roof:material` en priorite, fallback `building:material`
si le premier est absent. Mapping materiau OSM -> classes internes
(MATERIALS de materials.py) : voir OSM_MATERIAL_MAPPING ci-dessous. Toute
valeur OSM non reconnue est ignoree (pas de label, pas de ligne dans le
CSV) plutot que force dans une classe -- on ne veut pas de faux labels.

LIMITE CONNUE : `pyrosm` n'expose pas directement le type d'élément OSM
(way vs relation) de façon fiable dans `get_data_by_custom_criteria` --
`osm_type` est donc renseigné en dur à "way" (l'écrasante majorité des
bâtiments tagués sont des ways ; les rares relations multipolygones sont
donc mal étiquetées sur cette seule colonne, mais leur `osm_id` reste
correct pour le dédoublonnage puisque way_id et relation_id ne se
chevauchent pas dans la pratique OSM courante).
"""

from __future__ import annotations

import argparse
import sys
import time
from math import cos, radians
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from pyrosm import OSM
except ImportError:
    print(
        "ERREUR : pyrosm n'est pas installé. `pip install pyrosm`.",
        file=sys.stderr,
    )
    raise

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from confidence import CITY_CENTERS_APPROX  # dict[str, tuple[lat, lon]]
except ImportError:
    CITY_CENTERS_APPROX = None
    print(
        "ERREUR : impossible d'importer confidence.CITY_CENTERS_APPROX. "
        "C'est la seule source de centres ville disponible dans ce depot "
        "pour rattacher un bâtiment à une ville -- lance ce script depuis "
        "la racine du projet.",
        file=sys.stderr,
    )

# Mapping valeurs OSM -> classes MATERIALS (materials.py). Cle = valeur du
# tag OSM en minuscule. Inchangé par rapport à la v1 (Overpass).
OSM_MATERIAL_MAPPING: dict[str, str] = {
    "zinc": "zinc",
    "metal_sheet": "zinc",  # tôle métallique plate/ondulée, visuellement proche du zinc
    "tin": "zinc",          # même famille visuelle que le zinc
    "metal": "zinc",        # AJOUT 23/07/2026 -- decision assumee malgre le risque
    # documente precedemment : "metal" est un tag OSM generique qui couvre
    # aussi bien de la tole grise que du bac acier peint (hangars
    # industriels rouges/verts). Reintegre car le plateau du zinc apres
    # 10 villes scrapees (64/300) rend le compromis necessaire. CONSEQUENCE
    # ATTENDUE : la classe zinc va probablement absorber des faux positifs
    # visuellement heterogenes (couleurs non-zinc). A surveiller au moment
    # de l'entrainement / calibration albedo -- si la variance intra-classe
    # de zinc explose ou si les predictions albedo pour "zinc" deviennent
    # incoherentes, revenir sur cette decision (retirer "metal" a nouveau).
    # "copper" (cuivre patiné, vert céladon) reste volontairement EXCLU --
    # trop distinct visuellement, pas de justification de plateau pour lui.
    "slate": "ardoise",
    "ardoise": "ardoise",
    "tile": "tuile_terre_cuite",
    "roof_tiles": "tuile_terre_cuite",
    "tiles": "tuile_terre_cuite",
    "terracotta": "tuile_terre_cuite",
    "concrete": "beton",
    "beton": "beton",
    "béton": "beton",
    "concrete_block": "beton",     # bloc béton, meme famille grise/texturee
    "reinforced_concrete": "beton",  # beton arme, visuellement identique au beton simple
    "cement_block": "beton",       # parpaing/bloc ciment, tres proche visuellement
    # NB : "fibre_cement"/"fibrociment" est volontairement EXCLU malgre une
    # couleur grise proche -- c'est un materiau composite different (plaques
    # ondulees, texture distincte), pas du beton coule. A rouvrir seulement
    # si le diagnostic des tags rejetes (cf. discussion precedente) montre
    # un volume significatif a recuperer specifiquement sur ce materiau.
}

# Rayon (metres) autour du centre-ville pour rattacher un bâtiment à une
# ville. Un bâtiment hors de ce rayon pour TOUTES les villes demandées est
# ignoré (pas de ville "la plus proche mais très loin").
DEFAULT_RADIUS_M = 8000


def _map_material_series(raw_tags: pd.Series) -> pd.Series:
    """Version vectorisee de l'ancien _map_material() : normalise (strip/lower) et
    mappe via OSM_MATERIAL_MAPPING en une seule passe pandas, au lieu d'un appel
    Python par ligne. Retourne NaN pour tout tag absent ou non reconnu.
    """
    normalized = raw_tags.astype("string").str.strip().str.lower()
    return normalized.map(OSM_MATERIAL_MAPPING)


def _nearest_city_vectorized(
    lat: np.ndarray, lon: np.ndarray, city_centers: dict[str, tuple[float, float]], max_radius_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Equivalent vectorise de l'ancien _nearest_city() applique ligne par ligne.

    Calcule la distance haversine de CHAQUE batiment vers CHAQUE ville d'un coup
    (matrice n_batiments x n_villes via broadcasting numpy), au lieu d'une boucle
    Python imbriquee (n_batiments * n_villes appels de fonction). Sur une region
    dense (des centaines de milliers de batiments), c'est la difference entre
    quelques secondes et plusieurs minutes.

    Retourne (city_idx, within_radius) : indice de la ville la plus proche pour
    chaque batiment (dans l'ordre de city_centers) et masque booleen indiquant
    si cette ville la plus proche est bien dans max_radius_m.
    """
    r_earth = 6_371_000.0
    city_names = list(city_centers.keys())
    city_lat = np.array([c[0] for c in city_centers.values()])
    city_lon = np.array([c[1] for c in city_centers.values()])

    lat_r = np.radians(lat)[:, None]          # (n_batiments, 1)
    lon_r = np.radians(lon)[:, None]
    clat_r = np.radians(city_lat)[None, :]    # (1, n_villes)
    clon_r = np.radians(city_lon)[None, :]

    dlat = clat_r - lat_r
    dlon = clon_r - lon_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat_r) * np.cos(clat_r) * np.sin(dlon / 2) ** 2
    dist_m = 2 * r_earth * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # (n_batiments, n_villes)

    nearest_idx = dist_m.argmin(axis=1)
    nearest_dist = dist_m[np.arange(len(lat)), nearest_idx]
    within = nearest_dist <= max_radius_m
    return nearest_idx, within


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pbf", type=Path, required=True, help="chemin vers l'extrait .osm.pbf (Geofabrik)")
    parser.add_argument("--out", type=Path, default=Path("data_collection/osm_materials_raw.csv"))
    parser.add_argument("--villes", nargs="*", default=None, help="cles CITY_CENTERS_APPROX a filtrer (defaut: toutes)")
    parser.add_argument("--radius-m", type=int, default=DEFAULT_RADIUS_M)
    args = parser.parse_args()

    if CITY_CENTERS_APPROX is None:
        print("ERREUR : CITY_CENTERS_APPROX introuvable, impossible de continuer.", file=sys.stderr)
        sys.exit(1)

    if not args.pbf.exists():
        print(f"ERREUR : fichier .osm.pbf introuvable : {args.pbf}", file=sys.stderr)
        sys.exit(1)

    villes = args.villes or list(CITY_CENTERS_APPROX.keys())
    city_centers: dict[str, tuple[float, float]] = {}
    for v in villes:
        center = CITY_CENTERS_APPROX.get(v)
        if center is None:
            print(f"AVERTISSEMENT : ville inconnue dans CITY_CENTERS_APPROX: {v}, ignoree.", file=sys.stderr)
            continue
        city_centers[v] = center

    if not city_centers:
        print("ERREUR : aucune ville valide a traiter.", file=sys.stderr)
        sys.exit(1)

    # Bounding box calculée à partir des villes demandées (+ rayon), pour
    # limiter la mémoire nécessaire au chargement -- utile même sur un
    # extrait régional, et quasi indispensable sur un extrait national.
    # 1 degre lat ~= 111_320 m ; 1 degre lon ~= 111_320 * cos(lat) m.
    lats = [c[0] for c in city_centers.values()]
    lons = [c[1] for c in city_centers.values()]
    margin_lat_deg = args.radius_m / 111_320.0
    mean_lat = sum(lats) / len(lats)
    margin_lon_deg = args.radius_m / (111_320.0 * max(cos(radians(mean_lat)), 1e-6))
    bbox = [
        min(lons) - margin_lon_deg,  # min lon
        min(lats) - margin_lat_deg,  # min lat
        max(lons) + margin_lon_deg,  # max lon
        max(lats) + margin_lat_deg,  # max lat
    ]
    print(f"Bounding box calculee pour les villes demandees : {bbox}")

    print(f"Chargement de l'extrait OSM local : {args.pbf} (peut prendre plusieurs minutes selon la taille)...")
    osm = OSM(str(args.pbf), bounding_box=bbox)

    print("Extraction des bâtiments avec roof:material / building:material...")
    buildings = osm.get_data_by_custom_criteria(
        custom_filter={"building": True},
        extra_attributes=["roof:material", "building:material"],
        keep_nodes=False,
        keep_ways=True,
        keep_relations=True,
    )

    if buildings is None or len(buildings) == 0:
        print("Aucun bâtiment trouvé dans l'extrait (verifie le fichier .osm.pbf et son emprise geographique).")
        rows: list[dict] = []
    else:
        print(f"{len(buildings)} bâtiments (avec tag building) chargés depuis l'extrait, filtrage en cours...")

        # Tag brut : roof:material en priorite, fallback building:material.
        raw_tag = buildings.get("roof:material")
        if raw_tag is None:
            raw_tag = pd.Series([None] * len(buildings), index=buildings.index)
        fallback = buildings.get("building:material")
        if fallback is not None:
            raw_tag = raw_tag.fillna(fallback)
        buildings = buildings.assign(_raw_tag=raw_tag)

        # Mapping materiau vectorise -> on jette tout de suite les lignes sans materiau reconnu.
        buildings = buildings.assign(_material=_map_material_series(buildings["_raw_tag"]))
        buildings = buildings[buildings["_material"].notna()]

        # Centroides (geopandas vectorise en bulk plutot qu'un .centroid par ligne).
        buildings = buildings[buildings.geometry.notna() & ~buildings.geometry.is_empty]
        centroids = buildings.geometry.centroid
        lon = centroids.x.to_numpy()
        lat = centroids.y.to_numpy()

        # Ville la plus proche, vectorise (matrice n_batiments x n_villes).
        nearest_idx, within = _nearest_city_vectorized(lat, lon, city_centers, args.radius_m)
        city_names = list(city_centers.keys())
        city_key = np.array(city_names, dtype=object)[nearest_idx]

        osm_id = buildings["id"].to_numpy() if "id" in buildings.columns else np.arange(len(buildings))

        df_rows = pd.DataFrame({
            "osm_id": osm_id,
            "osm_type": "way",  # cf. LIMITE CONNUE dans le docstring
            "city_key": city_key,
            "lat": lat,
            "lon": lon,
            "material_raw_tag": buildings["_raw_tag"].to_numpy(),
            "material": buildings["_material"].to_numpy(),
        })

        # Ne garder que les batiments dont la ville la plus proche est bien dans le rayon.
        df_rows = df_rows[within]
        # Dedoublonnage sur (osm_type, osm_id).
        df_rows = df_rows.drop_duplicates(subset=["osm_type", "osm_id"])
        rows = df_rows.to_dict("records")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "osm_id", "osm_type", "city_key", "lat", "lon", "material_raw_tag", "material",
    ])
    df.to_csv(args.out, index=False)

    print(f"\nTotal : {len(df)} batiments labellises exportes vers {args.out}")
    if not df.empty:
        print(df["material"].value_counts().to_string())
        print("\nPar ville :")
        print(df["city_key"].value_counts().to_string())


if __name__ == "__main__":
    main()