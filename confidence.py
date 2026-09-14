"""
confidence.py — AlbedoNet App
================================
Score de confiance composite par prediction, construit a partir de signaux
internes deja disponibles dans le pipeline (PAS une marge d'erreur calibree
sur de la verite terrain -- cf. METHODOLOGIE.md, "a faire pour lever le
statut prototype non valide", item 5).

Objectif : donner une indication honnete de fiabilite ("haute"/"moyenne"/
"faible") a cote de chaque albedo, en attendant un vrai MAE stratifie issu
de la vraie verite terrain (annotation manuelle + validation OSM, cf. suivi
projet 10/07/2026).

Signaux utilises (tous deja calcules ailleurs dans le pipeline, aucun calcul
couteux supplementaire) :
  1. Confiance du matching batiment (ign_fetch.BuildingMatch.confidence :
     "high"/"medium"/"none") -- disponible uniquement en mode adresse
     unique / lot. Le mode zone ne fait AUCUN matching adresse->batiment
     (les batiments viennent directement du WFS, pas d'un geocodage
     individuel) -- ce signal est donc absent (None) en mode zone, ce qui
     est normal et attendu, pas une degradation a corriger.
  2. plausibilite_materiau (postprocess_plausibility.py) : un batiment
     "suspecte_reclassification" a un albedo incoherent avec son materiau
     estime -- signal de confusion materiau probable, qui rejaillit sur la
     confiance globale de la ligne (pas seulement sur le materiau seul).
  3. Ratio de nodata dans le patch (patch_extraction.py) : un patch avec
     beaucoup de pixels nodata (bord de tuile/emprise) est un signal
     indirect de qualite image degradee, meme s'il reste sous le seuil de
     rejet (MAX_NODATA_RATIO=0.10) -- proche du seuil = moins fiable que 0%.
  4. Distance approximative au centre-ville de reference du modele utilise :
     signal TRES grossier (cf. CITY_CENTERS_APPROX ci-dessous, coordonnees
     approximatives non verifiees precisement) -- un modele "paris" applique
     a un batiment tres excentre (grande couronne, autre departement mal
     rattache) est statistiquement moins fiable qu'un batiment proche du
     coeur de la zone d'entrainement. Poids volontairement faible dans le
     score final.
  5. Probabilite du classifieur materiau (materials.py v3.0, RandomForest
     entraine, cf. training/model_registry.json) : une prediction avec un
     proba_max faible est techniquement tranchee mais peu sure -- remplace
     l'ancien signal "distance au centroide RGB le plus proche", obsolete
     depuis que le materiau n'est plus estime par heuristique couleur.

CE MODULE NE PRODUIT PAS UNE MARGE D'ERREUR STATISTIQUE CALIBREE. C'est une
heuristique de transparence ("le systeme est-il, sur la base de ce qu'il
sait de lui-meme, plutot confiant ou pas"), pas une probabilite validee. A
completer/recalibrer des que la vraie verite terrain (analyze_annotations.py)
permet de mesurer la correlation reelle entre ces signaux et l'erreur
observee -- cf. suivi projet 10/07/2026, "score d'incertitude par
prediction" (item critique n°2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from math import asin, cos, radians, sin, sqrt
from typing import Optional

logger = logging.getLogger("albedo.app.confidence")

# ──────────────────────────────────────────────────────────────────────────
# Ponderation des signaux -- calibree a l'oeil (meme logique de prudence
# que materials.py / heat_island_calibration.py / postprocess_plausibility.py
# : explicite, ajustable, PAS statistiquement validee).
# ──────────────────────────────────────────────────────────────────────────

# Points de penalite (score de depart = 100). Choix des valeurs : la
# confusion materiau (30) penalise plus que le nodata ou la distance
# centre-ville, car c'est le signal le plus directement relie a un vrai
# probleme de prediction observe sur les 4 scans du 10/07/2026 (cf.
# postprocess_plausibility.py). Le "medium" building match (15) est
# intermediaire : le batiment retenu est un candidat proche, pas exact,
# mais reste un match plausible.
PENALTY_BUILDING_MATCH_MEDIUM: int = 15
PENALTY_BUILDING_MATCH_NONE: int = 100  # garde-fou -- ne devrait pas arriver ici (pas d'albedo si "none")
PENALTY_MATERIAU_SUSPECT: int = 30
PENALTY_MATERIAU_SANS_REFERENCE: int = 5  # "indetermine" -- pas suspect, juste moins d'info disponible
PENALTY_NODATA_MAX: int = 20   # penalite si nodata_ratio est au maximum autorise (proche du seuil de rejet)
PENALTY_DISTANCE_CENTRE_VILLE: int = 10
DISTANCE_WARNING_KM: float = 20.0  # seuil approximatif pour un batiment considere comme tres excentre
PENALTY_MATERIAU_PROBA_FAIBLE: int = 15  # cf. materials.py v3.0 -- classifieur ML peu sur de sa prediction

# Seuil de probabilite (predict_proba, classifieur RandomForest de
# materials.py v3.0) en-dessous duquel une prediction materiau, bien que
# tranchee, est consideree peu fiable. DOIT rester coherent avec
# materials.MIN_RELIABLE_PROBA -- les deux existent separement (pas
# d'import croise pour eviter un couplage dur entre les deux modules),
# mais un changement de l'un doit s'accompagner d'une revue de l'autre.
MATERIAU_PROBA_WARNING: float = 0.6

# Verification runtime (pas un import croise dur au niveau module, pour ne
# pas creer de dependance de chargement entre confidence.py et materials.py --
# cf. discussion ci-dessus). Import local + try/except : si materials.py
# n'est pas encore chargeable (ex. modele .joblib absent, cf.
# materials._ModelCache._load), on ne veut pas faire planter confidence.py
# pour autant, ce module doit rester utilisable independamment. Le but est
# uniquement de detecter une desynchronisation SILENCIEUSE des deux
# constantes (bug potentiel deja signale dans la docstring ci-dessus).
def _check_materiau_proba_threshold_sync() -> None:
    try:
        from materials import MIN_RELIABLE_PROBA
    except Exception:
        logger.debug(
            "Impossible de verifier la coherence MATERIAU_PROBA_WARNING vs "
            "materials.MIN_RELIABLE_PROBA (import materials.py a echoue) -- "
            "ignore, ce module reste utilisable independamment."
        )
        return
    if MIN_RELIABLE_PROBA != MATERIAU_PROBA_WARNING:
        logger.warning(
            "DESYNCHRONISATION DETECTEE : confidence.MATERIAU_PROBA_WARNING=%.2f "
            "!= materials.MIN_RELIABLE_PROBA=%.2f. Ces deux constantes existent "
            "separement par choix (pas d'import croise dur), mais un changement "
            "de l'une doit s'accompagner d'une revue de l'autre (cf. commentaire "
            "au-dessus de MATERIAU_PROBA_WARNING). Verifier si ce changement est "
            "delibere ou si l'une des deux constantes a ete oubliee lors d'une "
            "mise a jour.",
            MATERIAU_PROBA_WARNING, MIN_RELIABLE_PROBA,
        )


_check_materiau_proba_threshold_sync()

# Bornes de score -> niveau. >= 80 haute, [55,80) moyenne, < 55 faible.
SCORE_THRESHOLDS: dict[str, int] = {"haute": 80, "moyenne": 55}

# Coordonnees APPROXIMATIVES du centre-ville de chaque cle CITY_MODELS (cf.
# config.py) -- pas une valeur officielle/precise, juste un point de
# reference grossier pour detecter un batiment tres excentre par rapport a
# la zone probable d'entrainement du modele. A NE PAS utiliser pour autre
# chose qu'un signal faible dans ce score composite.
CITY_CENTERS_APPROX: dict[str, tuple[float, float]] = {
    "paris": (48.8566, 2.3522),
    "lyon": (45.7640, 4.8357),
    "marseille": (43.2965, 5.3698),
    "toulouse": (43.6047, 1.4442),
    "nice": (43.7102, 7.2620),
    "nantes": (47.2184, -1.5536),
    "strasbourg": (48.5734, 7.7521),
    "montpellier": (43.6108, 3.8767),
    "lille": (50.6292, 3.0573),
    "grenoble": (45.1885, 5.7245),
    "clermont": (45.7772, 3.0870),
    "metz": (49.1193, 6.1757),
    "tours": (47.3941, 0.6848),
    # Ajouts (23/07/2026) -- villes ciblees specifiquement pour combler le
    # deficit d'echantillons zinc (cf. suivi projet), pas issues de
    # CITY_MODELS existants. A confirmer si elles doivent aussi devenir des
    # modeles CITY_MODELS a part entiere, ou rester des cles techniques
    # utilisees uniquement pour le scraping OSM.
    "rouen": (49.4432, 1.0999),
    "reims": (49.2583, 4.0317),
    "dijon": (47.3220, 5.0415),
    "bordeaux": (44.8378, -0.5792),
    "rennes": (48.1173, -1.6778),
    "amiens": (49.8941, 2.2958),
    "caen": (49.1829, -0.3707),
    # Ajouts (23/07/2026) -- villes ciblees pour l'expansion regionale
    # PACA / Nouvelle-Aquitaine / Bretagne (cf. runs osm_material_scraper.py
    # du 23/07). Meme statut que les ajouts precedents : cles techniques
    # pour le scraping OSM, coordonnees approximatives non verifiees
    # precisement, pas necessairement des CITY_MODELS a part entiere.
    "toulon": (43.1242, 5.9280),
    "aix_en_provence": (43.5297, 5.4474),
    "bayonne": (43.4929, -1.4748),
    "pau": (43.2951, -0.3708),
    "brest": (48.3904, -4.4861),
    # "provence_rurale" volontairement absent : pas une ville, pas de centre
    # ponctuel pertinent -- ce signal est simplement ignore pour ce modele
    # (city_key present mais absent de ce dict -> pas de penalite appliquee).
}


@dataclass
class ConfidenceResult:
    score: int  # 0-100, PAS une probabilite calibree
    niveau: str  # "haute" / "moyenne" / "faible"
    raisons: list[str] = field(default_factory=list)  # explications lisibles des penalites appliquees


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_r, lon1_r, lat2_r, lon2_r = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = sin(dlat / 2) ** 2 + cos(lat1_r) * cos(lat2_r) * sin(dlon / 2) ** 2
    return 6371.0 * 2 * asin(sqrt(a))


def compute_confidence(
    plausibilite_materiau: Optional[str],
    building_match_confidence: Optional[str] = None,
    nodata_ratio: Optional[float] = None,
    city_key: Optional[str] = None,
    building_lat: Optional[float] = None,
    building_lon: Optional[float] = None,
    material_proba_max: Optional[float] = None,
) -> ConfidenceResult:
    """Calcule un score de confiance composite (0-100) pour UNE prediction.

    Tous les arguments sont optionnels sauf plausibilite_materiau : chaque
    signal absent (None) est simplement ignore (pas de penalite ni bonus),
    pour que cette fonction reste utilisable a l'identique dans les 3 modes
    de l'app (adresse unique/lot : building_match_confidence dispo ; zone :
    absent -- cf. docstring module).
    """
    score = 100
    raisons: list[str] = []

    if building_match_confidence == "medium":
        score -= PENALTY_BUILDING_MATCH_MEDIUM
        raisons.append(
            f"batiment associe par proximite, pas par correspondance exacte "
            f"(-{PENALTY_BUILDING_MATCH_MEDIUM})"
        )
    elif building_match_confidence == "none":
        score -= PENALTY_BUILDING_MATCH_NONE
        raisons.append("aucun batiment fiable associe -- score invalide")

    if plausibilite_materiau == "suspecte_reclassification":
        score -= PENALTY_MATERIAU_SUSPECT
        raisons.append(
            f"albedo incoherent avec le materiau estime, probable confusion "
            f"materiau (-{PENALTY_MATERIAU_SUSPECT})"
        )
    elif plausibilite_materiau == "sans_reference":
        score -= PENALTY_MATERIAU_SANS_REFERENCE
        raisons.append(
            f"materiau indetermine, pas de plage de reference pour "
            f"sanity-check (-{PENALTY_MATERIAU_SANS_REFERENCE})"
        )

    if material_proba_max is not None and material_proba_max < MATERIAU_PROBA_WARNING:
        score -= PENALTY_MATERIAU_PROBA_FAIBLE
        raisons.append(
            f"classifieur materiau peu sur de sa prediction "
            f"(probabilite={material_proba_max:.0%}), risque de confusion "
            f"(-{PENALTY_MATERIAU_PROBA_FAIBLE})"
        )

    if nodata_ratio is not None and nodata_ratio > 0:
        penalty = round(PENALTY_NODATA_MAX * min(nodata_ratio, 1.0) / 0.10)
        if penalty > 0:
            score -= penalty
            raisons.append(f"{nodata_ratio:.0%} de pixels nodata dans le patch (-{penalty})")

    if city_key and building_lat is not None and building_lon is not None:
        center = CITY_CENTERS_APPROX.get(city_key)
        if center:
            dist_km = _haversine_km(building_lat, building_lon, center[0], center[1])
            if dist_km > DISTANCE_WARNING_KM:
                score -= PENALTY_DISTANCE_CENTRE_VILLE
                raisons.append(
                    f"batiment a ~{dist_km:.0f} km du centre-ville de reference du "
                    f"modele '{city_key}' (signal grossier, -{PENALTY_DISTANCE_CENTRE_VILLE})"
                )

    score = max(0, min(100, score))

    if score >= SCORE_THRESHOLDS["haute"]:
        niveau = "haute"
    elif score >= SCORE_THRESHOLDS["moyenne"]:
        niveau = "moyenne"
    else:
        niveau = "faible"

    if not raisons:
        raisons.append("aucun signal de degradation detecte")

    return ConfidenceResult(score=score, niveau=niveau, raisons=raisons)