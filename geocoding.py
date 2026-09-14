"""
geocoding.py — AlbedoNet App
==============================
Adresse (texte libre) -> lat/lon + resolution de la cle ville a utiliser
pour choisir le checkpoint .pt (cf. config.py).

Strategie de matching, du plus precis au moins precis :
  1. Match direct par nom de commune renvoye par l'API BAN
     (config.COMMUNE_TO_CITY_KEY).
  2. Si pas de match direct (ou score BAN trop faible / pas de resultat) :
     deduire le departement depuis le code INSEE (citycode), puis la region
     depuis le departement, puis chercher les villes couvertes dans cette
     region (config.REGION_TO_CITY_KEYS).
  3. Si l'API BAN ne renvoie rien du tout (adresse introuvable) : on ne peut
     pas deduire de departement -> on demande explicitement a l'utilisateur
     de choisir sa region dans une liste (config.REGIONS_FR), geree cote
     frontend (static/index.html, via l'API exposee par main.py -- ce
     module expose juste resolve_city_key_from_region()).
  4. Si la region choisie n'a aucun modele associe : fallback generique
     (config.GENERIC_FALLBACK_MODEL).

ASCII uniquement dans le code (contrainte Pablo).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import requests

from config import COMMUNE_TO_CITY_KEY, REGION_TO_CITY_KEYS

BAN_SEARCH_URL = "https://api-adresse.data.gouv.fr/search/"
BAN_REVERSE_URL = "https://api-adresse.data.gouv.fr/reverse/"

# En dessous de ce score BAN (0 a 1), on considere le geocodage trop
# incertain pour faire confiance au nom de commune retourne.
BAN_MIN_SCORE = 0.5


# ──────────────────────────────────────────────────────────────────────────
# 1. Table departement (code INSEE, 2 caracteres) -> region administrative
# ──────────────────────────────────────────────────────────────────────────
# France metropolitaine uniquement (pas de DOM-TOM dans les checkpoints
# actuels). Noms de region sans accents pour matcher config.REGION_TO_CITY_KEYS.

DEPT_TO_REGION: dict[str, str] = {
    # Auvergne-Rhone-Alpes
    "01": "Auvergne-Rhone-Alpes", "03": "Auvergne-Rhone-Alpes",
    "07": "Auvergne-Rhone-Alpes", "15": "Auvergne-Rhone-Alpes",
    "26": "Auvergne-Rhone-Alpes", "38": "Auvergne-Rhone-Alpes",
    "42": "Auvergne-Rhone-Alpes", "43": "Auvergne-Rhone-Alpes",
    "63": "Auvergne-Rhone-Alpes", "69": "Auvergne-Rhone-Alpes",
    "73": "Auvergne-Rhone-Alpes", "74": "Auvergne-Rhone-Alpes",
    # Bourgogne-Franche-Comte
    "21": "Bourgogne-Franche-Comte", "25": "Bourgogne-Franche-Comte",
    "39": "Bourgogne-Franche-Comte", "58": "Bourgogne-Franche-Comte",
    "70": "Bourgogne-Franche-Comte", "71": "Bourgogne-Franche-Comte",
    "89": "Bourgogne-Franche-Comte", "90": "Bourgogne-Franche-Comte",
    # Bretagne
    "22": "Bretagne", "29": "Bretagne", "35": "Bretagne", "56": "Bretagne",
    # Centre-Val de Loire
    "18": "Centre-Val de Loire", "28": "Centre-Val de Loire",
    "36": "Centre-Val de Loire", "37": "Centre-Val de Loire",
    "41": "Centre-Val de Loire", "45": "Centre-Val de Loire",
    # Corse
    "2A": "Corse", "2B": "Corse",
    # Grand Est
    "08": "Grand Est", "10": "Grand Est", "51": "Grand Est",
    "52": "Grand Est", "54": "Grand Est", "55": "Grand Est",
    "57": "Grand Est", "67": "Grand Est", "68": "Grand Est", "88": "Grand Est",
    # Hauts-de-France
    "02": "Hauts-de-France", "59": "Hauts-de-France",
    "60": "Hauts-de-France", "62": "Hauts-de-France", "80": "Hauts-de-France",
    # Ile-de-France
    "75": "Ile-de-France", "77": "Ile-de-France", "78": "Ile-de-France",
    "91": "Ile-de-France", "92": "Ile-de-France", "93": "Ile-de-France",
    "94": "Ile-de-France", "95": "Ile-de-France",
    # Normandie
    "14": "Normandie", "27": "Normandie", "50": "Normandie",
    "61": "Normandie", "76": "Normandie",
    # Nouvelle-Aquitaine
    "16": "Nouvelle-Aquitaine", "17": "Nouvelle-Aquitaine",
    "19": "Nouvelle-Aquitaine", "23": "Nouvelle-Aquitaine",
    "24": "Nouvelle-Aquitaine", "33": "Nouvelle-Aquitaine",
    "40": "Nouvelle-Aquitaine", "47": "Nouvelle-Aquitaine",
    "64": "Nouvelle-Aquitaine", "79": "Nouvelle-Aquitaine",
    "86": "Nouvelle-Aquitaine", "87": "Nouvelle-Aquitaine",
    # Occitanie
    "09": "Occitanie", "11": "Occitanie", "12": "Occitanie",
    "30": "Occitanie", "31": "Occitanie", "32": "Occitanie",
    "34": "Occitanie", "46": "Occitanie", "48": "Occitanie",
    "65": "Occitanie", "66": "Occitanie", "81": "Occitanie", "82": "Occitanie",
    # Pays de la Loire
    "44": "Pays de la Loire", "49": "Pays de la Loire",
    "53": "Pays de la Loire", "72": "Pays de la Loire", "85": "Pays de la Loire",
    # Provence-Alpes-Cote d'Azur
    "04": "Provence-Alpes-Cote d'Azur", "05": "Provence-Alpes-Cote d'Azur",
    "06": "Provence-Alpes-Cote d'Azur", "13": "Provence-Alpes-Cote d'Azur",
    "83": "Provence-Alpes-Cote d'Azur", "84": "Provence-Alpes-Cote d'Azur",
}


# ──────────────────────────────────────────────────────────────────────────
# 2. Normalisation de nom de commune
# ──────────────────────────────────────────────────────────────────────────

def normalize_city_name(name: str) -> str:
    """
    "Clermont-Ferrand" -> "clermontferrand"
    "Paris 15e Arrondissement" -> "paris15earrondissement" (voir _strip_arrondissement)
    """
    name = _strip_arrondissement(name)
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(c for c in nfkd if not unicodedata.combining(c))
    return "".join(ch for ch in ascii_only.lower() if ch.isalnum())


def _strip_arrondissement(name: str) -> str:
    """
    Les grandes villes decoupees en arrondissements (Paris, Lyon, Marseille)
    sont renvoyees par l'API BAN sous la forme "Paris 15e Arrondissement".
    On garde juste le nom de la ville mere pour matcher COMMUNE_TO_CITY_KEY.
    """
    lowered = name.lower()
    for marker in ("arrondissement", "1er arrondissement"):
        idx = lowered.find(marker)
        if idx != -1:
            # Retire aussi le numero juste avant ("15e", "1er", etc.)
            before = name[:idx]
            words = before.split()
            # enleve le dernier mot type "15e"/"1er" s'il y en a un
            if words and (words[-1][:-1].isdigit() or words[-1].lower() in ("1er",)):
                words = words[:-1]
            return " ".join(words).strip()
    return name


def department_from_citycode(citycode: str) -> str | None:
    """
    citycode INSEE : 5 caracteres. Les 2 premiers = departement, SAUF Corse
    (2A/2B) et les codes commencant par "97"/"98" (DOM-TOM, non geres ici).
    """
    if not citycode or len(citycode) < 2:
        return None
    if citycode.startswith("97") or citycode.startswith("98"):
        return None  # DOM-TOM non couverts par les checkpoints actuels
    prefix = citycode[:2]
    if prefix == "20":
        # Corse : distinction 2A/2B via le 3e caractere du citycode
        return "2A" if citycode[2] < "5" else "2B"
    return prefix


# ──────────────────────────────────────────────────────────────────────────
# 3. Appel API BAN
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class GeocodeResult:
    found: bool
    address_label: str | None = None
    lat: float | None = None
    lon: float | None = None
    city: str | None = None          # nom de commune brut renvoye par la BAN
    citycode: str | None = None      # code INSEE
    postcode: str | None = None
    score: float | None = None       # confiance BAN, 0 a 1


def geocode_address(address: str, timeout: float = 5.0) -> GeocodeResult:
    """
    Appelle l'API BAN (api-adresse.data.gouv.fr) et retourne le meilleur
    resultat. found=False si aucun resultat exploitable.
    """
    if not address or not address.strip():
        return GeocodeResult(found=False)

    try:
        resp = requests.get(
            BAN_SEARCH_URL,
            params={"q": address.strip(), "limit": 1},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return GeocodeResult(found=False)

    features = data.get("features", [])
    if not features:
        return GeocodeResult(found=False)

    best = features[0]
    props = best.get("properties", {})
    coords = best.get("geometry", {}).get("coordinates", [None, None])
    lon, lat = coords[0], coords[1]

    if lat is None or lon is None:
        return GeocodeResult(found=False)

    return GeocodeResult(
        found=True,
        address_label=props.get("label"),
        lat=lat,
        lon=lon,
        city=props.get("city"),
        citycode=props.get("citycode"),
        postcode=props.get("postcode"),
        score=props.get("score"),
    )


# ──────────────────────────────────────────────────────────────────────────
# 4. Resolution vers une cle ville (config.CITY_MODELS)
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class CityResolution:
    city_key: str | None                 # match direct et fiable, sinon None
    region: str | None                   # region deduite (si dispo)
    region_candidates: list[str]         # villes couvertes dans cette region
    needs_user_region: bool              # True -> le frontend doit demander la region


def reverse_geocode(lat: float, lon: float, timeout: float = 5.0) -> str | None:
    """
    Geocodage inverse (coordonnees -> adresse la plus proche) via l'API BAN.
    Utilise pour afficher une adresse lisible aux batiments detectes comme
    suspects ilot de chaleur/fraicheur dans zone_scan (on n'a que des
    centroides lat/lon a ce stade, pas d'adresse).

    Retourne None si aucune adresse n'est trouvee ou en cas d'erreur reseau
    (ne doit jamais lever d'exception -- best-effort, purement pour
    l'affichage).
    """
    try:
        resp = requests.get(
            BAN_REVERSE_URL,
            params={"lon": lon, "lat": lat},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    features = data.get("features", [])
    if not features:
        return None

    return features[0].get("properties", {}).get("label")


def resolve_city_key(result: GeocodeResult) -> CityResolution:
    """
    Tente de deduire la cle ville a utiliser depuis un GeocodeResult.

    - Si match direct fiable (nom de commune + score suffisant) -> city_key rempli.
    - Sinon, si on peut deduire une region (via citycode) avec au moins une
      ville couverte -> region + region_candidates remplis, city_key=None
      (le frontend doit alors demander a l'utilisateur de confirmer/affiner,
      ou choisir directement si un seul candidat).
    - Sinon (rien d'exploitable, ou region non couverte) -> needs_user_region=True,
      le frontend doit proposer la liste complete config.REGIONS_FR.
    """
    if not result.found:
        return CityResolution(
            city_key=None, region=None, region_candidates=[],
            needs_user_region=True,
        )

    # Etape 1 : match direct par nom de commune, seulement si score BAN fiable
    if result.city and (result.score is None or result.score >= BAN_MIN_SCORE):
        key = COMMUNE_TO_CITY_KEY.get(normalize_city_name(result.city))
        if key:
            return CityResolution(
                city_key=key, region=None, region_candidates=[],
                needs_user_region=False,
            )

    # Etape 2 : deduire la region depuis le departement (citycode)
    dept = department_from_citycode(result.citycode) if result.citycode else None
    region = DEPT_TO_REGION.get(dept) if dept else None

    if region:
        candidates = REGION_TO_CITY_KEYS.get(region, [])
        if candidates:
            return CityResolution(
                city_key=candidates[0] if len(candidates) == 1 else None,
                region=region,
                region_candidates=candidates,
                needs_user_region=len(candidates) > 1,
            )
        # region identifiee mais aucun modele dedie -> fallback generique,
        # pas besoin de redemander a l'utilisateur
        return CityResolution(
            city_key=None, region=region, region_candidates=[],
            needs_user_region=False,
        )

    # Etape 3 : rien d'exploitable -> demander explicitement la region
    return CityResolution(
        city_key=None, region=None, region_candidates=[],
        needs_user_region=True,
    )


def resolve_city_key_from_region(region: str) -> list[str]:
    """
    A utiliser cote frontend (via l'API main.py) une fois que l'utilisateur
    a choisi manuellement sa region dans la liste config.REGIONS_FR.
    Retourne la liste des cles ville couvertes (0, 1 ou plusieurs -- le
    frontend affiche un 2e select si >1).
    """
    return REGION_TO_CITY_KEYS.get(region, [])
