"""
postprocess_plausibility.py — AlbedoNet App
==============================================
Post-traitement de plausibilité par materiau, construit suite au constat du
10/07/2026 sur les 4 gros scans reels (n=65904) : la categorie "ardoise" a
une distribution d'albedo predite anormale par rapport a
albedo_reference_table.md (plage vieilli/expose attendue : 0.08-0.20) :

  - Corps principal (~76% des cas, albedo 0.19-0.28, mediane 0.224) :
    legerement au-dessus de la plage attendue, sans rien d'aberrant.
  - Queue haute (~11.4% des cas, albedo 0.30-0.52) : chevauche completement
    le territoire beton/tuile_terre_cuite -- tres probablement une
    MAUVAISE CLASSIFICATION MATERIAU (materials.py), pas un biais d'albedo.

CE MODULE NE CORRIGE PAS AVEUGLEMENT L'ALBEDO. Deux fonctions distinctes,
avec des garanties differentes :

  1. flag_material_plausibility() -- DETECTION uniquement, n'altere JAMAIS
     albedo ni materiau_estime. Ajoute une colonne `plausibilite_materiau`
     ("normale" / "suspecte_reclassification") pour les cas ou l'albedo
     predit est largement hors de la plage bibliographique du materiau
     estime -- signal fort de confusion materiau (ex. ardoise vs beton),
     PAS une correction. A utiliser pour prioriser une revue manuelle
     ciblee (cf. discussion "validation sans terrain", option 4 : revue
     experte legere sur les cas suspects plutot qu'un echantillon aveugle).

  2. apply_literature_anchor() -- OPTIONNEL, DESACTIVE PAR DEFAUT.
     Nudge (pas un clamp brutal) de l'albedo vers la plage bibliographique
     UNIQUEMENT sur le corps "normal" (jamais sur les cas deja flagges
     suspects par la fonction 1 -- les corriger masquerait l'erreur de
     classification au lieu de la reveler). Force reglable (alpha), tracee
     dans une colonne separee `albedo_ajuste_biblio` qui NE REMPLACE JAMAIS
     `albedo` (colonne originale du modele) -- les deux doivent rester
     visibles dans tout export/rapport.

     ATTENTION -- CE N'EST PAS UNE CALIBRATION VALIDEE. C'est un ajustement
     base sur l'hypothese que la litterature (sources heterogenes,
     internationales, pas specifiques aux toits francais photographies en
     IGN 20cm) est plus fiable que le modele -- hypothese non verifiee.
     A n'utiliser QUE si explicitement mentionne comme tel dans tout
     livrable (cf. METHODOLOGIE.md, a completer). Ne remplace en aucun cas
     une vraie campagne de validation terrain ou OSM.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

# Plages bibliographiques "vieilli/expose" issues de albedo_reference_table.md
# -- NE PAS elargir/retrecir sans mettre a jour ce fichier .md en parallele
# (source de verite documentaire).
# Alpha par defaut utilise par zone_scan.py/pipeline.py/app.py -- 0.0 =
# fonctionnalite INERTE (albedo_ajuste_biblio == albedo). A ne monter que si
# une vraie validation (OSM ou annotation manuelle) confirme un biais reel
# et pas seulement une plage bibliographique trop etroite (cf. discussion
# du 10/07/2026 -- pas de correction sans preuve). Modifier ICI uniquement
# (source unique) plutot que dans chaque module appelant.
# GARDE-FOU (point 4 du plan de dette technique, cf. discussion) : ne PAS
# relever cette valeur au-dessus de 0.0 sans un rapport de
# validate_thresholds.py (points 2/3) documentant un biais reel mesure sur
# un jeu stratifie (12-15 villes / 150-200 echantillons, cf. brief Sprint 4)
# -- pas sur un seul scan comme les 4 du 10/07/2026 qui ont motive ce module
# a l'origine. Si tu relis ce commentaire en train de bumper cette valeur :
# as-tu bien le rapport sous la main, ou est-ce une correction sans preuve
# (cf. discussion du 10/07/2026 explicitement citee dans ce module) ?
DEFAULT_ANCHOR_ALPHA: float = 0.0

LITERATURE_RANGES: dict[str, tuple[float, float]] = {
    "zinc":              (0.15, 0.30),
    "ardoise":           (0.08, 0.20),
    "tuile_terre_cuite": (0.25, 0.45),
    "beton":             (0.20, 0.35),
    # "indetermine" volontairement absent : categorie fourre-tout, aucune
    # plage physique n'a de sens a lui appliquer.
}

# Multiplicateur de tolerance avant de considerer un ecart "suspect" plutot
# que "un peu au-dessus/en-dessous, normal vu l'incertitude des plages".
# Ex. ardoise (0.08-0.20, largeur 0.12) : marge = 0.12*0.5 = 0.06 de chaque
# cote -> suspect si albedo > 0.26 ou < 0.02. Calibre a l'oeil sur la
# distribution observee (queue haute ardoise commence vers 0.30, bien
# au-dela de cette marge) -- PAS statistiquement valide, meme logique de
# prudence que les seuils de materials.py.
SUSPECT_MARGIN_RATIO: float = 0.5


@dataclass
class PlausibilityFlag:
    n_total: int
    n_suspect: int
    n_no_range: int  # materiau sans plage definie (ex. indetermine)


def classify_plausibility_single(material: Optional[str], albedo: Optional[float]) -> str:
    """Version scalaire (1 batiment), SANS dependance pandas -- utilisee par
    zone_scan.py et pipeline.py (modes adresse unique/lot/zone) pour ne pas
    dupliquer les seuils LITERATURE_RANGES/SUSPECT_MARGIN_RATIO a 3 endroits.
    flag_material_plausibility() (version DataFrame, pour analyse offline
    des CSV exportes) appelle desormais celle-ci ligne par ligne.

    Retourne : "sans_reference" / "normale" / "suspecte_reclassification".
    Ne modifie jamais albedo ni material -- detection uniquement (cf.
    docstring module).
    """
    if material is None or albedo is None:
        return "sans_reference"
    bounds = LITERATURE_RANGES.get(material)
    if bounds is None:
        return "sans_reference"
    lo, hi = bounds
    margin = (hi - lo) * SUSPECT_MARGIN_RATIO
    if albedo < lo - margin or albedo > hi + margin:
        return "suspecte_reclassification"
    return "normale"


def adjust_albedo_single(
    material: Optional[str],
    albedo: Optional[float],
    plausibilite: str,
    alpha: float = 0.0,
    only_normal: bool = True,
) -> Optional[float]:
    """Equivalent scalaire de apply_literature_anchor(), pour usage direct
    dans zone_scan.py/pipeline.py sans construire de DataFrame.

    alpha=0.0 (defaut) -> retourne toujours albedo inchange (fonctionnalite
    inerte tant qu'elle n'est pas explicitement activee, cf. discussion
    "pas de correction sans preuve" du 10/07/2026).
    """
    if albedo is None or material is None:
        return albedo
    if alpha <= 0.0:
        return albedo
    if only_normal and plausibilite != "normale":
        return albedo
    bounds = LITERATURE_RANGES.get(material)
    if bounds is None:
        return albedo
    lo, hi = bounds
    above = max(0.0, albedo - hi)
    below = max(0.0, lo - albedo)
    return albedo - alpha * above + alpha * below


def flag_material_plausibility(df: pd.DataFrame) -> tuple[pd.DataFrame, PlausibilityFlag]:
    """Ajoute `plausibilite_materiau` sans jamais modifier `albedo` ni
    `materiau_estime`. Colonnes requises : `albedo`, `materiau_estime`.

    Valeurs de sortie :
        "normale"                : albedo dans la plage biblio (+/- marge)
        "suspecte_reclassification" : albedo largement hors plage -- le
            materiau estime est probablement faux, PAS l'albedo. A prioriser
            pour une revue manuelle ciblee.
        "sans_reference"         : materiau sans plage biblio (indetermine)
    """
    out = df.copy()
    out["plausibilite_materiau"] = [
        classify_plausibility_single(m, a)
        for m, a in zip(out["materiau_estime"], out["albedo"])
    ]

    n_suspect = int((out["plausibilite_materiau"] == "suspecte_reclassification").sum())
    n_no_range = int((out["plausibilite_materiau"] == "sans_reference").sum())

    flag_summary = PlausibilityFlag(
        n_total=len(out), n_suspect=n_suspect, n_no_range=n_no_range,
    )
    return out, flag_summary


def apply_literature_anchor(
    df: pd.DataFrame,
    alpha: float = DEFAULT_ANCHOR_ALPHA,
    only_normal: bool = True,
) -> pd.DataFrame:
    """AJOUTE `albedo_ajuste_biblio` (ne remplace jamais `albedo`).

    Nudge lineaire partiel vers le bord le plus proche de la plage
    bibliographique, PAS vers le centre (plus conservateur : on ramene juste
    a la frontiere de plausibilite, pas a une valeur "ideale" arbitraire) :

        albedo_ajuste = albedo - alpha * max(0, albedo - hi)   si au-dessus
        albedo_ajuste = albedo + alpha * max(0, lo - albedo)   si en-dessous

    alpha=0   -> aucun ajustement (albedo_ajuste_biblio == albedo)
    alpha=1   -> ramene exactement a la frontiere de la plage
    alpha=0.3 (defaut) -> ajustement partiel, deliberement prudent

    IMPLEMENTATION (10/07/2026) : delegue desormais a adjust_albedo_single()
    ligne par ligne, plutot que de reimplementer la meme formule en pandas
    .clip() vectorise. Avant ce changement, les deux fonctions codaient
    independamment la meme formule mathematique -- risque de divergence si
    l'une etait modifiee sans l'autre (cf. revue du 10/07/2026). Le cout de
    perf vs numpy vectorise est negligeable ici (quelques milliers de lignes
    max par scan, pas un hot-path).

    Args:
        only_normal: si True (defaut), n'ajuste QUE les lignes deja
            flagguees "normale" par flag_material_plausibility() -- les cas
            "suspecte_reclassification" ne sont PAS touches ici (leur
            probleme est le materiau, pas l'albedo ; les "corriger"
            masquerait l'erreur de classification, cf. docstring module).
            Necessite que flag_material_plausibility() ait deja ete appele
            sur ce DataFrame (colonne `plausibilite_materiau` presente).

    Leve ValueError si only_normal=True mais que la colonne
    `plausibilite_materiau` est absente (appel dans le mauvais ordre).
    """
    if only_normal and "plausibilite_materiau" not in df.columns:
        raise ValueError(
            "apply_literature_anchor(only_normal=True) requiert d'avoir "
            "appele flag_material_plausibility() d'abord (colonne "
            "'plausibilite_materiau' absente)."
        )

    out = df.copy()

    if "plausibilite_materiau" in out.columns:
        plausibilites = out["plausibilite_materiau"]
    else:
        # only_normal=False et colonne absente : adjust_albedo_single ne
        # regarde plausibilite que si only_normal=True, donc la valeur
        # exacte ici n'a pas d'importance -- "normale" par convention.
        plausibilites = pd.Series(["normale"] * len(out), index=out.index)

    out["albedo_ajuste_biblio"] = [
        adjust_albedo_single(material, albedo, plaus, alpha=alpha, only_normal=only_normal)
        for material, albedo, plaus in zip(out["materiau_estime"], out["albedo"], plausibilites)
    ]

    return out


if __name__ == "__main__":
    # Petit test manuel sur des cas synthetiques -- pas un test unitaire
    # complet, juste une verification rapide du comportement attendu.
    df = pd.DataFrame({
        "materiau_estime": ["ardoise", "ardoise", "ardoise", "beton", "indetermine"],
        "albedo":          [0.224,     0.35,      0.10,      0.33,    0.27],
    })
    flagged, summary = flag_material_plausibility(df)
    print(flagged)
    print(summary)

    adjusted = apply_literature_anchor(flagged, alpha=0.3, only_normal=True)
    print(adjusted[["materiau_estime", "albedo", "plausibilite_materiau", "albedo_ajuste_biblio"]])
