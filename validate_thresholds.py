"""
validate_thresholds.py — AlbedoNet App (outillage Sprint 4)
==============================================================
Script offline pour DEUX usages distincts, tous les deux bloques faute de
donnees reelles au moment ou ce fichier a ete ecrit (aucun export CSV/GeoJSON
de scan disponible) :

  1. Calibration empirique de SHADOW_CORRECTION_FACTOR (shadow_calibration.py)
     -- comparer, par ville/materiau, l'albedo moyen predit sur pixels
     ombrages vs non-ombrages (methode decrite dans shadow_calibration.py,
     jamais executee faute de donnees).

  2. Validation/recalibrage de LITERATURE_RANGES et SUSPECT_MARGIN_RATIO
     (postprocess_plausibility.py) -- actuellement calibres sur 4 scans
     (n=65904) un seul jour (10/07/2026). Le brief Sprint 4 demande un
     jeu stratifie de 150-200 adresses / 12-15 villes, avec pools
     calibration/validation strictement separes -- ce script applique
     cette separation.

NE PAS lancer ce script sur un seul scan/une seule ville : ca reproduirait
exactement le biais deja identifie (seuils calibres sur un echantillon non
representatif). Le garde-fou _check_stratification() ci-dessous refuse de
continuer si la couverture est insuffisante.

Entree attendue : un ou plusieurs CSV produits par /api/zone.csv ou
/api/batch.csv (colonnes : lat, lon, modele_utilise / ou une colonne city
explicite, albedo, materiau_estime -- shadow_fraction et albedo_brut
necessaires en plus pour la partie 1, absents des CSV actuels de main.py,
cf. TODO plus bas).

Usage :
    python validate_thresholds.py --input scan1.csv scan2.csv ... \\
        --min-cities 12 --min-samples 150 \\
        --calib-fraction 0.5 --output-dir validation_out/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit  # meme outil que train_material_classifier.py, coherence du projet

from postprocess_plausibility import LITERATURE_RANGES, SUSPECT_MARGIN_RATIO, classify_plausibility_single

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RANDOM_STATE = 42


class InsufficientStratificationError(Exception):
    """Leve si le jeu de donnees ne respecte pas les exigences minimales du
    brief Sprint 4 -- empeche de recalibrer sur un echantillon non
    representatif, meme par accident (ex. un seul CSV oublie dans --input)."""


def _load_and_concat(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable : {p}")
        df = pd.read_csv(p)
        df["source_file"] = p.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _check_stratification(df: pd.DataFrame, city_col: str, min_cities: int, min_samples: int) -> None:
    n_cities = df[city_col].nunique()
    n_samples = len(df)
    if n_cities < min_cities:
        raise InsufficientStratificationError(
            f"{n_cities} ville(s) distincte(s) dans les donnees fournies, "
            f"{min_cities} requises par le brief Sprint 4. Ajouter des scans "
            f"d'autres villes avant de recalibrer -- sinon on reproduit le "
            f"biais deja identifie (seuils calibres sur un echantillon non "
            f"representatif, cf. postprocess_plausibility.py docstring)."
        )
    if n_samples < min_samples:
        raise InsufficientStratificationError(
            f"{n_samples} echantillon(s) fourni(s), {min_samples} requis par "
            f"le brief Sprint 4."
        )
    per_city = df[city_col].value_counts()
    logger.info("Repartition par ville :\n%s", per_city.to_string())
    if (per_city < 5).any():
        logger.warning(
            "Ville(s) avec moins de 5 echantillons : %s -- risque de bruit "
            "eleve sur ces sous-groupes, interpreter les resultats par-ville "
            "avec prudence.",
            per_city[per_city < 5].index.tolist(),
        )


def _split_calib_validation(
    df: pd.DataFrame, city_col: str, calib_fraction: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split calibration/validation GROUPE PAR VILLE (pas par ligne) : une
    ville entiere va soit en calibration, soit en validation, jamais les
    deux -- sinon la validation ne teste rien de nouveau (meme logique de
    fuite que train_material_classifier.py evite deja via group_id, ici le
    "groupe" naturel est la ville plutot qu'un cluster spatial ~500m)."""
    splitter = GroupShuffleSplit(n_splits=1, train_size=calib_fraction, random_state=RANDOM_STATE)
    idx_calib, idx_val = next(splitter.split(df, groups=df[city_col]))
    calib_df, val_df = df.iloc[idx_calib], df.iloc[idx_val]
    logger.info(
        "Split calibration/validation : %d villes / %d echantillons en calibration, "
        "%d villes / %d echantillons en validation.",
        calib_df[city_col].nunique(), len(calib_df),
        val_df[city_col].nunique(), len(val_df),
    )
    return calib_df, val_df


def _suspect_rate_report(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Applique LITERATURE_RANGES/SUSPECT_MARGIN_RATIO ACTUELS (sans les
    modifier) et rapporte le taux de 'suspecte_reclassification' par
    materiau -- point de depart pour juger si les seuils actuels tiennent
    sur un echantillon plus large que les 4 scans du 10/07/2026."""
    plaus = [
        classify_plausibility_single(m, a)
        for m, a in zip(df["materiau_estime"], df["albedo"])
    ]
    out = df.assign(plausibilite_materiau=plaus)
    report = (
        out.groupby("materiau_estime")["plausibilite_materiau"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
    )
    logger.info("[%s] Taux de plausibilite par materiau (seuils actuels) :\n%s", label, report)
    return report


def _propose_ranges_from_normal_body(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Propose de NOUVELLES plages (5e/95e percentile) par materiau, calculees
    UNIQUEMENT sur le pool de calibration -- a comparer manuellement a
    LITERATURE_RANGES avant toute mise a jour de postprocess_plausibility.py.
    Ne remplace RIEN automatiquement : affichage seulement, la decision de
    changer les seuils reste humaine (meme principe que threshold_guard.py --
    aucun changement de seuil sans revue deliberee)."""
    proposals: dict[str, tuple[float, float]] = {}
    for material in sorted(df["materiau_estime"].dropna().unique()):
        sub = df.loc[df["materiau_estime"] == material, "albedo"].dropna()
        if len(sub) < 10:
            logger.warning(
                "Materiau '%s' : seulement %d echantillons, percentiles peu fiables -- ignore.",
                material, len(sub),
            )
            continue
        lo, hi = float(np.percentile(sub, 5)), float(np.percentile(sub, 95))
        proposals[material] = (lo, hi)
        current = LITERATURE_RANGES.get(material)
        logger.info(
            "Materiau '%s' : plage proposee (P5-P95, n=%d) = (%.3f, %.3f) -- "
            "plage actuelle LITERATURE_RANGES = %s",
            material, len(sub), lo, hi, current,
        )
    return proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True,
                         help="CSV exportes via /api/zone.csv ou /api/batch.csv")
    parser.add_argument("--city-col", default="modele_utilise",
                         help="Colonne identifiant la ville/le modele utilise (defaut: modele_utilise)")
    parser.add_argument("--min-cities", type=int, default=12,
                         help="Minimum de villes distinctes requis (brief Sprint 4: 12-15)")
    parser.add_argument("--min-samples", type=int, default=150,
                         help="Minimum d'echantillons requis (brief Sprint 4: 150-200)")
    parser.add_argument("--calib-fraction", type=float, default=0.5,
                         help="Fraction des villes (pas des lignes) allouee au pool de calibration")
    parser.add_argument("--output-dir", type=Path, default=Path("validation_out"))
    args = parser.parse_args()

    df = _load_and_concat(args.input)

    if args.city_col not in df.columns:
        logger.error(
            "Colonne '%s' absente des CSV fournis (colonnes disponibles : %s). "
            "Utiliser --city-col pour pointer vers la bonne colonne.",
            args.city_col, list(df.columns),
        )
        sys.exit(1)

    required_cols = {"albedo", "materiau_estime", args.city_col}
    missing = required_cols - set(df.columns)
    if missing:
        logger.error("Colonnes requises manquantes : %s", missing)
        sys.exit(1)

    df = df.dropna(subset=["albedo", "materiau_estime"])

    try:
        _check_stratification(df, args.city_col, args.min_cities, args.min_samples)
    except InsufficientStratificationError as exc:
        logger.error(str(exc))
        sys.exit(1)

    calib_df, val_df = _split_calib_validation(df, args.city_col, args.calib_fraction)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Rapport sur pool de CALIBRATION (seuils actuels) ===")
    calib_report = _suspect_rate_report(calib_df, "calibration")
    calib_report.to_csv(args.output_dir / "suspect_rate_calibration.csv")

    logger.info("=== Rapport sur pool de VALIDATION (seuils actuels, jamais vus en calibration) ===")
    val_report = _suspect_rate_report(val_df, "validation")
    val_report.to_csv(args.output_dir / "suspect_rate_validation.csv")

    logger.info("=== Propositions de nouvelles plages (calculees sur calibration uniquement) ===")
    proposals = _propose_ranges_from_normal_body(calib_df)
    pd.DataFrame(
        [{"materiau": m, "lo_propose": lo, "hi_propose": hi, "lo_actuel": LITERATURE_RANGES.get(m, (None, None))[0],
          "hi_actuel": LITERATURE_RANGES.get(m, (None, None))[1]} for m, (lo, hi) in proposals.items()]
    ).to_csv(args.output_dir / "ranges_proposees.csv", index=False)

    logger.info(
        "Termine. AUCUN fichier de seuils n'a ete modifie -- rapports ecrits "
        "dans %s pour revue humaine. Si une mise a jour de LITERATURE_RANGES "
        "est decidee sur cette base, l'appliquer manuellement dans "
        "postprocess_plausibility.py ET regenerer le hash via "
        "threshold_guard.py / regenerate_hash.py, comme documente.",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
