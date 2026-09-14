"""Interface d'annotation manuelle de matériaux de toiture, basée sur Streamlit."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from geocoding import geocode_address
from ign_fetch import fetch_building_and_ortho
from patch_extraction import extract_raw_and_normalized_patch

ANNOTATION_FILE = Path("datasets/annotations_manuelles.csv")
PATCH_DIR = Path("datasets/patches")
MATERIALS = ["zinc", "ardoise", "tuile_terre_cuite", "beton", "indetermine"]

st.title("Annotateur de matériaux de toiture")


def load_annotations() -> pd.DataFrame:
    if ANNOTATION_FILE.exists():
        return pd.read_csv(ANNOTATION_FILE)
    return pd.DataFrame(columns=["address", "lat", "lon", "material", "patch_path"])


def already_annotated(df: pd.DataFrame, lat: float, lon: float, tol: float = 1e-6) -> bool:
    if df.empty:
        return False
    return (
        ((df["lat"] - lat).abs() < tol) & ((df["lon"] - lon).abs() < tol)
    ).any()


df = load_annotations()
address = st.text_input("Adresse à annoter")

if address:
    try:
        geocode = geocode_address(address)
    except Exception as exc:
        st.error(f"Erreur de géocodage : {exc}")
        st.stop()

    if not geocode.found:
        st.error("Adresse introuvable")
        st.stop()

    if already_annotated(df, geocode.lat, geocode.lon):
        st.warning("Ce bâtiment est déjà annoté. Continuer créera un doublon.")

    with st.spinner("Récupération orthophoto..."):
        try:
            fetch_result = fetch_building_and_ortho(geocode.lat, geocode.lon, Path("."), with_candidates=False)
        except Exception as exc:
            st.error(f"Erreur lors de la récupération IGN : {exc}")
            st.stop()

    if fetch_result.match.confidence == "none":
        st.error("Aucun bâtiment trouvé à cette adresse")
        st.stop()

    eff_x = fetch_result.match.centroid_x or fetch_result.point_l93.x
    eff_y = fetch_result.match.centroid_y or fetch_result.point_l93.y

    try:
        _, patch, _ = extract_raw_and_normalized_patch(fetch_result.ortho_path, eff_x, eff_y)
    except Exception as exc:
        st.error(f"Erreur d'extraction du patch : {exc}")
        st.stop()

    if patch is None:
        st.error("Patch invalide (hors orthophoto ou taille insuffisante)")
        st.stop()

    img = (patch.transpose(1, 2, 0) * 255).astype("uint8")
    pil_img = Image.fromarray(img)
    st.image(pil_img, caption="Patch extrait", width=256)
    st.caption(f"Confiance du matching bâtiment : {fetch_result.match.confidence}")

    material = st.selectbox("Matériau", MATERIALS)
    if st.button("Enregistrer"):
        PATCH_DIR.mkdir(parents=True, exist_ok=True)
        patch_filename = f"{geocode.lat:.6f}_{geocode.lon:.6f}.png"
        patch_path = PATCH_DIR / patch_filename
        pil_img.save(patch_path)

        new_entry = pd.DataFrame([{
            "address": address,
            "lat": geocode.lat,
            "lon": geocode.lon,
            "material": material,
            "patch_path": str(patch_path),
        }])
        df = pd.concat([df, new_entry], ignore_index=True)
        ANNOTATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(ANNOTATION_FILE, index=False)
        st.success("Annotation enregistrée")
