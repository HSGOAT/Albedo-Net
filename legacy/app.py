"""
app.py — AlbedoNet App
========================
Interface Streamlit : adresse -> albedo estime du toit.

Assemble le pipeline complet :
  1. geocoding.geocode_address()          adresse -> lat/lon + commune/citycode
  2. geocoding.resolve_city_key()          -> cle ville, ou besoin de fallback region
     (selectbox region + ville si necessaire, via config.REGIONS_FR /
     geocoding.resolve_city_key_from_region())
  3. ign_fetch.fetch_building_and_ortho()  -> batiment matche (cascade) + orthophoto
  4. patch_extraction.extract_and_normalize_patch()  -> patch 64x64 min-max [0,1]
  5. config.get_checkpoint_path()          -> chemin du .pt pour la cle ville
  6. inference.predict_albedo()            -> albedo (z-score + MAEEncoder + AlbedoHead)
  7. Affichage : albedo, carte (point + batiment matche), image du toit decoupe

Filet de securite (cf. suivi projet, §3) : si le matching batiment renvoie
confidence == "none", l'app avertit explicitement l'utilisateur qu'aucune
correspondance fiable n'a ete trouvee et n'affiche PAS de prediction
automatique -- seulement la carte + l'image brute pour verification visuelle.
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

import streamlit as st
import pydeck as pdk

from config import GENERIC_FALLBACK_MODEL, REGIONS_FR, get_checkpoint_path
from geocoding import geocode_address, resolve_city_key, resolve_city_key_from_region, reverse_geocode
from ign_fetch import crop_thumbnail, fetch_building_and_ortho
from inference import predict_albedo
from materials import classify_material
from patch_extraction import extract_raw_and_normalized_patch, patch_to_png_bytes
from pipeline import CityPreResolution, pre_resolve_addresses, process_batch
from zone_scan import DARK_ROOF_ALBEDO_THRESHOLD, MAX_ZONE_RADIUS_M, MIN_DARK_NEIGHBORS, MIN_BRIGHT_NEIGHBORS, scan_zone

st.set_page_config(page_title="AlbedoNet — Albedo par adresse", page_icon="🏠", layout="centered")

# ──────────────────────────────────────────────────────────────────────────
# Style "pro" leger -- cards uniformes pour les vignettes de batiments/
# resultats (candidats, galerie lot/zone), sans toucher au theme Streamlit
# global (reste compatible clair/sombre).
# ──────────────────────────────────────────────────────────────────────────
# ──────────────────────────────────────────────────────────────────────────
# Direction visuelle AlbedoNet -- reprend le design system du mockup
# (typo Space Grotesk / Public Sans / JetBrains Mono, palette "toitures"
# copper/sage/amber/red/blue sur fond sombre instrument). Habillage CSS
# uniquement via st.markdown : aucune logique applicative n'est modifiee.
# ──────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Public+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
    :root{
        --ink:#0D1420; --surface:#141C2B; --surface-raised:#1B2436; --surface-hover:#212B40;
        --line:rgba(255,255,255,0.08); --line-strong:rgba(255,255,255,0.16);
        --text-primary:#EDEFF3; --text-secondary:#8B93A3; --text-muted:#828BA0;
        --copper:#E8874A; --copper-dim:#B96A34; --copper-soft:rgba(232,135,74,0.14);
        --sage:#8CB79A; --sage-soft:rgba(140,183,154,0.14);
        --amber:#D9A441; --amber-soft:rgba(217,164,65,0.14);
        --red:#E0645F; --red-soft:rgba(224,100,95,0.14);
        --blue:#5B8FD4; --blue-soft:rgba(91,143,212,0.14);
        --font-display:'Space Grotesk',sans-serif; --font-body:'Public Sans',sans-serif;
        --font-mono:'JetBrains Mono',ui-monospace,monospace;
        --r-sm:8px; --r-md:14px; --r-lg:20px;
        --charbon:#16171A; --craie:#F4F1EA;
    }

    /* ---- fond + typo globale ---- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"]{
        background:
            radial-gradient(ellipse 1000px 620px at 18% -6%, rgba(232,135,74,0.05), transparent 60%),
            linear-gradient(180deg, rgba(20,28,43,0.9) 0%, var(--ink) 45%), var(--ink) !important;
    }
    [data-testid="stHeader"]{background:transparent !important;}
    html, body, [class*="css"], .stMarkdown, .stMarkdown p, label, .stCaptionContainer{
        font-family:var(--font-body) !important; color:var(--text-primary);
    }
    [data-testid="stMainBlockContainer"]{padding-top:2.5rem;}
    h1,h2,h3,h4, [data-testid="stHeading"] h1{
        font-family:var(--font-display) !important; font-weight:600 !important;
        letter-spacing:0.01em; color:var(--text-primary) !important;
    }
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p{
        color:var(--text-muted) !important; font-size:12.5px;
    }
    code, .stMarkdown code{
        font-family:var(--font-mono) !important; background:var(--surface-raised) !important;
        color:var(--copper) !important; border-radius:4px;
    }
    hr, [data-testid="stDivider"]{border-color:var(--line) !important;}
    a{color:var(--copper) !important;}
    :focus-visible{outline:2px solid var(--copper) !important; outline-offset:2px;}

    /* ---- corner HUD (signature du mockup) ---- */
    .anet-hud{position:fixed;inset:0;pointer-events:none;z-index:0;}
    .anet-hud span{position:absolute;width:22px;height:22px;border-color:rgba(232,135,74,0.32);}
    .anet-hud .tl{top:14px;left:14px;border-top:1px solid;border-left:1px solid;}
    .anet-hud .tr{top:14px;right:14px;border-top:1px solid;border-right:1px solid;}
    .anet-hud .bl{bottom:14px;left:14px;border-bottom:1px solid;border-left:1px solid;}
    .anet-hud .br{bottom:14px;right:14px;border-bottom:1px solid;border-right:1px solid;}

    /* ---- onglets (pilule, comme le mockup) ---- */
    [data-testid="stTabs"]{font-family:var(--font-body);}
    [data-baseweb="tab-list"]{
        gap:4px; border:1px solid var(--line); border-radius:999px; padding:4px;
        background:var(--surface); width:fit-content;
    }
    [data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{display:none !important;}
    [data-testid="stTab"], button[data-baseweb="tab"]{
        border-radius:999px !important; color:var(--text-secondary) !important;
        font-weight:500; font-size:13.5px; padding:8px 18px !important;
        background:transparent !important; transition:background .15s ease,color .15s ease;
    }
    [data-testid="stTab"][aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"]{
        background:var(--surface-hover) !important; color:var(--text-primary) !important;
    }
    [data-testid="stTabPanel"]{padding-top:20px;}

    /* ---- boutons ---- */
    [data-testid="stBaseButton-primary"]{
        background:var(--copper) !important; color:#241206 !important; border:none !important;
        font-family:var(--font-body); font-weight:600; border-radius:var(--r-sm) !important;
    }
    [data-testid="stBaseButton-primary"]:hover{background:#f0985e !important;}
    [data-testid="stBaseButton-secondary"]{
        background:transparent !important; color:var(--text-secondary) !important;
        border:1px solid var(--line-strong) !important; border-radius:var(--r-sm) !important;
        font-family:var(--font-body); font-weight:500;
    }
    [data-testid="stBaseButton-secondary"]:hover{border-color:var(--copper) !important; color:var(--copper) !important;}

    /* ---- champs texte / zone de texte / selectbox ---- */
    [data-testid="stTextInputRootElement"], [data-testid="stTextAreaRootElement"],
    div[data-baseweb="select"] > div, div[data-baseweb="base-input"]{
        background:var(--surface) !important; border:1px solid var(--line-strong) !important;
        border-radius:var(--r-sm) !important; color:var(--text-primary) !important;
    }
    input, textarea{color:var(--text-primary) !important; font-family:var(--font-body) !important;}
    input::placeholder, textarea::placeholder{color:var(--text-muted) !important;}
    [data-baseweb="select"] *{color:var(--text-primary) !important;}
    ul[data-testid="stSelectboxVirtualDropdown"], div[role="listbox"]{
        background:var(--surface-raised) !important; border:1px solid var(--line) !important;
    }

    /* ---- radio ---- */
    [data-testid="stRadioGroup"] label{
        background:var(--surface); border:1px solid var(--line); border-radius:999px;
        padding:6px 14px; margin-right:6px; color:var(--text-secondary) !important;
    }
    [data-testid="stRadioGroup"] label:has(input:checked){
        border-color:var(--copper); color:var(--text-primary) !important;
    }

    /* ---- slider ---- */
    [data-testid="stSlider"] [role="slider"]{background:var(--copper) !important; border-color:var(--copper) !important;}
    [data-testid="stSliderTickBar"]{color:var(--text-muted) !important;}
    div[data-baseweb="slider"] > div > div{background:var(--copper) !important;}

    /* ---- file uploader ---- */
    [data-testid="stFileUploaderDropzone"]{
        background:var(--surface) !important; border:1px dashed var(--line-strong) !important;
        border-radius:var(--r-md) !important;
    }

    /* ---- alertes (success/info/warning/error) recolorees palette mockup ---- */
    [data-testid="stAlertContainer"]{border-radius:var(--r-md) !important; border:1px solid var(--line) !important;}
    [data-testid="stAlertContentSuccess"]{background:var(--sage-soft) !important; color:var(--sage) !important;}
    [data-testid="stAlertContentInfo"]{background:var(--blue-soft) !important; color:var(--blue) !important;}
    [data-testid="stAlertContentWarning"]{background:var(--amber-soft) !important; color:var(--amber) !important;}
    [data-testid="stAlertContentError"]{background:var(--red-soft) !important; color:var(--red) !important;}
    [data-testid="stAlertContainer"] p{color:inherit !important;}

    /* ---- spinner / progress ---- */
    [data-testid="stSpinner"]{color:var(--text-secondary) !important;}
    [data-testid="stSpinner"] svg{color:var(--copper) !important;}
    [data-testid="stProgress"] div[role="progressbar"] > div{background:var(--copper) !important;}
    [data-testid="stProgress"]{background:var(--surface-raised) !important; border-radius:999px;}

    /* ---- metric ---- */
    [data-testid="stMetric"]{
        background:var(--surface); border:1px solid var(--line); border-radius:var(--r-md);
        padding:14px 16px;
    }
    [data-testid="stMetricValue"]{font-family:var(--font-mono) !important; color:var(--text-primary) !important;}
    [data-testid="stMetricLabel"]{color:var(--text-muted) !important; font-size:11.5px !important; text-transform:uppercase; letter-spacing:.03em;}

    /* ---- expander (= carte du mockup) ---- */
    [data-testid="stExpander"]{
        background:var(--surface) !important; border:1px solid var(--line) !important;
        border-radius:var(--r-lg) !important; overflow:hidden;
    }
    [data-testid="stExpander"] summary{font-family:var(--font-display); color:var(--text-primary) !important;}

    /* ---- dataframe ---- */
    [data-testid="stDataFrame"]{border:1px solid var(--line) !important; border-radius:var(--r-md) !important; overflow:hidden;}

    /* ---- images / carte ---- */
    [data-testid="stImageContainer"] img{border-radius:var(--r-sm);}
    [data-testid="stDeckGlJsonChart"], [data-testid="stFullScreenFrame"]{border-radius:var(--r-md); overflow:hidden; border:1px solid var(--line);}

    /* ---- cartes vignettes (candidats / galerie lot / galerie zone) ---- */
    .albedo-card{
        border:1px solid var(--line); border-radius:var(--r-md);
        padding:10px 10px 8px; margin-bottom:10px; background:var(--surface);
        transition:border-color .15s ease;
    }
    .albedo-card:hover{border-color:var(--copper);}
    .albedo-card img{border-radius:var(--r-sm); width:100%;}
    .albedo-card-title{font-family:var(--font-body); font-weight:600; font-size:12.5px; margin-top:8px; color:var(--text-primary);}
    .albedo-card-meta{font-family:var(--font-mono); font-size:11px; color:var(--text-muted); margin-top:2px;}
    .albedo-badge{
        display:inline-block; padding:3px 10px; border-radius:999px;
        font-family:var(--font-body); font-size:11px; font-weight:600; margin-right:6px; margin-top:6px;
    }
    .badge-high{background:var(--sage-soft); color:var(--sage);}
    .badge-medium{background:var(--amber-soft); color:var(--amber);}
    .badge-none{background:var(--red-soft); color:var(--red);}

    /* ---- jauge d'albedo (composant signature du mockup) ---- */
    .reflectance{margin-top:10px; margin-bottom:6px;}
    .reflectance-value{display:flex; align-items:baseline; gap:10px; margin-bottom:14px;}
    .reflectance-value .num{font-family:var(--font-mono); font-size:40px; font-weight:500; letter-spacing:-0.01em; color:var(--text-primary);}
    .reflectance-value .unit{font-size:13px; color:var(--text-muted);}
    .reflectance-bar-wrap{position:relative; padding-top:22px; margin-bottom:8px;}
    .reflectance-bar{
        height:10px; border-radius:999px;
        background:linear-gradient(90deg,var(--charbon) 0%,#4b4b4d 26%,#8c8c8a 52%,#c7c5bd 78%,var(--craie) 100%);
        border:1px solid rgba(255,255,255,0.1);
    }
    .reflectance-marker{position:absolute; top:0; transform:translateX(-50%); display:flex; flex-direction:column; align-items:center;}
    .reflectance-marker .pin{width:2px; height:20px; background:var(--copper);}
    .reflectance-marker .tri{width:0; height:0; border-left:5px solid transparent; border-right:5px solid transparent; border-top:6px solid var(--copper); margin-top:-1px;}
    .reflectance-ticks{display:flex; justify-content:space-between; font-family:var(--font-mono); font-size:10.5px; color:var(--text-muted); margin-top:6px;}
    .reflectance-labels{display:flex; justify-content:space-between; font-size:10.5px; color:var(--text-muted); margin-top:2px;}

    @media (prefers-reduced-motion:reduce){*{transition:none !important;}}
    </style>
    <div class="anet-hud" aria-hidden="true"><span class="tl"></span><span class="tr"></span><span class="bl"></span><span class="br"></span></div>
    """,
    unsafe_allow_html=True,
)

CONFIDENCE_BADGE_CLASS = {"high": "badge-high", "medium": "badge-medium", "none": "badge-none"}
CONFIDENCE_BADGE_LABEL = {
    "high": "✅ Correspondance fiable",
    "medium": "🟡 Correspondance approximative",
    "none": "🔴 Aucune correspondance fiable",
}


def _reflectance_bar_html(albedo: float) -> str:
    """Jauge d'albedo (0=noir/asphalte -> 1=blanc reflechissant), reprise du
    mockup ("reflectance scale"). Le marqueur est positionne dynamiquement
    selon la vraie valeur predite -- ce n'est PAS la donnee fictive du mockup."""
    pct = max(0.0, min(1.0, albedo)) * 100
    return (
        '<div class="reflectance">'
        f'<div class="reflectance-value"><span class="num">{albedo:.3f}</span>'
        '<span class="unit">fraction reflechie [0-1]</span></div>'
        '<div class="reflectance-bar-wrap">'
        f'<div class="reflectance-marker" style="left:{pct:.1f}%;">'
        '<div class="tri"></div><div class="pin"></div></div>'
        '<div class="reflectance-bar"></div>'
        '</div>'
        '<div class="reflectance-ticks"><span>0.0</span><span>0.25</span>'
        '<span>0.5</span><span>0.75</span><span>1.0</span></div>'
        '<div class="reflectance-labels"><span>asphalte neuf</span><span></span>'
        '<span>gris moyen</span><span></span><span>blanc reflechissant</span></div>'
        '</div>'
    )


def _confidence_badge_html(confidence: str) -> str:
    cls = CONFIDENCE_BADGE_CLASS.get(confidence, "badge-medium")
    label = CONFIDENCE_BADGE_LABEL.get(confidence, confidence)
    return f'<span class="albedo-badge {cls}">{label}</span>'


# ──────────────────────────────────────────────────────────────────────────
# Etat de session (pour gerer le flux en plusieurs etapes : geocodage ->
# eventuel choix region/ville -> calcul)
# ──────────────────────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults = {
        "geocode_result": None,
        "city_resolution": None,
        "chosen_city_key": None,
        "chosen_region": None,
        "batch_signature": None,
        "batch_preresolved": None,
        "batch_overrides": {},
        "batch_results": None,
        # Selection manuelle de batiment (points 1/2/5, cf. suivi projet) :
        # None = pas encore choisi -> affiche le selecteur si site dense ;
        # "auto" = accepte le pick automatique (score distance+surface) ;
        # (x, y) = centroide L93 choisi manuellement par l'utilisateur.
        "chosen_building_xy": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _reset_state() -> None:
    for key in ("geocode_result", "city_resolution", "chosen_city_key", "chosen_region", "chosen_building_xy"):
        st.session_state[key] = None


# ──────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────

st.title("🏠 AlbedoNet — Albedo du toit par adresse")
st.caption(
    "Entrez une adresse francaise : l'app recupere le batiment (IGN BD TOPO), "
    "telecharge une orthophoto haute resolution (IGN, 0.20 m/px) et estime "
    "l'albedo du toit via un modele MAE pre-entraine par ville/region."
)

_init_state()

tab_single, tab_batch, tab_zone = st.tabs(
    ["📍 Adresse unique", "📋 Lot d'adresses", "🗺️ Scan de zone"]
)

with tab_single:
    address = st.text_input("Adresse", placeholder="ex. 10 rue de Rivoli, Paris")
    col_go, col_reset = st.columns([1, 1])
    go_clicked = col_go.button("Estimer l'albedo", type="primary", use_container_width=True)
    if col_reset.button("Reinitialiser", use_container_width=True):
        _reset_state()
        st.rerun()

    if go_clicked:
        if not address.strip():
            st.warning("Merci d'entrer une adresse.")
        else:
            with st.spinner("Geocodage de l'adresse..."):
                result = geocode_address(address)
            if not result.found:
                st.error(
                    "Adresse introuvable via l'API BAN. Verifiez l'orthographe, "
                    "ou renseignez une adresse plus complete (numero + rue + ville)."
                )
                _reset_state()
            else:
                st.session_state["geocode_result"] = result
                st.session_state["city_resolution"] = resolve_city_key(result)
                st.session_state["chosen_city_key"] = st.session_state["city_resolution"].city_key
                st.session_state["chosen_region"] = st.session_state["city_resolution"].region
                st.session_state["chosen_building_xy"] = None  # nouvelle adresse -> re-choisir le batiment


    # ──────────────────────────────────────────────────────────────────────────
    # Etape intermediaire : fallback region / ville si necessaire
    # ──────────────────────────────────────────────────────────────────────────

    geocode_result = st.session_state["geocode_result"]
    city_resolution = st.session_state["city_resolution"]

    if geocode_result is not None and city_resolution is not None:
        st.success(f"📍 {geocode_result.address_label}")

        # Cas 1 : match direct fiable -> rien a demander
        if city_resolution.city_key:
            st.session_state["chosen_city_key"] = city_resolution.city_key

        # Cas 2 : region deduite avec plusieurs villes candidates -> demander laquelle
        elif city_resolution.region and len(city_resolution.region_candidates) > 1:
            st.info(
                f"Commune non couverte directement, mais rattachee a la region "
                f"**{city_resolution.region}**. Plusieurs modeles sont disponibles pour "
                f"cette region : merci de preciser."
            )
            chosen = st.selectbox(
                "Modele a utiliser", city_resolution.region_candidates, key="region_city_select"
            )
            st.session_state["chosen_city_key"] = chosen

        # Cas 3 : region deduite, un seul candidat -> deja rempli par resolve_city_key
        elif city_resolution.region and len(city_resolution.region_candidates) == 1:
            st.info(
                f"Commune non couverte directement. Modele de la region "
                f"**{city_resolution.region}** utilise : `{city_resolution.region_candidates[0]}`."
            )
            st.session_state["chosen_city_key"] = city_resolution.region_candidates[0]

        # Cas 3bis : region deduite mais aucun modele dedie -> fallback generique, on previent
        elif city_resolution.region and not city_resolution.region_candidates:
            st.warning(
                f"La region **{city_resolution.region}** n'a pas de modele dedie. "
                f"Le modele generique sera utilise (moins precis pour cette zone)."
            )
            st.session_state["chosen_city_key"] = None  # -> fallback generique

        # Cas 4 : rien d'exploitable -> demander la region manuellement
        elif city_resolution.needs_user_region:
            st.warning(
                "Impossible de deduire automatiquement une region a partir de cette adresse. "
                "Merci de choisir votre region :"
            )
            region_choice = st.selectbox("Region", REGIONS_FR, key="manual_region_select")
            candidates = resolve_city_key_from_region(region_choice)
            if not candidates:
                st.warning(
                    f"Aucun modele dedie pour **{region_choice}**. "
                    f"Le modele generique sera utilise."
                )
                st.session_state["chosen_city_key"] = None
            elif len(candidates) == 1:
                st.session_state["chosen_city_key"] = candidates[0]
            else:
                chosen = st.selectbox("Modele a utiliser", candidates, key="manual_city_select")
                st.session_state["chosen_city_key"] = chosen

        # ──────────────────────────────────────────────────────────────────
        # Calcul final : recuperation batiment/ortho + patch + prediction
        # ──────────────────────────────────────────────────────────────────

        city_key = st.session_state["chosen_city_key"]
        checkpoint_path = get_checkpoint_path(city_key) if city_key else GENERIC_FALLBACK_MODEL
        model_label = city_key if city_key else "generique (paris)"

        st.divider()
        st.caption(f"Modele utilise : `{model_label}` ({checkpoint_path})")

        with st.spinner("Recuperation du batiment et de l'orthophoto IGN..."):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_dir = Path(tmp)
                try:
                    fetch_result = fetch_building_and_ortho(
                        lat=geocode_result.lat, lon=geocode_result.lon, tmp_dir=tmp_dir,
                        with_candidates=True,
                    )
                except Exception as exc:
                    st.error(f"Echec de la recuperation IGN (WFS/WMS) : {exc}")
                    fetch_result = None

                if fetch_result is not None:
                    match = fetch_result.match

                    # --- Carte : point geocode + centroide batiment matche (si dispo) ---
                    map_points = [{"lat": geocode_result.lat, "lon": geocode_result.lon}]
                    st.map(map_points, zoom=18)

                    # ── Points 1/2/5 : selection visuelle du batiment ──────
                    # Site "dense" (plusieurs batiments proches, cf.
                    # is_dense_site) OU l'utilisateur veut revenir sur son
                    # choix (bouton "changer de batiment" plus bas) -> on
                    # affiche le selecteur de candidats plutot que de faire
                    # confiance aveuglement au matching automatique.
                    show_picker = (
                        st.session_state["chosen_building_xy"] is None
                        and (fetch_result.is_dense_site or len(fetch_result.candidates) > 1)
                    )

                    if show_picker and fetch_result.area_ortho_path is not None:
                        badge_class = {"high": "badge-high", "medium": "badge-medium", "none": "badge-none"}
                        st.warning(
                            f"🏢 **{len(fetch_result.candidates)} batiment(s) trouve(s) a proximite** "
                            "-- ce site semble avoir plusieurs constructions rapprochees "
                            "(ecole, complexe industriel, etc.). Choisissez celui qui correspond "
                            "a l'adresse recherchee :"
                        )
                        cols = st.columns(min(3, len(fetch_result.candidates)))
                        for i, cand in enumerate(fetch_result.candidates):
                            with cols[i % len(cols)]:
                                thumb = crop_thumbnail(
                                    fetch_result.area_ortho_path, cand.centroid_x, cand.centroid_y
                                )
                                default_tag = " ⭐ suggere" if cand.is_default_pick else ""
                                st.markdown('<div class="albedo-card">', unsafe_allow_html=True)
                                if thumb is not None:
                                    st.image(thumb)
                                st.markdown(
                                    f'<div class="albedo-card-title">Candidat {i+1}{default_tag}</div>'
                                    f'<div class="albedo-card-meta">{cand.distance_m:.0f} m • '
                                    f'{cand.area_m2:.0f} m²</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                                if st.button("Choisir", key=f"pick_cand_{i}", use_container_width=True):
                                    st.session_state["chosen_building_xy"] = (
                                        cand.centroid_x, cand.centroid_y
                                    )
                                    st.rerun()
                        st.caption(
                            "💡 Le candidat suggere combine proximite ET surface (les grands "
                            "batiments sont favorises a distance comparable, un revetement "
                            "type Cool Roof etant plus souvent pose sur un grand toit)."
                        )
                        st.stop()

                    # ── Determination du centroide effectif a utiliser ─────
                    # Choix manuel (point 5) prioritaire si present ; sinon
                    # le pick automatique de match_building (deja pondere
                    # par proximite -- la ponderation surface complete,
                    # point 4, n'entre en jeu que via le selecteur ci-dessus
                    # sur les sites detectes comme denses).
                    manual_xy = st.session_state["chosen_building_xy"]
                    if manual_xy is not None:
                        eff_x, eff_y = manual_xy
                        st.caption("📍 Batiment choisi manuellement.")
                    else:
                        eff_x = match.centroid_x if match.centroid_x is not None else fetch_result.point_l93.x
                        eff_y = match.centroid_y if match.centroid_y is not None else fetch_result.point_l93.y

                    st.markdown(_confidence_badge_html(match.confidence), unsafe_allow_html=True)
                    if match.distance_m is not None and match.confidence != "high":
                        st.caption(f"Distance au batiment le plus proche : {match.distance_m:.1f} m")

                    # --- Filet de securite : pas de prediction automatique si confidence == none ---
                    if match.confidence == "none" and manual_xy is None:
                        st.error(
                            "Aucune correspondance fiable entre l'adresse geocodee et un batiment "
                            "de la BD TOPO n'a pu etre etablie. Aucune estimation automatique "
                            "d'albedo n'est proposee ci-dessous : verifiez visuellement l'image "
                            "et la carte, ou essayez une adresse plus precise."
                        )
                        if fetch_result.ortho_path is not None:
                            st.image(
                                str(fetch_result.ortho_path),
                                caption="Orthophoto centree sur le point geocode brut (pas de batiment matche)",
                            )
                    elif fetch_result.ortho_path is None:
                        st.error(
                            "Le batiment a bien ete identifie, mais le telechargement de "
                            "l'orthophoto IGN a echoue. Reessayez dans un instant."
                        )
                    else:
                        with st.spinner("Decoupe et normalisation du patch..."):
                            raw_patch, patch, _ratio = extract_raw_and_normalized_patch(
                                fetch_result.ortho_path, eff_x, eff_y
                            )

                        if patch is None:
                            st.error(
                                "Le patch extrait autour du batiment est invalide (hors emprise "
                                "de l'orthophoto ou trop de zones sans donnee). Aucune estimation "
                                "n'est proposee."
                            )
                        else:
                            with st.spinner("Inference du modele..."):
                                try:
                                    albedo = predict_albedo(patch, checkpoint_path)
                                except Exception as exc:
                                    st.error(f"Echec de l'inference : {exc}")
                                    albedo = None

                            if albedo is not None:
                                st.markdown(_reflectance_bar_html(albedo), unsafe_allow_html=True)
                                try:
                                    mat = classify_material(patch)
                                    st.caption(
                                        f"🧱 Materiau estime : **{mat.material}** "
                                        "(heuristique couleur, pas un classifieur ML -- indicatif)"
                                    )
                                except Exception:
                                    pass
                                st.image(
                                    str(fetch_result.ortho_path),
                                    caption="Orthophoto autour du batiment matche",
                                )
                                if match.confidence == "medium" and manual_xy is None:
                                    st.caption(
                                        "⚠️ Correspondance approximative : verifiez que le batiment "
                                        "visible sur l'image correspond bien a l'adresse recherchee."
                                    )

                                # ── Point 5 : correction a posteriori ──────
                                if fetch_result.candidates:
                                    if st.button("🔁 Changer de batiment", use_container_width=True):
                                        st.session_state["chosen_building_xy"] = None
                                        st.rerun()


with tab_batch:
    st.write(
        "Charge un fichier CSV avec une colonne `adresse` (une adresse par "
        "ligne), ou colle directement une liste d'adresses (une par ligne)."
    )
    st.caption(
        "⚡ Traitement parallelise (6 adresses en meme temps) -- le "
        "geocodage et les appels IGN sont independants entre adresses, "
        "donc plus rapide qu'un traitement un par un."
    )

    resolution_mode = st.radio(
        "Mode de resolution du modele",
        ["Automatique (rapide)", "Manuel (je choisis les cas ambigus)"],
        horizontal=True,
        help=(
            "Automatique : pour chaque adresse en region multi-candidats, "
            "le premier candidat de la region est choisi sans demander "
            "(signale dans la colonne region_ambigue du resultat). "
            "Manuel : un pre-scan liste d'abord les adresses ambigues et te "
            "laisse choisir le modele pour chacune avant de lancer le calcul."
        ),
    )
    is_manual = resolution_mode.startswith("Manuel")

    uploaded_csv = st.file_uploader("Fichier CSV", type=["csv"], key="batch_csv")
    pasted_addresses = st.text_area(
        "Ou colle une liste d'adresses (une par ligne)",
        height=150,
        placeholder="10 rue de Rivoli, Paris\n1 Place Bellecour, Lyon\n...",
        key="batch_text",
    )

    batch_addresses: list[str] = []

    if uploaded_csv is not None:
        text = uploaded_csv.getvalue().decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text))
        if reader.fieldnames and "adresse" in [f.strip().lower() for f in reader.fieldnames]:
            key_name = next(f for f in reader.fieldnames if f.strip().lower() == "adresse")
            batch_addresses = [row[key_name].strip() for row in reader if row.get(key_name, "").strip()]
        else:
            io_text = io.StringIO(text)
            batch_addresses = [line.strip() for line in io_text if line.strip()]
        st.caption(f"{len(batch_addresses)} adresse(s) detectee(s) dans le CSV.")
    elif pasted_addresses.strip():
        batch_addresses = [line.strip() for line in pasted_addresses.splitlines() if line.strip()]

    # Reinitialise le pre-scan si la liste d'adresses ou le mode change,
    # pour ne jamais lancer un traitement avec des choix perimes.
    batch_signature = (tuple(batch_addresses), is_manual)
    if st.session_state.get("batch_signature") != batch_signature:
        st.session_state["batch_signature"] = batch_signature
        st.session_state["batch_preresolved"] = None
        st.session_state["batch_overrides"] = {}

    # ──────────────────────────────────────────────────────────────────
    # Mode automatique : comportement inchange, lancement direct
    # ──────────────────────────────────────────────────────────────────
    if not is_manual:
        run_batch = st.button(
            "Lancer le scan du lot", type="primary", disabled=not batch_addresses,
            use_container_width=True, key="run_auto",
        )
        if run_batch:
            progress_bar = st.progress(0.0, text="Demarrage...")
            _last_shown_batch = {"pct": -1}

            def _on_progress(i: int, total: int, addr: str) -> None:
                pct = int(100 * i / total) if total else 100
                if pct == _last_shown_batch["pct"] and i < total:
                    return
                _last_shown_batch["pct"] = pct
                progress_bar.progress(i / total, text=f"{i}/{total} — {addr[:60]}")

            with st.spinner(f"Traitement de {len(batch_addresses)} adresse(s)..."):
                results = process_batch(batch_addresses, progress_callback=_on_progress)
            progress_bar.empty()
            st.session_state["batch_results"] = results

    # ──────────────────────────────────────────────────────────────────
    # Mode manuel : pre-scan -> selectboxes pour les cas ambigus -> lancement
    # ──────────────────────────────────────────────────────────────────
    else:
        prescan_clicked = st.button(
            "1. Analyser les adresses (geocodage + detection des cas ambigus)",
            type="secondary", disabled=not batch_addresses,
            use_container_width=True, key="run_prescan",
        )
        if prescan_clicked:
            with st.spinner(f"Analyse de {len(batch_addresses)} adresse(s)..."):
                st.session_state["batch_preresolved"] = pre_resolve_addresses(batch_addresses)
                st.session_state["batch_overrides"] = {}

        preresolved: list[CityPreResolution] | None = st.session_state.get("batch_preresolved")

        if preresolved is not None:
            n_ambiguous = sum(1 for p in preresolved if p.is_ambiguous)
            n_not_found = sum(1 for p in preresolved if not p.found)
            n_auto = len(preresolved) - n_ambiguous - n_not_found

            st.success(
                f"Analyse terminee : {n_auto} resolu(s) automatiquement, "
                f"{n_ambiguous} ambigu(s) a trancher, {n_not_found} introuvable(s)."
            )

            if n_ambiguous:
                st.write(f"**{n_ambiguous} adresse(s) ambigue(s) — choisis le modele pour chacune :**")
                for p in preresolved:
                    if not p.is_ambiguous:
                        continue
                    label = p.address_label or p.address
                    region_txt = f" (region : {p.region})" if p.region else " (aucune region deduite)"
                    options = ["generique (fallback)"] + p.candidates
                    # Defaut = premier candidat SEULEMENT si une region a ete
                    # deduite (coherent avec le mode automatique). Si aucune
                    # region n'a pu etre deduite (ex. DOM-TOM), la liste
                    # candidates contient TOUS les modeles sans rapport avec
                    # l'adresse -- le defaut sur doit alors etre "generique",
                    # jamais un choix arbitraire (bug corrige : l'ordre
                    # alphabetique faisait tomber sur "beauce" par hasard).
                    default_idx = 1 if (p.candidates and p.region) else 0
                    choice = st.selectbox(
                        f"📍 {label}{region_txt}",
                        options,
                        index=default_idx,
                        key=f"override_select_{p.index}",
                    )
                    st.session_state["batch_overrides"][p.index] = (
                        None if choice == "generique (fallback)" else choice
                    )

            run_final = st.button(
                "2. Lancer le traitement complet avec ces choix",
                type="primary", use_container_width=True, key="run_manual_final",
            )

            if run_final:
                progress_bar = st.progress(0.0, text="Demarrage...")
                _last_shown_batch2 = {"pct": -1}

                def _on_progress2(i: int, total: int, addr: str) -> None:
                    pct = int(100 * i / total) if total else 100
                    if pct == _last_shown_batch2["pct"] and i < total:
                        return
                    _last_shown_batch2["pct"] = pct
                    progress_bar.progress(i / total, text=f"{i}/{total} — {addr[:60]}")

                with st.spinner(f"Traitement de {len(batch_addresses)} adresse(s)..."):
                    results = process_batch(
                        batch_addresses,
                        overrides=st.session_state["batch_overrides"],
                        progress_callback=_on_progress2,
                    )
                progress_bar.empty()
                st.session_state["batch_results"] = results

    # ──────────────────────────────────────────────────────────────────
    # Affichage des resultats (commun aux deux modes)
    # ──────────────────────────────────────────────────────────────────
    results = st.session_state.get("batch_results")

    if results:
        rows = []
        n_ok, n_ambiguous_auto, n_error = 0, 0, 0
        for r in results:
            if r.albedo is not None:
                n_ok += 1
            if r.ambiguous_region:
                n_ambiguous_auto += 1
            if r.error:
                n_error += 1
            rows.append({
                "adresse_saisie": r.address_input,
                "adresse_geocodee": r.address_label or "",
                "lat": r.lat,
                "lon": r.lon,
                "modele_utilise": r.city_key_used or "",
                "region_ambigue": "oui" if r.ambiguous_region else "",
                "confiance_batiment": r.confidence or "",
                "distance_batiment_m": round(r.distance_m, 1) if r.distance_m is not None else "",
                "albedo": round(r.albedo, 3) if r.albedo is not None else "",
                "materiau_estime": r.material or "",
                "erreur": r.error or "",
            })

        st.divider()
        st.success(
            f"Termine : {n_ok}/{len(results)} albedo(s) estime(s), "
            f"{n_ambiguous_auto} region(s) ambigue(s) resolue(s) automatiquement, "
            f"{n_error} echec(s)."
        )

        st.dataframe(rows, use_container_width=True)
        st.caption(
            "🧱 `materiau_estime` : heuristique couleur (pas un classifieur ML) -- indicatif uniquement."
        )

        # ── Galerie visuelle (verification directe des batiments matches) ──
        # Utilise thumbnail_png, genere gratuitement depuis le patch deja
        # extrait pour l'inference (pas de re-telechargement/redecoupe) --
        # reste rapide meme sur des lots de plusieurs dizaines/centaines
        # d'adresses.
        results_with_thumb = [r for r in results if r.thumbnail_png is not None]
        if results_with_thumb:
            with st.expander(
                f"🖼️ Galerie visuelle ({len(results_with_thumb)}/{len(results)} "
                "batiment(s) avec vignette)", expanded=False,
            ):
                n_cols = 4
                cols = st.columns(n_cols)
                for i, r in enumerate(results_with_thumb):
                    with cols[i % n_cols]:
                        st.markdown('<div class="albedo-card">', unsafe_allow_html=True)
                        st.image(r.thumbnail_png)
                        badge_class = {
                            "high": "badge-high", "medium": "badge-medium", "none": "badge-none",
                        }.get(r.confidence, "badge-medium")
                        label = (r.address_label or r.address_input)[:40]
                        albedo_txt = f"{r.albedo:.3f}" if r.albedo is not None else "?"
                        st.markdown(
                            f'<div class="albedo-card-title">{label}</div>'
                            f'<span class="albedo-badge {badge_class}">{r.confidence or "?"}</span>'
                            f'<div class="albedo-card-meta">albedo {albedo_txt} • '
                            f'{r.material or "?"}</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("</div>", unsafe_allow_html=True)

        csv_buffer = io.StringIO()
        writer = csv.DictWriter(csv_buffer, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

        st.download_button(
            "Telecharger les resultats (CSV)",
            data=csv_buffer.getvalue(),
            file_name="albedo_resultats_lot.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_results",
        )

        if n_ambiguous_auto:
            st.caption(
                "ℹ️ Pour les adresses en region ambigue resolues automatiquement "
                "(mode automatique, ou non tranchees en mode manuel), le premier "
                "candidat de la region a ete utilise. Repasse en mode manuel "
                "pour choisir precisement."
            )

with tab_zone:
    st.write(
        "Scanne tous les batiments BD TOPO autour d'une adresse centre "
        "(quartier, ville) et estime l'albedo de chacun. L'orthophoto est "
        "telechargee par tuiles (pas un appel par batiment), donc le temps "
        "de calcul augmente surtout avec le RAYON, pas juste le nombre "
        "de batiments."
    )
    st.caption(
        f"Rayon maximum : {int(MAX_ZONE_RADIUS_M)} m. Au-dela, le temps de "
        "scan devient long -- commence par une zone modeste (200-400 m) "
        "pour un premier essai, augmente ensuite si besoin. "
        "⚡ Telechargement des tuiles et inference par batiment parallelises "
        "(6 threads)."
    )

    zone_center = st.text_input(
        "Adresse centre de la zone",
        placeholder="ex. Place Bellecour, Lyon",
        key="zone_center_address",
    )
    zone_radius = st.slider(
        "Rayon (m)", min_value=50, max_value=int(MAX_ZONE_RADIUS_M),
        value=250, step=50, key="zone_radius",
    )

    run_zone = st.button(
        "Lancer le scan de zone", type="primary",
        disabled=not zone_center.strip(), use_container_width=True,
        key="run_zone_scan",
    )

    if run_zone:
        progress_bar = st.progress(0.0, text="Demarrage...")

        # NOTE 09/07/2026 : throttling ajoute ici. Sans throttling, l'etape
        # "inference" appelle progress_bar.progress() une fois PAR BATIMENT
        # (cf. scan_zone -> as_completed dans zone_scan.py). Sur une zone
        # dense (ex. 1500m sur Lyon = plusieurs milliers de batiments), ca
        # envoie des milliers de mises a jour websocket en rafale, plus vite
        # que le navigateur ne peut les absorber -> la connexion finit
        # consideree fermee pendant que Streamlit essaie encore d'envoyer
        # (erreur "websocket.send ... after websocket.close" observee en
        # usage reel, scan qui semble se figer). On ne met a jour l'affichage
        # que tous les ~1% de progression (ou a chaque etape/fin), ce qui
        # reste fluide visuellement tout en divisant le volume de messages
        # par ~total/100.
        _last_shown = {"pct": -1}

        def _on_zone_progress(step: str, i: int, total: int) -> None:
            frac = (i / total) if total else 0.0
            pct = int(frac * 100)
            is_last = (i >= total) if total else True
            if step == "inference" and pct == _last_shown["pct"] and not is_last:
                return  # meme palier de 1% deja affiche, on saute cette mise a jour
            _last_shown["pct"] = pct
            labels = {
                "geocodage": "Geocodage du centre...",
                "batiments": "Recuperation des batiments (WFS)...",
                "tuiles": f"Telechargement des tuiles orthophoto ({i}/{total})...",
                "inference": f"Inference par batiment ({i}/{total})...",
            }
            progress_bar.progress(min(frac, 1.0), text=labels.get(step, step))

        with st.spinner("Scan de zone en cours..."):
            zone_result = scan_zone(
                zone_center.strip(), float(zone_radius),
                progress_callback=_on_zone_progress,
            )
        progress_bar.empty()
        st.session_state["zone_result"] = zone_result

    zone_result = st.session_state.get("zone_result")

    if zone_result is not None:
        if zone_result.error and not zone_result.buildings:
            st.error(zone_result.error)
        else:
            st.success(
                f"{zone_result.n_buildings} batiment(s) trouve(s), "
                f"{zone_result.n_tiles} tuile(s) orthophoto telechargee(s), "
                f"modele utilise : `{zone_result.city_key_used}`."
            )

            ok_results = [b for b in zone_result.buildings if b.albedo is not None]
            err_results = [b for b in zone_result.buildings if b.albedo is None]
            stats = zone_result.stats

            if stats:
                st.subheader("Statistiques")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Moyenne", f"{stats.mean:.3f}")
                col2.metric("Mediane", f"{stats.median:.3f}")
                col3.metric("Ecart-type", f"{stats.std:.3f}")
                col4.metric("Moy. ponderee (surface)", f"{stats.area_weighted_mean:.3f}")

                col5, col6, col7 = st.columns(3)
                col5.metric("Min", f"{stats.min:.3f}")
                col6.metric("Max", f"{stats.max:.3f}")
                col7.metric("Batiments valides", f"{stats.n}/{zone_result.n_buildings}")

                st.caption(
                    "La moyenne ponderee par surface donne plus de poids aux "
                    "grands batiments (utile si tu veux une estimation "
                    "representative de la surface totale de toiture, plutot "
                    "qu'un batiment = un vote)."
                )

                st.write("**Distribution des albedos**")
                hist_data = {label: count for label, count in stats.histogram}
                st.bar_chart(hist_data)

                threshold_used = zone_result.dark_threshold_used or DARK_ROOF_ALBEDO_THRESHOLD
                bright_threshold_used = zone_result.bright_threshold_used

                # dark_results sert uniquement aux messages/stats ci-dessous
                # (nombre de toits sous le seuil) -- la heatmap ponderee par
                # densite a ete retiree (cf. note plus bas), donc plus besoin
                # de construire heat_data.
                dark_results = [b for b in ok_results if b.albedo < threshold_used]
                heat_suspects = [b for b in ok_results if b.heat_island_suspect]
                cool_suspects = [b for b in ok_results if b.cool_island_suspect]

                st.write("**Carte de chaleur — clusters de toits sombres**")
                st.caption(
                    f"Seuil de noirceur applique pour cette zone : {threshold_used:.2f} "
                    f"(calibre pour la ville detectee -- cf. heat_island_calibration.py -- "
                    "ou valeur par defaut si ville non encore calibree). "
                    f"Un batiment est signale suspect ilot de chaleur s'il est sous ce "
                    f"seuil ET a au moins {MIN_DARK_NEIGHBORS} autre(s) toit(s) aussi "
                    "sombre(s) a proximite (regroupement spatial, pas un batiment isole). "
                    "Signal indicatif base sur l'albedo estime, pas une mesure de "
                    "temperature reelle. "
                    + (
                        f"🔴 {len(heat_suspects)} batiment(s) signale(s) dans cette zone "
                        f"({len(dark_results)} toit(s) sous le seuil au total, dont certains "
                        "isoles donc non retenus)."
                        if heat_suspects else
                        f"Aucun cluster confirme ({len(dark_results)} toit(s) isole(s) sous "
                        "le seuil, mais pas assez regroupes pour former un cluster)."
                        if dark_results else
                        "Aucun toit sous le seuil de noirceur dans cette zone."
                    )
                )
                if bright_threshold_used:
                    st.caption(
                        f"🔵 Ilots de fraicheur (clusters de toits CLAIRS, seuil "
                        f"{bright_threshold_used:.2f}) : {len(cool_suspects)} batiment(s) "
                        "signale(s) -- logique symetrique (seuil absolu + clustering), "
                        "meme niveau d'incertitude que pour la chaleur."
                    )

                if not dark_results and not cool_suspects:
                    st.info(
                        "Pas de carte a afficher : aucun batiment de cette zone n'est "
                        f"sous le seuil de {threshold_used:.2f} (min observe : "
                        f"{min(b.albedo for b in ok_results):.3f})."
                    )
                else:
                    # NOTE 09/07/2026 : HeatmapLayer retiree ici. Elle avait ete
                    # documentee comme "abandonnee" dans METHODOLOGIE.md (§7 --
                    # peu lisible, effet de densite trompeur) mais etait restee
                    # active dans le code, superposee aux points nets -- d'ou le
                    # halo jaune/orange flou observe en usage reel. Seuls les
                    # points rouges/bleus (clusters confirmes) sont conserves.
                    layers = []
                    if heat_suspects:
                        # Points rouges = batiments confirmes suspects ilot de
                        # chaleur (cluster reel, cf. flag_heat_island_suspects),
                        # pas juste "la zone a l'air un peu sombre".
                        layers.append(pdk.Layer(
                            "ScatterplotLayer",
                            data=[
                                {"lat": b.centroid_lat, "lon": b.centroid_lon}
                                for b in heat_suspects
                            ],
                            get_position="[lon, lat]",
                            get_radius=8,
                            get_fill_color=[230, 57, 70, 220],
                            pickable=False,
                        ))
                    if cool_suspects:
                        # Points bleus pour les clusters de toits CLAIRS
                        # (ilots de fraicheur), symetrique visuel des points
                        # rouges ci-dessus.
                        layers.append(pdk.Layer(
                            "ScatterplotLayer",
                            data=[
                                {"lat": b.centroid_lat, "lon": b.centroid_lon}
                                for b in cool_suspects
                            ],
                            get_position="[lon, lat]",
                            get_radius=8,
                            get_fill_color=[69, 123, 157, 220],
                            pickable=False,
                        ))
                    view_state = pdk.ViewState(
                        latitude=zone_result.center_lat,
                        longitude=zone_result.center_lon,
                        zoom=16,
                    )
                    st.pydeck_chart(pdk.Deck(
                        layers=layers,
                        initial_view_state=view_state,
                        map_style=None,
                    ))

                # ── Listes d'adresses + vignettes (verification visuelle) ──
                # Limite volontaire : on ne fait le reverse-geocoding QUE
                # pour les suspects confirmes (pas tous les toits sombres),
                # pour eviter de spammer l'API BAN sur des zones a forte
                # densite. Plafond de securite supplementaire si jamais un
                # cluster est enorme. Les vignettes (thumbnail_png) ne
                # coutent rien de plus (deja generees pendant l'inference),
                # donc affichees ici sans impact sur la vitesse du scan.
                MAX_ADDRESSES_TO_RESOLVE = 30

                def _render_suspect_gallery(suspects: list, icon: str, label_kind: str) -> None:
                    to_resolve = suspects[:MAX_ADDRESSES_TO_RESOLVE]
                    n_cols = 4
                    cols = st.columns(n_cols)
                    with st.spinner("Recherche des adresses..."):
                        for i, b in enumerate(to_resolve):
                            addr_label = reverse_geocode(b.centroid_lat, b.centroid_lon)
                            with cols[i % n_cols]:
                                st.markdown('<div class="albedo-card">', unsafe_allow_html=True)
                                if b.thumbnail_png is not None:
                                    st.image(b.thumbnail_png)
                                st.markdown(
                                    f'<div class="albedo-card-title">{icon} '
                                    f'{(addr_label or f"({b.centroid_lat:.5f}, {b.centroid_lon:.5f})")[:35]}</div>'
                                    f'<div class="albedo-card-meta">albedo {b.albedo:.3f} • '
                                    f'{b.material or "?"}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown("</div>", unsafe_allow_html=True)
                    if len(suspects) > MAX_ADDRESSES_TO_RESOLVE:
                        st.caption(
                            f"... et {len(suspects) - MAX_ADDRESSES_TO_RESOLVE} de plus "
                            f"({label_kind}, non affiches -- cf. export CSV pour la liste complete)."
                        )

                if heat_suspects:
                    with st.expander(f"🔴 {len(heat_suspects)} batiment(s) suspect(s) ilot de chaleur"):
                        _render_suspect_gallery(heat_suspects, "🔴", "ilots de chaleur")

                if cool_suspects:
                    with st.expander(f"🔵 {len(cool_suspects)} batiment(s) suspect(s) ilot de fraicheur"):
                        _render_suspect_gallery(cool_suspects, "🔵", "ilots de fraicheur")

            if err_results:
                st.caption(
                    f"{len(err_results)} batiment(s) sans prediction "
                    "(hors emprise tuile / trop de nodata / echec inference)."
                )

            csv_buffer = io.StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow([
                "building_id", "centroid_lat", "centroid_lon",
                "area_m2", "albedo", "materiau_estime",
                "ilot_chaleur_suspect", "ilot_fraicheur_suspect", "erreur",
            ])
            for b in zone_result.buildings:
                writer.writerow([
                    b.building_id, b.centroid_lat, b.centroid_lon,
                    round(b.area_m2, 1),
                    round(b.albedo, 4) if b.albedo is not None else "",
                    b.material or "",
                    "oui" if b.heat_island_suspect else "",
                    "oui" if b.cool_island_suspect else "",
                    b.error or "",
                ])
            st.download_button(
                "Telecharger les resultats (CSV)",
                data=csv_buffer.getvalue(),
                file_name="albedo_resultats_zone.csv",
                mime="text/csv",
                use_container_width=True,
            )