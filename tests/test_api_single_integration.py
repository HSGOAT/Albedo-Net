"""
tests/test_api_single_integration.py — AlbedoNet App
=======================================================
Tests d'integration legers sur les endpoints de main.py (Tier 3, point 8).

Contrairement aux audits existants (test_city_key_consistency.py,
test_city_model_smoke.py, test_material_classifier_regression.py,
test_split_isolation.py, test_confidence_materials_sync.py) qui couvrent
les donnees et les modeles, CE fichier couvre la glue FastAPI elle-meme :
routing, codes de statut, forme du JSON retourne, gestion des cas d'erreur
cote endpoint. Tout ce qui touche au reseau (IGN, geocodage) ou au modele
(inference, classification materiau) est mocke -- ces tests ne font AUCUN
appel reseau et ne chargent AUCUN checkpoint .pt, ils doivent donc passer
meme sans connexion internet ni checkpoints presents (contrairement a
test_city_model_smoke.py, qui lui a besoin des vrais checkpoints).

Strategie de mock : on patche les noms tels qu'importes DANS main.py
(ex. "main.geocode_address", pas "geocoding.geocode_address") -- main.py
fait `from geocoding import geocode_address`, donc le nom vit dans
l'espace de noms de main, c'est la qu'il faut le remplacer.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client():
    return TestClient(main.app)


# ──────────────────────────────────────────────────────────────────────────
# Doubles de test -- imitent juste les attributs lus par main.py, pas les
# vraies dataclasses (pas besoin d'importer geocoding/ign_fetch/materials
# pour construire ces objets).
# ──────────────────────────────────────────────────────────────────────────

def _fake_geocode_found(**overrides):
    base = dict(
        found=True, address_label="12 Rue de la Paix, 75002 Paris",
        lat=48.8566, lon=2.3522,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_geocode_not_found():
    return SimpleNamespace(found=False)


def _fake_fetch_result(confidence="high", ortho_path="/tmp/fake_ortho.tif", **overrides):
    match = SimpleNamespace(
        confidence=confidence, distance_m=1.2,
        centroid_x=650000.0, centroid_y=6860000.0,
    )
    point_l93 = SimpleNamespace(x=650000.0, y=6860000.0)
    base = dict(match=match, ortho_path=ortho_path, point_l93=point_l93)
    base.update(overrides)
    return SimpleNamespace(**base)


def _fake_material_result(material="zinc", proba_max=0.87):
    return SimpleNamespace(material=material, proba_max=proba_max)


def _fake_calibration(albedo_corrige=0.22, albedo_brut=0.20, shadow_fraction=0.05):
    return SimpleNamespace(
        albedo_corrige=albedo_corrige, albedo_brut=albedo_brut,
        shadow_fraction=shadow_fraction,
    )


# ──────────────────────────────────────────────────────────────────────────
# /api/single
# ──────────────────────────────────────────────────────────────────────────

def test_single_empty_address_returns_400(client):
    resp = client.get("/api/single", params={"address": "   "})
    assert resp.status_code == 400


def test_single_address_not_found(client):
    with patch("main.geocode_address", return_value=_fake_geocode_not_found()):
        resp = client.get("/api/single", params={"address": "adresse inexistante xyz"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert "error" in body


def test_single_happy_path(client):
    """Chemin nominal complet : geocodage OK, batiment matche, patch valide,
    inference OK, materiau OK, calibration OK -> reponse riche et coherente."""
    with (
        patch("main.geocode_address", return_value=_fake_geocode_found()),
        patch("main._resolve_city_key_auto", return_value="paris"),
        patch("main.fetch_building_and_ortho", return_value=_fake_fetch_result()),
        patch(
            "main.extract_raw_and_normalized_patch",
            return_value=("raw_patch_stub", "normalized_patch_stub", 0.0),
        ),
        patch("main.predict_albedo", return_value=0.20),
        patch("main.classify_material", return_value=_fake_material_result()),
        patch("main.calibrate_albedo", return_value=_fake_calibration()),
        patch("main._ortho_to_data_url", return_value="data:image/tiff;base64,AAAA"),
    ):
        resp = client.get("/api/single", params={"address": "12 Rue de la Paix, Paris"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is True
    assert body["error"] is None
    assert body["albedo"] == pytest.approx(0.22)
    assert body["albedo_brut"] == pytest.approx(0.20)
    assert body["material"] == "zinc"
    assert body["material_proba_max"] == pytest.approx(0.87)
    assert body["material_is_heuristic"] is False
    assert body["model_used"] == "paris"


def test_single_no_reliable_building_match(client):
    """match.confidence == 'none' -> l'endpoint doit renvoyer albedo=None et
    un message d'erreur explicite, PAS planter ni renvoyer un albedo
    fantaisiste."""
    with (
        patch("main.geocode_address", return_value=_fake_geocode_found()),
        patch("main._resolve_city_key_auto", return_value="paris"),
        patch(
            "main.fetch_building_and_ortho",
            return_value=_fake_fetch_result(confidence="none"),
        ),
        patch("main._ortho_to_data_url", return_value=None),
    ):
        resp = client.get("/api/single", params={"address": "adresse ambigue"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["albedo"] is None
    assert body["confidence"] == "none"
    assert "correspondance" in body["error"].lower()


def test_single_ign_fetch_failure_returns_error_not_500(client):
    """Un echec reseau IGN (WFS/WMS) doit produire une reponse 200 avec un
    champ error explicite -- pas une exception non geree qui remonterait en
    500. C'est le comportement documente de main.py (le bloc except autour
    de fetch_building_and_ortho)."""
    with (
        patch("main.geocode_address", return_value=_fake_geocode_found()),
        patch("main._resolve_city_key_auto", return_value="paris"),
        patch(
            "main.fetch_building_and_ortho",
            side_effect=RuntimeError("Timeout WFS IGN"),
        ),
    ):
        resp = client.get("/api/single", params={"address": "12 Rue de la Paix, Paris"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["albedo"] is None
    assert "Timeout WFS IGN" in body["error"]


def test_single_inference_failure_returns_error_not_500(client):
    """Meme principe pour un echec au moment de l'inference (checkpoint
    corrompu, OOM, etc.) : reponse 200 avec error, pas de 500."""
    with (
        patch("main.geocode_address", return_value=_fake_geocode_found()),
        patch("main._resolve_city_key_auto", return_value="paris"),
        patch("main.fetch_building_and_ortho", return_value=_fake_fetch_result()),
        patch(
            "main.extract_raw_and_normalized_patch",
            return_value=("raw_patch_stub", "normalized_patch_stub", 0.0),
        ),
        patch("main.predict_albedo", side_effect=RuntimeError("checkpoint corrompu")),
        patch("main._ortho_to_data_url", return_value=None),
    ):
        resp = client.get("/api/single", params={"address": "12 Rue de la Paix, Paris"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["albedo"] is None
    assert "checkpoint corrompu" in body["error"]


def test_single_shadow_calibration_failure_falls_back_to_raw_albedo(client):
    """Si calibrate_albedo plante, l'endpoint doit quand meme renvoyer
    l'albedo brut (fallback documente dans main.py), pas planter."""
    with (
        patch("main.geocode_address", return_value=_fake_geocode_found()),
        patch("main._resolve_city_key_auto", return_value="paris"),
        patch("main.fetch_building_and_ortho", return_value=_fake_fetch_result()),
        patch(
            "main.extract_raw_and_normalized_patch",
            return_value=("raw_patch_stub", "normalized_patch_stub", 0.0),
        ),
        patch("main.predict_albedo", return_value=0.31),
        patch("main.classify_material", return_value=_fake_material_result()),
        patch("main.calibrate_albedo", side_effect=RuntimeError("erreur calibration")),
        patch("main._ortho_to_data_url", return_value=None),
    ):
        resp = client.get("/api/single", params={"address": "12 Rue de la Paix, Paris"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["albedo"] == pytest.approx(0.31)
    assert body["albedo_brut"] == pytest.approx(0.31)
    assert body["shadow_fraction"] is None


# ──────────────────────────────────────────────────────────────────────────
# /api/batch
# ──────────────────────────────────────────────────────────────────────────

def test_batch_empty_list_returns_400(client):
    resp = client.post("/api/batch", json={"addresses": []})
    assert resp.status_code == 400


def test_batch_happy_path(client):
    fake_result = SimpleNamespace(
        address_input="1 rue Test", address_label="1 Rue Test, Paris",
        lat=48.85, lon=2.35, city_key_used="paris", ambiguous_region=False,
        confidence="high", distance_m=0.5, albedo=0.25, material="zinc",
        material_proba_max=0.9, error=None, thumbnail_png=None,
    )
    with patch("main.process_batch", return_value=[fake_result]):
        resp = client.post("/api/batch", json={"addresses": ["1 rue Test"]})

    assert resp.status_code == 200
    body = resp.json()
    assert len(body["results"]) == 1
    assert body["results"][0]["model_used"] == "paris"
    assert body["results"][0]["albedo"] == pytest.approx(0.25)


# ──────────────────────────────────────────────────────────────────────────
# /api/zone
# ──────────────────────────────────────────────────────────────────────────

def test_zone_empty_address_returns_400(client):
    resp = client.get("/api/zone", params={"address": "  "})
    assert resp.status_code == 400


def test_zone_error_without_buildings_returns_error_payload(client):
    fake_result = SimpleNamespace(
        error="Aucun batiment BD TOPO trouve dans cette zone.",
        buildings=[],
    )
    with patch("main.scan_zone", return_value=fake_result):
        resp = client.get("/api/zone", params={"address": "zone vide", "radius_m": 100})

    assert resp.status_code == 200
    body = resp.json()
    assert body["error"] is not None
