# checkpoint_integrity.py
"""
Verification d'integrite par hash des checkpoints .pt, meme pattern que
celui deja utilise pour le classifieur materiau (cf. versioning.py).

Usage :
    # 1. Une fois les 14 checkpoints en place localement/en prod :
    python checkpoint_integrity.py generate

    # 2. A committer : checkpoints/checkpoint_hashes.json

    # 3. Au demarrage de l'app (voir integration dans main.py plus bas) :
    python checkpoint_integrity.py verify
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from config import CITY_MODELS  # source de verite des cles ville

CHECKPOINTS_DIR = Path("checkpoints")
REGISTRY_PATH = CHECKPOINTS_DIR / "checkpoint_hashes.json"
CHECKPOINT_FILENAME = "best_finetune.pt"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_path(city_key: str) -> Path:
    return CHECKPOINTS_DIR / city_key / CHECKPOINT_FILENAME


def generate() -> None:
    """Calcule le hash de chaque checkpoint present et ecrit le registre."""
    registry: dict[str, str] = {}
    missing: list[str] = []
    for city_key in CITY_MODELS:
        p = _checkpoint_path(city_key)
        if not p.exists():
            missing.append(city_key)
            continue
        registry[city_key] = _sha256(p)

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(f"Registre ecrit : {REGISTRY_PATH} ({len(registry)} checkpoints)")
    if missing:
        print(f"ATTENTION - checkpoints absents (non inclus dans le registre) : {missing}")


def verify() -> None:
    """
    Verifie chaque checkpoint present contre le registre.
    Leve SystemExit(1) en cas de divergence de hash (corruption / mauvaise
    version) -- distinct d'un checkpoint simplement absent, qui reste gere
    par test_city_model_smoke.py / le comportement existant au chargement.
    """
    if not REGISTRY_PATH.exists():
        print(f"Pas de registre trouve a {REGISTRY_PATH} -- verification ignoree.")
        return

    expected: dict[str, str] = json.loads(REGISTRY_PATH.read_text())

    mismatches: list[str] = []
    missing: list[str] = []
    for city_key, expected_hash in expected.items():
        p = _checkpoint_path(city_key)
        if not p.exists():
            missing.append(city_key)
            continue
        actual_hash = _sha256(p)
        if actual_hash != expected_hash:
            mismatches.append(
                f"{city_key}: attendu {expected_hash[:12]}..., trouve {actual_hash[:12]}..."
            )

    if missing:
        print(f"Checkpoints absents (geres separement au chargement) : {missing}")

    if mismatches:
        print("ECHEC verification integrite checkpoints :")
        for m in mismatches:
            print(f"  - {m}")
        raise SystemExit(1)

    print(f"OK - {len(expected) - len(missing)} checkpoints verifies, hash conformes.")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("generate", "verify"):
        print(__doc__)
        raise SystemExit(2)
    {"generate": generate, "verify": verify}[sys.argv[1]]()
