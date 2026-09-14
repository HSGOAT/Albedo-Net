"""
inference.py

Charge un checkpoint .pt (encoder_state_dict, head_state_dict, config, band_stats)
et effectue la prediction d'albedo a partir d'un patch RGB (3, 64, 64) deja
normalise en min-max [0,1] par patch_extraction.py.

Aucun re-entrainement ici : uniquement forward pass en mode eval.
"""

import logging

import torch
import torch.nn as nn
import numpy as np

from models.mae_encoder import MAEEncoder
from utils.band_normalizer import BandNormalizer

logger = logging.getLogger("albedo.app.inference")


class AlbedoHead(nn.Module):
    """Tete MLP de regression albedo, architecture identique a celle entrainee
    dans Finetune_mae.py (768 -> hidden1 -> hidden1//2 -> 1), sinon
    load_state_dict echoue (cf. bug KeyError/size-mismatch du 08/07/2026) :

    Linear -> LayerNorm -> ReLU -> Dropout -> Linear -> LayerNorm -> ReLU
    -> Dropout -> Linear, puis Sigmoid appliquee HORS du nn.Sequential dans
    forward() (et non comme derniere couche de self.net).

    hidden1 et dropout doivent venir du config du checkpoint
    (cles "head_hidden"/"head_dropout" dans args, cf. Finetune_mae.py
    argparse) : un checkpoint peut avoir ete entraine avec hidden1 != 256
    (l'erreur size-mismatch observee correspondait a hidden1=128).
    """

    def __init__(
        self,
        in_dim: int = 768,
        hidden1: int = 256,
        dropout: float = 0.3,
        norm_type: str = "layernorm",
    ):
        super().__init__()
        hidden2 = hidden1 // 2

        def _norm(dim: int) -> nn.Module:
            if norm_type == "batchnorm":
                return nn.BatchNorm1d(dim)
            return nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            _norm(hidden1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            _norm(hidden2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Sigmoid hors du Sequential (comme dans Finetune_mae.py), pas de
        # squeeze ici : predict() gere la forme (B,1) -> float.
        return torch.sigmoid(self.net(x))


class AlbedoPredictor:
    """
    Wrapper haut niveau : charge un checkpoint une seule fois (cache modele)
    et expose predict(patch) -> float.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.config = checkpoint.get("config", {})
        if not self.config:
            # (10/07/2026) Avant ce warning, un checkpoint sans cle "config"
            # chargeait "silencieusement" avec les valeurs par defaut
            # (img_size=64, head_hidden=256, etc.) -- si ces defauts NE
            # correspondent PAS a l'architecture reelle du checkpoint mais
            # que les dimensions collent par coincidence (ex. meme
            # head_hidden par hasard), load_state_dict() ne leve AUCUNE
            # erreur : le modele "marche" mais peut etre mal configure sans
            # aucun signal. Ce warning ne remplace pas une vraie validation
            # de schema, mais rend au moins le cas visible dans les logs
            # plutot que totalement silencieux.
            logger.warning(
                "Checkpoint '%s' ne contient pas de cle 'config' -- "
                "utilisation des valeurs par defaut (img_size=64, "
                "patch_size=8, embed_dim=384, head_hidden=256, "
                "head_dropout=0.3, ...). Si ce checkpoint a ete entraine "
                "avec des hyperparametres differents, le modele chargera "
                "sans erreur mais avec une architecture potentiellement "
                "fausse (mismatch silencieux si les dimensions collent par "
                "coincidence). Verifier explicitement ce checkpoint.",
                checkpoint_path,
            )
        band_stats = checkpoint["band_stats"]

        # Les checkpoints produits par Finetune_mae.py (save_checkpoint) stockent
        # band_stats avec les cles "mean"/"std" (singulier, cf. construction du
        # dict juste avant l'appel a save_checkpoint dans Finetune_mae.py), et
        # non "means"/"stds". BandNormalizer, lui, attend means=/stds= dans son
        # constructeur : on fait donc le mapping explicitement ici plutot que
        # de supposer que les cles du checkpoint correspondent 1:1 aux noms
        # d'arguments du constructeur.
        self.normalizer = BandNormalizer(
            means=band_stats["mean"],
            stds=band_stats["std"],
        )

        # MAEEncoder construit lui-meme son ViTSmall interne : on ne passe
        # jamais de backbone deja instancie, seulement les hyperparametres.
        # mask_ratio/mask_strategy n'ont aucun effet ici car on force
        # mask_override=zeros (aucun masquage) a l'inference.
        self.encoder = MAEEncoder(
            img_size=self.config.get("img_size", 64),
            patch_size=self.config.get("patch_size", 8),
            in_chans=self.config.get("in_chans", 3),
            embed_dim=self.config.get("embed_dim", 384),
            depth=self.config.get("depth", 12),
            num_heads=self.config.get("num_heads", 6),
            mlp_ratio=self.config.get("mlp_ratio", 4.0),
            drop_path_rate=self.config.get("drop_path_rate", 0.1),
        )
        self.encoder.load_state_dict(checkpoint["encoder_state_dict"])
        self.encoder.to(self.device).eval()

        # hidden1/dropout doivent correspondre a ce qui a ete utilise pendant
        # l'entrainement (args.head_hidden / args.head_dropout dans
        # Finetune_mae.py), sinon load_state_dict echoue en size-mismatch.
        # Valeurs par defaut alignees sur les defauts argparse de
        # Finetune_mae.py au cas ou un vieux checkpoint ne stockerait pas ces
        # cles dans son config.
        # Detection automatique du type de norme (LayerNorm vs BatchNorm1d) :
        # certains checkpoints (ex. grenoble/best_finetune.pt) ont ete
        # entraines avec BatchNorm1d, qui possede des buffers
        # running_mean/running_var/num_batches_tracked absents de LayerNorm.
        # Charger le mauvais type leve un RuntimeError "Unexpected key(s)".
        head_state = checkpoint["head_state_dict"]
        norm_type = "batchnorm" if any(
            k.endswith("running_mean") for k in head_state.keys()
        ) else "layernorm"

        self.head = AlbedoHead(
            in_dim=768,
            hidden1=self.config.get("head_hidden", 256),
            dropout=self.config.get("head_dropout", 0.3),
            norm_type=norm_type,
        )
        self.head.load_state_dict(head_state)
        self.head.to(self.device).eval()

    @torch.no_grad()
    def predict(self, patch: np.ndarray) -> float:
        """
        patch : ndarray (3, 64, 64), float32, deja normalise min-max [0,1]
                (fait par patch_extraction.py, AVANT le z-score ci-dessous).

        Retourne l'albedo predit (float entre 0 et 1).
        """
        if patch.shape != (3, 64, 64):
            raise ValueError(
                f"Patch attendu de forme (3, 64, 64), recu {patch.shape}"
            )

        tensor = torch.from_numpy(patch).float().unsqueeze(0).to(self.device)  # (1,3,64,64)

        # Normalisation z-score avec les stats du pretrain (stockees dans le checkpoint)
        tensor = self.normalizer.normalize(tensor)

        # Pas de masking en inference : mask_override tout a False (True=masque
        # dans la convention de MAEEncoder) => aucun token masque, tous visibles.
        patch_size = self.config.get("patch_size", 8)
        num_patches = (64 // patch_size) ** 2  # 64 tokens
        mask_override = torch.zeros(1, num_patches, dtype=torch.bool, device=self.device)

        # forward retourne (latent, mask, ids_restore) ; latent = (1, len_keep, 384)
        # avec len_keep=64 puisque mask_override ne masque rien ici.
        latent, _mask, _ids_restore = self.encoder(tensor, mask_override=mask_override)

        pooled = self._pool_cls_mean(latent)  # (1, 768)

        albedo = self.head(pooled)  # (1, 1)

        return float(albedo.item())

    @torch.no_grad()
    def predict_batch(self, patches) -> list[float]:
        """
        patches : soit un ndarray (N, 3, 64, 64) deja empile, soit une liste/
                   sequence de N ndarrays (3, 64, 64) -- zone_scan.py appelle
                   predict_albedo_batch() avec une liste Python
                   (chunk_patches = [patch for _, patch in chunk]), pas un
                   ndarray deja stacke : on fait le np.stack() ici plutot que
                   d'exiger que l'appelant le fasse, sinon patches.ndim leve
                   une AttributeError sur une liste (bug du 10/07/2026 :
                   toute la zone scan finissait en "echec inference (batch)"
                   pour CHAQUE batiment, l'exception cassant tout le chunk
                   d'un coup avant meme d'atteindre le forward pass).
                   Chaque patch : float32, deja normalise min-max [0,1]
                   (fait par patch_extraction.py, AVANT le z-score ci-dessous).

        Retourne une liste de N albedos predits (float Python natif entre 0
        et 1 chacun -- PAS des numpy.float32). Meme contrat de type que
        predict(), qui fait deja float(albedo.item()) pour la meme raison :
        zone_scan.py stocke ces valeurs dans BuildingResult.albedo, qui finit
        serialise en JSON/GeoJSON (cf. geo_export.to_geojson_bytes ->
        gdf.to_json() -> json.dumps). json.dumps ne sait pas serialiser
        numpy.float32 (TypeError "Object of type float32 is not JSON
        serializable", bug du 10/07/2026) -- .tolist() convertit chaque
        element du ndarray en float Python natif, contrairement a
        list(ndarray) qui garde des numpy.float32 a l'interieur.

        Equivalent a appeler predict() N fois, mais en un seul forward pass
        (encoder + head) sur tout le batch -- plus efficace pour zone_scan.py
        qui doit scorer beaucoup de patches d'un coup.
        """
        if not isinstance(patches, np.ndarray):
            if len(patches) == 0:
                return []
            patches = np.stack(patches, axis=0)

        if patches.ndim != 4 or patches.shape[1:] != (3, 64, 64):
            raise ValueError(
                f"Patches attendus de forme (N, 3, 64, 64), recu {patches.shape}"
            )

        n = patches.shape[0]
        tensor = torch.from_numpy(patches).float().to(self.device)  # (N,3,64,64)

        # Meme normalisation z-score que predict(), appliquee au batch entier.
        tensor = self.normalizer.normalize(tensor)

        # Pas de masking en inference, comme dans predict() -- mais mask_override
        # doit avoir une taille de batch N ici (et non 1).
        patch_size = self.config.get("patch_size", 8)
        num_patches = (64 // patch_size) ** 2  # 64 tokens
        mask_override = torch.zeros(n, num_patches, dtype=torch.bool, device=self.device)

        latent, _mask, _ids_restore = self.encoder(tensor, mask_override=mask_override)

        pooled = self._pool_cls_mean(latent)  # (N, 768) -- deja generique sur B

        albedo = self.head(pooled)  # (N, 1)

        return albedo.squeeze(1).cpu().numpy().tolist()  # list[float], N elements

    @staticmethod
    def _pool_cls_mean(tokens: torch.Tensor) -> torch.Tensor:
        """
        Pooling cls_mean : concatenation du token central (celui du milieu de la
        grille spatiale) avec la moyenne spatiale de tous les tokens.
        tokens : (B, N, D) avec N=64, D=384 -> sortie (B, 2*D) = (B, 768)
        """
        b, n, d = tokens.shape
        center_idx = n // 2
        center_token = tokens[:, center_idx, :]        # (B, D)
        mean_token = tokens.mean(dim=1)                 # (B, D)
        return torch.cat([center_token, mean_token], dim=1)  # (B, 2D)


# Cache simple pour eviter de recharger le meme checkpoint a chaque appel
# (utile depuis app.py qui appelle predict() a chaque interaction Streamlit).
# threading.Lock ajoute pour le traitement parallelise (lot d'adresses / scan
# de zone, cf. pipeline.py et zone_scan.py) -- sans lock, deux threads
# demarrant en meme temps sur un checkpoint pas encore en cache pourraient
# le charger chacun de leur cote (perte de temps/memoire, pas de corruption
# grave, mais autant l'eviter proprement).
import threading

_predictor_cache: dict[str, AlbedoPredictor] = {}
_predictor_cache_lock = threading.Lock()


def get_predictor(checkpoint_path: str, device: str = "cpu") -> AlbedoPredictor:
    if checkpoint_path in _predictor_cache:
        return _predictor_cache[checkpoint_path]
    with _predictor_cache_lock:
        # Double-check apres acquisition du lock : un autre thread a peut-etre
        # deja charge ce checkpoint pendant qu'on attendait.
        if checkpoint_path not in _predictor_cache:
            _predictor_cache[checkpoint_path] = AlbedoPredictor(checkpoint_path, device=device)
        return _predictor_cache[checkpoint_path]


def predict_albedo(patch: np.ndarray, checkpoint_path: str, device: str = "cpu") -> float:
    """Point d'entree simple a appeler depuis app.py."""
    predictor = get_predictor(checkpoint_path, device=device)
    return predictor.predict(patch)


def predict_albedo_batch(
    patches, checkpoint_path: str, device: str = "cpu"
) -> list[float]:
    """Point d'entree batch a appeler depuis zone_scan.py (scan de zone /
    lot d'adresses) -- evite un forward pass par patch.

    patches accepte un ndarray (N, 3, 64, 64) deja empile OU une liste de N
    ndarrays (3, 64, 64) (cf. AlbedoPredictor.predict_batch pour le detail :
    zone_scan.py passe une liste Python, pas un ndarray)."""
    predictor = get_predictor(checkpoint_path, device=device)
    return predictor.predict_batch(patches)