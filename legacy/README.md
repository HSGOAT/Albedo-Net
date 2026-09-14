# legacy/

## app.py (Streamlit) — archivé le 14/09/2026

Décision produit : **FastAPI (`main.py`) + `static/index.html` est le
frontend officiel d'AlbedoNet.** L'app Streamlit qui vivait auparavant à
la racine du dépôt (`app.py`) est déplacée ici à titre d'archive, pas
supprimée — elle reste consultable et exécutable si besoin (`streamlit run
legacy/app.py` depuis la racine du dépôt, ses imports vers les modules
partagés — `inference`, `materials`, `pipeline`, `zone_scan`, etc. —
restent valides tant que ces modules ne bougent pas).

**Ce fichier ne sera plus maintenu.** Toute nouvelle feature (affichage du
score de confiance, filtre par matériau, etc.) doit désormais être
ajoutée uniquement côté `static/index.html` + `main.py`. C'était la
raison de trancher : avant cet archivage, chaque feature UI coûtait
double (à écrire une fois pour Streamlit, une fois pour le HTML/JS).

Pourquoi FastAPI plutôt que Streamlit :
- API REST découplée du rendu (`/api/single`, `/api/batch`,
  `/api/batch.csv`, `/api/zone`, `/api/zone.csv`, `/api/zone/stream`) —
  réutilisable par d'autres clients que le frontend web (scripts, futurs
  clients mobiles, intégrations tierces) sans dépendre du runtime
  Streamlit.
- `/api/zone/stream` (Server-Sent Events) donne un retour progressif sur
  un scan de zone, plus adapté à ce cas d'usage qu'un rerun complet de
  script Streamlit à chaque interaction.
- Le frontend HTML/JS (`static/index.html`) n'a pas les contraintes de
  layout de Streamlit (rerun de script à chaque interaction, pas de vrai
  contrôle fin du DOM) pour la carte et les vignettes de bâtiments.

Si le contenu de `app.py` doit être consulté pour son historique
(comportements, décisions passées), l'historique git complet reste
disponible via `git log -- app.py` avant le commit qui l'a déplacé ici.
