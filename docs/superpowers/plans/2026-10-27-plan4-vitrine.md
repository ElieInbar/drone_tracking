# Plan 4 — Vitrine : démo en ligne (étape 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Une démo publique sur Hugging Face Spaces : un recruteur dépose une vidéo de drone (ou clique sur l'exemple) et récupère, en moins d'une minute, la vidéo annotée avec les boîtes et les numéros de suivi, plus un petit résumé. Le lien de la démo va en haut du README et du CV.

**Architecture:** Toute la logique reste dans le paquet `dronetrack`, dans une fonction testée : `demo.track_video` (lire la vidéo → détecter → suivre → dessiner → écrire un MP4 lisible par un navigateur). Le dossier `space/` contient uniquement ce que le Space Hugging Face exécute : une interface Gradio fine (`app.py`), les dépendances, et une vidéo d'exemple. Le Space installe `dronetrack` depuis le dépôt GitHub, figé sur le tag `v1.0`. Il télécharge les poids du modèle depuis la release GitHub `v1.0`, ce qui laisse une seule source de vérité pour le code et pour les poids.

**Tech Stack:** Python 3.11, Gradio 5 (groupe de dépendances `demo`, absent des dépendances principales), imageio + imageio-ffmpeg (encodage H.264), le détecteur `models/visdrone_y26n.pt` en image entière, notre `ByteTracker`. Hébergement : Hugging Face Spaces avec le matériel gratuit « CPU basic » (2 vCPU, 16 Go).

**Spec:** `docs/superpowers/specs/2026-10-25-drone-tracking-design.md` (section 6, étape 5 : « README final, GIF, démo Hugging Face Space (Gradio) — Un inconnu comprend le projet en 30 secondes »).

## Global Constraints

- Les contraintes globales des plans 1 à 3 restent valables : uv, boîtes x1y1x2y2, classes 0–9. Le rythme avec Elie aussi : explication avant chaque tâche et accord, questions après, glossaire.
- Réglages de la démo, identiques à l'expérience 3 : détecteur en image entière, `TRACK_CONF = 0.1`, `imgsz = 640`, tracker `ByteTracker()` avec ses réglages par défaut. Pas de tuiles : elles sont 10 fois plus lentes et dégradent le suivi (MOTA −0,18, D8).
- Limite pour tenir sur le CPU gratuit : `MAX_FRAMES = 300` images traitées au maximum, et images réduites à `MAX_WIDTH = 1280` px de large avant la détection et l'écriture.
- Vidéo de sortie : MP4 en H.264, format de pixels `yuv420p`, largeur et hauteur paires. C'est la seule combinaison que lisent tous les navigateurs ; le `mp4v` d'OpenCV ne s'affiche pas dans Gradio.
- URL des poids : `https://github.com/ElieInbar/drone_tracking/releases/download/v1.0/visdrone_y26n.pt`.
- **Actions réservées à Elie, ou soumises à son accord explicite :**
  - créer le compte et le Space Hugging Face, puis se connecter (`hf auth login`) : Elie seul ;
  - pousser le tag `v1.0` et créer la release GitHub avec les poids : publication publique, accord explicite ;
  - pousser le dossier `space/` vers le Space : publication publique, accord explicite.
- Vidéo d'exemple : 10 secondes au plus, tirées d'une séquence VisDrone-MOT val. Le crédit VisDrone (lien et usage académique) est indiqué dans `space/README.md`. Elie valide la diffusion de cet extrait avant la mise en ligne.

## Review Focus

1. **Fichier qui n'est pas une vidéo, ou vidéo illisible** : message clair « Could not read any frame from this video », pas de trace d'erreur Python (test, tâche 1).
2. **Vidéo très longue ou en 4K** : seules les 300 premières images sont traitées, en 1280 px de large au plus, et le résumé le dit (tests, tâche 1).
3. **Dimensions impaires** (par exemple 1279 × 719 après réduction) : la sortie a des dimensions paires, sinon H.264 refuse d'encoder (test, tâche 1).
4. **Premier démarrage du Space sans poids** : les poids sont téléchargés une seule fois puis réutilisés ; un téléchargement interrompu ne laisse pas de fichier à moitié écrit (tests, tâche 2).
5. **Vidéo sans aucun objet** (ciel, mer) : sortie écrite normalement, résumé « 0 objects tracked » (test, tâche 1).

---

### Task 1: `track_video`, le cœur de la démo, testé hors Gradio

**Files:**
- Create: `src/dronetrack/demo.py`
- Modify: `pyproject.toml` (ajouter `imageio-ffmpeg` aux dépendances principales)
- Test: `tests/test_demo.py`

**Interfaces:**
- Consumes :
  - `iter_frames(source)` (`dronetrack.io`) ;
  - `draw_tracks(image, tracks, class_names)` et `CLASS_NAMES` ;
  - `ByteTracker().update(dets) -> list[Track]` ;
  - n'importe quel objet qui a `.detect(image) -> list[Detection]`.
- Produces :
  - `MAX_FRAMES = 300` et `MAX_WIDTH = 1280` ;
  - `@dataclass DemoResult` avec les champs `frames: int`, `truncated: bool` (la vidéo avait plus de `max_frames` images), `objects_per_class: dict[str, int]` (identités distinctes par nom de classe, classes vides absentes, triées par nombre décroissant), `seconds: float` ;
  - `DemoResult.summary() -> str` : texte en anglais pour l'interface. Exemple : `"300 frames (first 300 only) in 41 s · 57 objects tracked: car 38, pedestrian 12, van 7"`. S'il n'y a aucun objet : `"... · 0 objects tracked"` ;
  - `track_video(source: Path, out: Path, detector, tracker, max_frames: int = MAX_FRAMES, max_width: int = MAX_WIDTH) -> DemoResult`. Elle lève `ValueError("Could not read any frame from this video")` si aucune image n'est lue (`iter_frames` peut aussi renvoyer `None` pour une image illisible) ;
  - `resize_for_demo(frame: np.ndarray, max_width: int) -> np.ndarray` : réduit si `w > max_width` (proportions gardées), puis rogne d'un pixel pour rendre largeur et hauteur paires.

- [ ] **Step 1: Tests qui échouent.** Ils utilisent un faux détecteur (pas de modèle à charger) et une petite vidéo de test écrite dans `tmp_path` avec `cv2.VideoWriter` en `mp4v`.

```python
class FakeDetector:  # one car moving 4 px right per frame
    def __init__(self): self.n = 0
    def detect(self, image):
        self.n += 1
        return [Detection(10 + 4 * self.n, 20, 40 + 4 * self.n, 40, 0.9, 3)]

def write_video(path, n, w=320, h=240): ...  # n black frames

def test_tracks_one_object_and_writes_browser_mp4(tmp_path):
    src, out = write_video(tmp_path / "in.mp4", 20), tmp_path / "out.mp4"
    r = track_video(src, out, FakeDetector(), ByteTracker())
    assert r.frames == 20 and not r.truncated and r.objects_per_class == {"car": 1}
    reader = imageio.get_reader(out)
    assert reader.get_meta_data()["codec"] == "h264" and reader.count_frames() == 20

def test_stops_at_max_frames(tmp_path):
    r = track_video(write_video(tmp_path / "in.mp4", 30), tmp_path / "o.mp4", FakeDetector(), ByteTracker(), max_frames=10)
    assert r.frames == 10 and r.truncated and "first 10 only" in r.summary()

def test_resize_for_demo_caps_width_and_makes_sizes_even():
    assert resize_for_demo(np.zeros((1080, 1920, 3), np.uint8), 1280).shape == (720, 1280, 3)
    assert resize_for_demo(np.zeros((241, 321, 3), np.uint8), 1280).shape == (240, 320, 3)

def test_no_object_still_writes_video(tmp_path):
    class Empty:
        def detect(self, image): return []
    r = track_video(write_video(tmp_path / "in.mp4", 5), tmp_path / "o.mp4", Empty(), ByteTracker())
    assert r.objects_per_class == {} and "0 objects tracked" in r.summary() and (tmp_path / "o.mp4").exists()

def test_unreadable_file_raises_clear_error(tmp_path):
    bad = tmp_path / "notes.mp4"; bad.write_text("not a video")
    with pytest.raises(ValueError, match="Could not read any frame"):
        track_video(bad, tmp_path / "o.mp4", FakeDetector(), ByteTracker())
```

- [ ] **Step 2:** `uv add imageio-ffmpeg`, puis `uv run pytest tests/test_demo.py -v` → FAIL (import).
- [ ] **Step 3:** Implémenter `demo.py`. Le writer est `imageio.v2.get_writer(out, fps=fps, codec="libx264", pixelformat="yuv420p", macro_block_size=1)`. Il reçoit des images RVB : convertir depuis le BGR d'OpenCV. Lire `fps` dans la vidéo source avec `cv2.CAP_PROP_FPS`, avec 10 par défaut quand la source est un dossier ou que la valeur vaut 0. Compter les identités dans un `dict[track_id, cls]` puis grouper par `CLASS_NAMES[cls]`.
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5: Essai réel.** `scripts/make_example_clip.py --sequence data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v --out space/examples/visdrone_086.mp4 --frames 100 --fps 10` écrit un clip H.264 de 10 s, au plus 1280 px de large, d'environ 5 Mo au maximum. Un petit script dans le scratchpad lance ensuite `track_video` sur ce clip avec le vrai `YoloDetector("models/visdrone_y26n.pt", conf=TRACK_CONF, device="cpu")`. Noter le temps CPU et le résumé : ils servent à vérifier que 300 images tiennent en environ 1 minute sur 2 vCPU. Sinon, baisser `MAX_FRAMES` et le noter dans D9.
- [ ] **Step 6:** GLOSSAIRE (H.264 / codec, CPU contre GPU pour l'inférence). Commit `feat: demo pipeline (track_video) writing browser-playable MP4` (ajouter le clip d'exemple et son script).

---

### Task 2: Interface Gradio et fichiers du Space

**Files:**
- Create: `space/app.py`, `space/requirements.txt`, `space/packages.txt`, `space/README.md`
- Modify: `pyproject.toml` (groupe `demo = ["gradio>=5,<6"]`)
- Test: `tests/test_space_app.py`

**Interfaces:**
- Consumes : `track_video`, `DemoResult.summary()`, `MAX_FRAMES` (tâche 1) ; `YoloDetector`, `ByteTracker`, `TRACK_CONF`.
- Produces, dans `space/app.py` :
  - `WEIGHTS_URL` (constante globale), `WEIGHTS_PATH = Path("visdrone_y26n.pt")` ;
  - `ensure_weights(path: Path = WEIGHTS_PATH, url: str = WEIGHTS_URL) -> Path` : ne télécharge que si le fichier manque. Le téléchargement passe par `urllib.request.urlretrieve` vers `path.with_suffix(".part")`, suivi d'un `rename` : jamais de fichier à moitié écrit sous le vrai nom ;
  - `run(video_path: str | None) -> tuple[str, str]` : renvoie (chemin du MP4 annoté, résumé). Pas de vidéo → `gr.Error("Please upload a video or pick the example.")`. `ValueError` de `track_video` → `gr.Error(str(e))`. Le détecteur est chargé une seule fois au niveau du module, en lazy, sur `device="cpu"` ;
  - `build_ui() -> gr.Blocks`. En haut, un titre et 2 lignes de texte : ce que fait la démo, la limite de `MAX_FRAMES` images, le lien vers le dépôt GitHub. Ensuite : `gr.Video` en entrée, un bouton « Track », `gr.Video` et `gr.Textbox` en sortie, `gr.Examples` avec `examples/visdrone_086.mp4`, et un tableau des résultats clés (mAP 0,14 → 0,34 ; MOTA 0,193 contre 0,177 pour ByteTrack). En bas de `app.py` : `if __name__ == "__main__": build_ui().launch()`.
- `space/requirements.txt` :
  - `--extra-index-url https://download.pytorch.org/whl/cpu` : torch CPU, environ 200 Mo au lieu de 2 Go en CUDA ;
  - `torch`, `torchvision` ;
  - `dronetrack @ git+https://github.com/ElieInbar/drone_tracking@v1.0` ;
  - `gradio>=5,<6`.
- `space/packages.txt` : `libgl1` et `libglib2.0-0`, dont OpenCV a besoin sur l'image Linux du Space.
- `space/README.md` :
  - en-tête YAML du Space : `title: Drone Tracking`, `sdk: gradio`, `app_file: app.py`, `python_version: "3.11"`, `license: agpl-3.0` (celle d'Ultralytics) ;
  - un paragraphe de description ;
  - le crédit VisDrone pour le clip d'exemple.

- [ ] **Step 1: Tests qui échouent.** `pytest.importorskip("gradio")`. Le module est importé depuis `space/app.py` avec `importlib.util.spec_from_file_location`.

```python
def test_ensure_weights_downloads_once(tmp_path):
    src = tmp_path / "src.pt"; src.write_bytes(b"w")
    dst = tmp_path / "w.pt"
    assert app.ensure_weights(dst, src.as_uri()).read_bytes() == b"w"
    src.write_bytes(b"changed")
    assert app.ensure_weights(dst, src.as_uri()).read_bytes() == b"w"  # not downloaded again

def test_failed_download_leaves_no_weights_file(tmp_path):
    with pytest.raises(Exception):
        app.ensure_weights(tmp_path / "w.pt", (tmp_path / "missing.pt").as_uri())
    assert not (tmp_path / "w.pt").exists()

def test_run_without_video_shows_message():
    with pytest.raises(gr.Error, match="upload a video"):
        app.run(None)

def test_ui_builds():
    assert isinstance(app.build_ui(), gr.Blocks)
```

- [ ] **Step 2:** `uv add --group demo "gradio>=5,<6"`, puis `uv run --group demo pytest tests/test_space_app.py -v` → FAIL.
- [ ] **Step 3:** Écrire les 4 fichiers de `space/`. Pour les essais locaux, `WEIGHTS_PATH` peut être remplacé par la variable d'environnement `DRONETRACK_WEIGHTS`, ce qui permet d'utiliser `models/visdrone_y26n.pt` tant que la release n'existe pas.
- [ ] **Step 4:** `uv run --group demo pytest -v` → PASS (sans `--group demo`, les tests du Space sont ignorés).
- [ ] **Step 5: Essai dans le navigateur.** Lancer `DRONETRACK_WEIGHTS=models/visdrone_y26n.pt uv run --group demo python space/app.py` via `preview_start`, puis cliquer sur l'exemple et sur « Track ». La vidéo annotée doit s'afficher et se lire dans la page, et le résumé doit apparaître. Vérifier aussi à 375 px de large (mobile).
- [ ] **Step 6:** Commit `feat: Gradio app and Hugging Face Space files`.

---

### Task 3: Mise en ligne (release des poids et Space)

**Files:**
- Modify: `docs/DECISIONS.md` (D9)

**Interfaces:**
- Consumes : `space/` (tâche 2), `models/visdrone_y26n.pt`.
- Produces : la release GitHub `v1.0` avec l'asset `visdrone_y26n.pt` à l'URL `WEIGHTS_URL`, et le Space public `https://huggingface.co/spaces/<compte-HF-d-Elie>/drone-tracking`.

- [ ] **Step 1:** D9 dans DECISIONS : CPU gratuit plutôt que GPU payant, limite de 300 images, pas de tuiles, H.264, code et poids pris au tag `v1.0`. Commit.
- [ ] **Step 2 (accord d'Elie) :** `git tag v1.0 && git push origin main v1.0`, puis `gh release create v1.0 models/visdrone_y26n.pt --title "v1.0 — detection + tracking" --notes "<résultats des expériences 1 à 3, 4 lignes>"`. Vérifier : `curl -sIL <WEIGHTS_URL>` renvoie 200 et une taille de 5 443 269 octets.
- [ ] **Step 3 (Elie) :** créer un compte Hugging Face si besoin, puis le Space `drone-tracking` (SDK Gradio, matériel CPU basic gratuit), et lancer `hf auth login` dans le terminal.
- [ ] **Step 4 (accord d'Elie) :** cloner le Space dans le scratchpad, y copier le contenu de `space/` (le clip d'exemple fait moins de 10 Mo, donc pas besoin de Git LFS), puis faire un commit et un push.
- [ ] **Step 5: Vérifier en ligne.** Suivre le build dans les logs du Space. Puis, dans le navigateur intégré, ouvrir le Space et lancer l'exemple : la vidéo annotée doit se lire. Noter le temps réel dans D9. Si le build échoue, corriger dans `space/` et re-pousser (le tag `v1.0` ne bouge pas tant que le code du paquet ne change pas).

---

### Task 4: README et CV pointent vers la démo

**Files:**
- Modify: `README.md`, `docs/RESULTATS.md` (une ligne sur la démo et son temps CPU), CV `/Users/elieinbar/Documents/projets/cv/Elie_Inbar_CV_EN.html` et `.pdf` (hors du dépôt)

- [ ] **Step 1:** En tête du README, juste sous la première phrase : un badge « Open in Spaces » avec une ligne « **[Live demo](<URL du Space>)** — upload a drone video, get it back with tracked IDs ». Dans la feuille de route, cocher l'étape 5. Dans « Run it », télécharger les poids depuis la release (`curl -L -o models/visdrone_y26n.pt <WEIGHTS_URL>`) et ajouter la commande de lancement local de la démo.
- [ ] **Step 2:** Commit `docs: live demo link`, puis push (accord d'Elie).
- [ ] **Step 3:** CV : ajouter « Live demo » à côté du lien du dépôt sur la ligne de titre du projet drone. Régénérer le PDF avec Chrome headless et vérifier qu'il tient sur une page (capture d'écran).
- [ ] **Step 4:** Ouvrir les liens du README et du PDF un par un et vérifier qu'ils fonctionnent.
