# Plan 1 — Détection (étapes 0 à 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un détecteur YOLO ré-entraîné sur VisDrone, évalué par notre propre mAP (vérifiée contre Ultralytics), avec un premier GIF de démo et l'expérience 1 chiffrée.

**Architecture:** Paquet Python `dronetrack` (layout `src/`). Toutes les briques échangent des `Detection` (types.py). Le détecteur est un mince wrapper autour d'Ultralytics ; conversion des données, statistiques et mAP sont faites maison et testées sur des cas calculables à la main.

**Tech Stack:** Python 3.11, uv, ultralytics (YOLO26), numpy, opencv-python, imageio, gdown, pytest. Entraînement sur Kaggle (GPU T4).

**Spec:** `docs/superpowers/specs/2026-10-25-drone-tracking-design.md` (étapes 0, 1, 2 des sections 6 et 10). Plans suivants : 2 = tuiles, 3 = suivi, 4 = vitrine.

## Global Constraints

- Python 3.11, dépendances gérées par `uv` ; tests avec `uv run pytest`.
- Boîtes : `x1, y1, x2, y2` en pixels, coin haut-gauche / bas-droit, partout dans le code.
- Classes YOLO (index 0–9) : `pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor` = catégorie VisDrone − 1. Catégories VisDrone 0 (« ignored regions ») et 11 (« others ») exclues.
- Modèle : `yolo26n.pt` (nano), `imgsz=640` pour le ré-entraînement.
- Données et poids hors Git : `data/`, `runs/`, `models/`, `outputs/` dans `.gitignore`.
- Dépôt GitHub public dès la tâche 1 (créé seulement après accord explicite d'Elie).
- Rythme Elie (mémoire du projet) : avant chaque tâche, 3–4 phrases simples (où on en est, ce que la tâche produit, pourquoi) et accord ; après, ce qui a changé, 1–2 questions de compréhension, ajout des nouveaux termes à `docs/GLOSSAIRE.md`.
- Tout téléchargement (VisDrone, poids) et toute installation globale (uv) : demander l'accord d'Elie en donnant nom, source et taille.

## Review Focus

1. **Boîtes VisDrone qui débordent de l'image ou de taille nulle** → recadrées dans l'image, ignorées si largeur ou hauteur ≤ 0 après recadrage (test tâche 3).
2. **Image sans aucun objet annoté** → fichier label vide écrit quand même, l'image reste dans le jeu (test tâche 3).
3. **Dossier de séquence contenant `.DS_Store` ou des noms non triés** → seuls `.jpg/.jpeg/.png` lus, dans l'ordre alphabétique (test tâche 2).
4. **Entrées vides** (aucune détection, aucune vérité terrain) → `iou_matrix` renvoie une matrice (0, M) ; la mAP compte AP = 0 pour une classe sans prédiction et ignore une classe sans vérité terrain (tests tâches 1 et 5).
5. **Classes du modèle COCO qui n'existent pas dans VisDrone** → supprimées par `class_map`, jamais renvoyées avec un mauvais index (test tâche 2).

---

### Task 1: Squelette du projet, types et IoU

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `README.md`, `src/dronetrack/__init__.py`, `src/dronetrack/types.py`, `src/dronetrack/geometry.py`, `docs/GLOSSAIRE.md`, `docs/DECISIONS.md`, `docs/RESULTATS.md`
- Test: `tests/test_types.py`, `tests/test_geometry.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class Detection: x1: float; y1: float; x2: float; y2: float; score: float; cls: int` avec propriétés `width`, `height`, `area`.
  - `@dataclass(frozen=True) class Track(Detection): track_id: int`.
  - `boxes_to_array(dets: Sequence[Detection]) -> np.ndarray` de forme (N, 4) float.
  - `iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray` : a (N,4), b (M,4) → (N,M).

- [ ] **Step 1:** Installer uv (`brew install uv`, après accord d'Elie), puis `uv init --lib --python 3.11` adapté au nom `dronetrack` ; dépendances : `ultralytics`, `numpy`, `opencv-python`, `imageio`, `gdown` ; dev : `pytest`. `.gitignore` : `data/ runs/ models/ outputs/ .venv/ __pycache__/ *.pt .DS_Store`.
- [ ] **Step 2: Tests qui échouent**

```python
def test_detection_size():
    d = Detection(10, 20, 40, 80, 0.9, 3)
    assert (d.width, d.height, d.area) == (30, 60, 1800)

def test_iou_identical_is_one():
    b = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(b, b)[0, 0] == pytest.approx(1.0)

def test_iou_disjoint_is_zero():
    assert iou_matrix(np.array([[0, 0, 10, 10]], float), np.array([[20, 20, 30, 30]], float))[0, 0] == 0.0

def test_iou_half_overlap():
    # intersection 50, union 150
    assert iou_matrix(np.array([[0, 0, 10, 10]], float), np.array([[5, 0, 15, 10]], float))[0, 0] == pytest.approx(1 / 3)

def test_iou_empty_shapes():
    assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)
```

- [ ] **Step 3:** `uv run pytest -v` → FAIL (import).
- [ ] **Step 4:** Implémenter `types.py` et `geometry.py` (IoU vectorisée numpy, sans boucle).
- [ ] **Step 5:** `uv run pytest -v` → PASS.
- [ ] **Step 6:** Docs squelettes : `GLOSSAIRE.md` (entrées : pixel, bounding box, IoU), `DECISIONS.md` (D1 : boîtes en x1y1x2y2 ; D2 : YOLO26n), `RESULTATS.md` (titres des 3 expériences), `README.md` (titre, mission en une phrase, « en construction »).
- [ ] **Step 7:** Commit `feat: project skeleton, Detection types and IoU`. Puis, **après accord d'Elie**, `gh repo create drone_tracking --public --source . --push`.

---

### Task 2: Étape 0 — YOLO pré-entraîné sur une vidéo VisDrone (premier GIF)

**Files:**
- Create: `src/dronetrack/io.py`, `src/dronetrack/detect/__init__.py`, `src/dronetrack/detect/yolo.py`, `src/dronetrack/viz/__init__.py`, `src/dronetrack/viz/draw.py`, `scripts/download_visdrone.py`, `scripts/run_detect.py`
- Test: `tests/test_io.py`, `tests/test_yolo.py`, `tests/test_draw.py`

**Interfaces:**
- Consumes: `Detection` (tâche 1).
- Produces:
  - `iter_frames(source: Path) -> Iterator[np.ndarray]` : dossier d'images (séquence MOT) ou fichier vidéo ; images BGR.
  - `COCO_TO_VISDRONE: dict[int, int] = {0: 0, 1: 2, 2: 3, 3: 9, 5: 8, 7: 5}` (person→pedestrian, bicycle, car, motorcycle→motor, bus, truck).
  - `to_detections(xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray, class_map: dict[int, int] | None) -> list[Detection]` (fonction pure).
  - `class YoloDetector(weights: str, conf: float = 0.25, imgsz: int = 640, class_map: dict[int, int] | None = None, device: str | None = None)` avec `detect(image: np.ndarray) -> list[Detection]`. `device=None` → `"mps"` si disponible, sinon `"cpu"`.
  - `draw_detections(image: np.ndarray, dets: Sequence[Detection], class_names: Sequence[str]) -> np.ndarray` (renvoie une copie ; couleur fixe par classe).
  - `scripts/download_visdrone.py {det-train,det-val,mot-val} --out data/` : DET depuis `https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{train,val}.zip` (1,55 Go / 82 Mo) ; MOT-val via `gdown` depuis Google Drive id `1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu` (1,48 Go). Dézippe, saute si déjà présent.
  - `scripts/run_detect.py --source PATH --weights yolo26n.pt [--coco] --out outputs/x.mp4 --gif outputs/x.gif --max-frames 150`.

- [ ] **Step 1: Tests qui échouent**

```python
def test_iter_frames_folder_sorted_and_filtered(tmp_path):
    for name, v in [("0000002.jpg", 2), ("0000001.jpg", 1)]:
        cv2.imwrite(str(tmp_path / name), np.full((4, 4, 3), v, np.uint8))
    (tmp_path / ".DS_Store").write_text("x")
    frames = list(iter_frames(tmp_path))
    assert [f[0, 0, 0] for f in frames] == [1, 2]

def test_to_detections_applies_class_map_and_drops_unknown():
    dets = to_detections(np.array([[0, 0, 5, 5], [1, 1, 6, 6]], float), np.array([0.9, 0.8]),
                         np.array([2, 14]), COCO_TO_VISDRONE)  # 2=car, 14=bird
    assert len(dets) == 1 and dets[0].cls == 3

def test_to_detections_without_map_keeps_classes():
    dets = to_detections(np.array([[0, 0, 5, 5]], float), np.array([0.5]), np.array([7]), None)
    assert dets[0].cls == 7

def test_draw_returns_copy_and_marks_pixels():
    img = np.zeros((50, 50, 3), np.uint8)
    out = draw_detections(img, [Detection(10, 10, 30, 30, 0.9, 0)], ["pedestrian"])
    assert img.sum() == 0 and out.sum() > 0

@pytest.mark.slow
def test_yolo_detector_runs_on_blank_image():
    assert YoloDetector("yolo26n.pt").detect(np.zeros((640, 640, 3), np.uint8)) == []
```

Déclarer le marqueur `slow` dans `pyproject.toml` ; par défaut `uv run pytest -m "not slow"`.

- [ ] **Step 2:** `uv run pytest -v` → FAIL.
- [ ] **Step 3:** Implémenter `io.py`, `detect/yolo.py`, `viz/draw.py`.
- [ ] **Step 4:** `uv run pytest -v` (slow inclus une fois, il télécharge `yolo26n.pt`, ~5 Mo) → PASS.
- [ ] **Step 5:** Implémenter les deux scripts. Après accord d'Elie, `uv run scripts/download_visdrone.py mot-val`.
- [ ] **Step 6:** `uv run scripts/run_detect.py --source data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v --weights yolo26n.pt --coco --gif outputs/etape0_coco.gif --max-frames 150` → le GIF existe ; le montrer à Elie et noter ce qu'on voit (piétons manqués, faux positifs).
- [ ] **Step 7:** GLOSSAIRE (COCO, pré-entraîné, seuil de confiance), DECISIONS (D3 : correspondance COCO→VisDrone). Commit `feat: step 0 pretrained YOLO demo on VisDrone video`.

---

### Task 3: Étape 1 — Lecture VisDrone et conversion au format YOLO

**Files:**
- Create: `src/dronetrack/data/__init__.py`, `src/dronetrack/data/visdrone.py`, `src/dronetrack/data/convert.py`, `scripts/convert_visdrone.py`
- Test: `tests/test_visdrone.py`, `tests/test_convert.py`

**Interfaces:**
- Consumes: `Detection`.
- Produces:
  - `CLASS_NAMES: tuple[str, ...]` (les 10 noms, ordre des Global Constraints).
  - `parse_det_annotation(path: Path, img_w: int, img_h: int) -> list[Detection]` : ligne `left,top,w,h,score,category,truncation,occlusion` ; garde `score == 1` et `1 <= category <= 10` ; `cls = category - 1` ; recadre dans l'image ; ignore les boîtes de largeur ou hauteur ≤ 0 ; `score` du `Detection` = 1.0.
  - `to_yolo_line(det: Detection, img_w: int, img_h: int) -> str` : `"cls cx cy w h"` normalisés, 6 décimales.
  - `convert_split(src: Path, dst: Path) -> int` : `src/{images,annotations}` → `dst/images` (liens symboliques) + `dst/labels/*.txt` ; renvoie le nombre d'images.
  - `write_data_yaml(root: Path, out: Path) -> None` : `path`, `train: train/images`, `val: val/images`, `names` = CLASS_NAMES.
  - `scripts/convert_visdrone.py --raw data --out data/visdrone_yolo` (train et val).

- [ ] **Step 1: Tests qui échouent** (petits fichiers d'annotation écrits dans `tmp_path`)

```python
def test_parse_keeps_valid_and_shifts_class(tmp_path):
    p = tmp_path / "a.txt"; p.write_text("10,20,30,40,1,4,0,0\n")
    assert parse_det_annotation(p, 100, 100) == [Detection(10, 20, 40, 60, 1.0, 3)]

def test_parse_drops_ignored_others_and_score0(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0,0,5,5,0,0,0,0\n0,0,5,5,1,11,0,0\n0,0,5,5,0,4,0,0\n")
    assert parse_det_annotation(p, 100, 100) == []

def test_parse_clips_and_drops_degenerate(tmp_path):
    p = tmp_path / "a.txt"; p.write_text("90,90,20,20,1,1,0,0\n50,50,0,10,1,1,0,0\n")
    assert parse_det_annotation(p, 100, 100) == [Detection(90, 90, 100, 100, 1.0, 0)]

def test_to_yolo_line():
    assert to_yolo_line(Detection(0, 0, 50, 100, 1.0, 3), 100, 200) == "3 0.250000 0.250000 0.500000 0.500000"

def test_convert_split_writes_empty_label_for_empty_image(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "images").mkdir(parents=True); (src / "annotations").mkdir()
    cv2.imwrite(str(src / "images" / "x.jpg"), np.zeros((100, 100, 3), np.uint8))
    (src / "annotations" / "x.txt").write_text("")
    assert convert_split(src, dst) == 1
    assert (dst / "labels" / "x.txt").read_text() == ""
    assert (dst / "images" / "x.jpg").exists()
```

- [ ] **Step 2:** `uv run pytest -v` → FAIL.
- [ ] **Step 3:** Implémenter `visdrone.py`, `convert.py`, le script.
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** Après accord d'Elie, `uv run scripts/download_visdrone.py det-train det-val` puis `uv run scripts/convert_visdrone.py --raw data --out data/visdrone_yolo` → affiche 6471 images train et 548 val.
- [ ] **Step 6:** GLOSSAIRE (annotation, format YOLO, coordonnées normalisées, ignored regions). Commit `feat: VisDrone to YOLO conversion`.

---

### Task 4: Étape 1 (suite) — Statistiques et vérification visuelle

**Files:**
- Create: `src/dronetrack/data/stats.py`, `scripts/dataset_stats.py`, `scripts/show_annotations.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `parse_det_annotation`, `CLASS_NAMES`, `draw_detections`.
- Produces:
  - `size_bucket(det: Detection) -> str` : `"small"` si aire < 32², `"medium"` si < 96², sinon `"large"` (seuils COCO).
  - `dataset_stats(annotations: Iterable[list[Detection]]) -> dict` avec clés `per_class: dict[int, int]`, `per_size: dict[str, int]`, `objects_per_image_mean: float`.
  - `scripts/dataset_stats.py --raw data/VisDrone2019-DET-train` imprime un petit tableau lisible.
  - `scripts/show_annotations.py --raw ... --image NOM --out outputs/verif.jpg` dessine la vérité terrain.

- [ ] **Step 1: Tests qui échouent**

```python
def test_size_buckets_coco_thresholds():
    assert size_bucket(Detection(0, 0, 31, 31, 1, 0)) == "small"
    assert size_bucket(Detection(0, 0, 32, 32, 1, 0)) == "medium"
    assert size_bucket(Detection(0, 0, 96, 96, 1, 0)) == "large"

def test_dataset_stats_counts():
    s = dataset_stats([[Detection(0, 0, 10, 10, 1, 3), Detection(0, 0, 100, 100, 1, 3)], []])
    assert s["per_class"] == {3: 2} and s["per_size"]["small"] == 1 and s["objects_per_image_mean"] == 1.0
```

- [ ] **Step 2:** `uv run pytest tests/test_stats.py -v` → FAIL.
- [ ] **Step 3:** Implémenter `stats.py` et les deux scripts.
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** Lancer les deux scripts sur le train ; montrer à Elie le tableau et `outputs/verif.jpg` ; Elie confirme que les boîtes tombent sur les objets. Reporter dans `RESULTATS.md` (section « Données ») la part d'objets « small ».
- [ ] **Step 6:** GLOSSAIRE (petits / moyens / grands objets, déséquilibre de classes). Commit `feat: dataset statistics and visual check`.

---

### Task 5: mAP faite maison

**Files:**
- Create: `src/dronetrack/eval/__init__.py`, `src/dronetrack/eval/det_metrics.py`
- Test: `tests/test_det_metrics.py`

**Interfaces:**
- Consumes: `Detection`, `boxes_to_array`, `iou_matrix`.
- Produces:
  - `average_precision(recall: np.ndarray, precision: np.ndarray) -> float` : enveloppe de précision décroissante puis interpolation sur 101 points de rappel `np.linspace(0, 1, 101)` et intégration trapèzes (même méthode qu'Ultralytics).
  - `@dataclass class DetMetrics: map50: float; map50_95: float; ap50_per_class: dict[int, float]`.
  - `evaluate_detections(preds: dict[str, list[Detection]], gts: dict[str, list[Detection]], classes: Sequence[int], iou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10)) -> DetMetrics`. Clés = noms d'images. Par classe et par seuil : prédictions triées par score décroissant (toutes images confondues), chacune associée à la vérité terrain non encore prise de plus grand IoU dans la même image si IoU ≥ seuil (sinon faux positif). Moyenne sur les classes de `classes` ayant au moins une vérité terrain.

- [ ] **Step 1: Tests qui échouent**

```python
G = {"im": [Detection(0, 0, 10, 10, 1, 0)]}

def test_perfect_predictions_give_one():
    m = evaluate_detections({"im": [Detection(0, 0, 10, 10, 0.9, 0)]}, G, [0])
    assert m.map50 == pytest.approx(1.0) and m.map50_95 == pytest.approx(1.0)

def test_no_prediction_gives_zero():
    assert evaluate_detections({"im": []}, G, [0]).map50 == 0.0

def test_false_positive_ranked_below_true_positive_is_free():
    p = {"im": [Detection(0, 0, 10, 10, 0.9, 0), Detection(50, 50, 60, 60, 0.5, 0)]}
    assert evaluate_detections(p, G, [0]).map50 == pytest.approx(1.0)

def test_false_positive_ranked_above_halves_ap():
    p = {"im": [Detection(0, 0, 10, 10, 0.5, 0), Detection(50, 50, 60, 60, 0.9, 0)]}
    assert evaluate_detections(p, G, [0]).map50 == pytest.approx(0.5, abs=0.01)

def test_duplicate_detection_counts_as_false_positive():
    p = {"im": [Detection(0, 0, 10, 10, 0.9, 0), Detection(0, 0, 10, 10, 0.8, 0)]}
    assert evaluate_detections(p, G, [0]).map50 == pytest.approx(1.0)  # FP après le TP

def test_class_without_ground_truth_is_ignored():
    m = evaluate_detections({"im": [Detection(0, 0, 10, 10, 0.9, 0)]}, G, [0, 5])
    assert m.map50 == pytest.approx(1.0) and 5 not in m.ap50_per_class
```

- [ ] **Step 2:** `uv run pytest tests/test_det_metrics.py -v` → FAIL.
- [ ] **Step 3:** Implémenter `det_metrics.py`.
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** GLOSSAIRE (précision, rappel, courbe précision-rappel, AP, mAP@50, mAP@50-95). Commit `feat: homemade mAP`.

---

### Task 6: Étape 2 — Ré-entraînement sur Kaggle

**Files:**
- Create: `scripts/train.py`, `notebooks/kaggle_train.ipynb`, `docs/KAGGLE.md`

**Interfaces:**
- Consumes: `scripts/download_visdrone.py`, `scripts/convert_visdrone.py`.
- Produces:
  - `scripts/train.py --data data/visdrone_yolo/data.yaml --model yolo26n.pt --epochs 50 --imgsz 640 --batch 16 --project runs --name visdrone_y26n [--fraction 1.0] [--resume runs/.../last.pt]` (`--fraction` = part du jeu d'entraînement utilisée, pour les essais rapides) : appelle `YOLO(model).train(..., save_period=5)` ; avec `--resume`, `YOLO(path).train(resume=True)`.
  - Poids finaux attendus en local : `models/visdrone_y26n.pt` (copie de `best.pt`).

- [ ] **Step 1:** Écrire `train.py` ; vérification locale rapide sur Mac : `uv run scripts/train.py --data data/visdrone_yolo/data.yaml --epochs 1 --imgsz 320 --batch 4 --fraction 0.02` (ajouter l'option `--fraction`) → se termine sans erreur et crée `runs/.../weights/last.pt`.
- [ ] **Step 2:** Écrire le notebook (cellules : `git clone` du dépôt public, `pip install -e .`, téléchargement det-train + det-val, conversion, `train.py`, copie de `best.pt` dans `/kaggle/working/`) et `docs/KAGGLE.md` : ce qu'Elie fait à la main (compte Kaggle vérifié par téléphone, Accelerator « GPU T4 », Internet activé, « Save Version → Run All », reprise avec `--resume` si la session coupe, téléchargement de `best.pt`).
- [ ] **Step 3:** Commit `feat: Kaggle training script and notebook` ; push.
- [ ] **Step 4 (Elie, sur Kaggle) :** lancer l'entraînement ; Claude explique pendant ce temps les courbes de perte affichées. Fini quand `best.pt` est placé dans `models/visdrone_y26n.pt`.
- [ ] **Step 5:** GLOSSAIRE (epoch, batch, fine-tuning, perte, sur-apprentissage) ; DECISIONS (D4 : 50 epochs, imgsz 640, batch 16, et pourquoi). Commit.

---

### Task 7: Expérience 1 — avant / après, et notre mAP contre celle d'Ultralytics

**Files:**
- Create: `scripts/evaluate_detector.py`
- Modify: `docs/RESULTATS.md`, `docs/DECISIONS.md`, `README.md`

**Interfaces:**
- Consumes: `YoloDetector`, `COCO_TO_VISDRONE`, `parse_det_annotation`, `evaluate_detections`, `CLASS_NAMES`.
- Produces: `scripts/evaluate_detector.py --weights PATH [--coco] --raw data/VisDrone2019-DET-val [--classes 0 2 3 5 8 9] [--max-images N] [--compare-ultralytics data/visdrone_yolo/data.yaml]` : détecte avec `conf=0.001`, calcule notre mAP, et avec `--compare-ultralytics` lance `YOLO(weights).val(data=..., imgsz=640, conf=0.001)` et affiche les deux chiffres côte à côte.

- [ ] **Step 1:** Implémenter le script ; le lancer sur `--max-images 20` pour vérifier qu'il tourne.
- [ ] **Step 2: Vérification de notre mAP.** `--weights models/visdrone_y26n.pt --compare-ultralytics ...` sur les 10 classes → écart attendu ≤ 0,01 en mAP@50. Au-delà : chercher la cause (letterbox, appariement) et la noter dans DECISIONS.md avant de continuer.
- [ ] **Step 3: Expérience 1.** Sur les 6 classes communes (`0 2 3 5 8 9`) : `yolo26n.pt --coco` contre `models/visdrone_y26n.pt`. Tableau mAP@50 et mAP@50-95, par classe, dans `RESULTATS.md` avec 2–3 phrases d'interprétation (en particulier sur les piétons).
- [ ] **Step 4:** Regénérer le GIF de l'étape 0 avec le modèle ré-entraîné (`outputs/etape2_finetuned.gif`) ; README : les deux GIF côte à côte + tableau de l'expérience 1.
- [ ] **Step 5:** Commit `feat: experiment 1 - fine-tuning gain` ; push. Proposer à Elie la ligne de CV correspondante (uniquement ce qui est fait).
