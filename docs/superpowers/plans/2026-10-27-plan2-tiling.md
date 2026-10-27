# Plan 2 — Tuiles pour petits objets (étape 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un détecteur par tuiles fait maison (inspiré de SAHI) et l'expérience 2 chiffrée : gain de mAP sur les petits objets contre coût en images par seconde.

**Architecture:** Le découpage, la fusion des doublons et le détecteur par tuiles sont trois briques séparées dans `detect/` et `geometry.py`. `TiledDetector` enveloppe n'importe quel objet qui a `detect_batch` (donc `YoloDetector`), et renvoie des `Detection` comme lui : le reste du pipeline ne voit pas la différence. La mAP gagne un filtre par taille d'objet (méthode COCO) pour mesurer le gain sur les petits.

**Tech Stack:** Python 3.11, numpy, ultralytics (YOLO26n ré-entraîné : `models/visdrone_y26n.pt`), pytest. Tout tourne sur le Mac M1 (pas de GPU cloud).

**Spec:** `docs/superpowers/specs/2026-10-25-drone-tracking-design.md` (section 3 « tuiles », section 6 étape 3, section 10 expérience 2). Point ouvert repris du plan 1 : `docs/DECISIONS.md` D6.

## Global Constraints

- Toutes les contraintes globales du plan 1 restent valables (Python 3.11, uv, boîtes x1y1x2y2 en pixels, classes 0–9, rythme Elie : explication avant chaque tâche + accord, questions de compréhension après, glossaire enrichi).
- Tuile par défaut : 640 × 640 px (taille d'entrée du modèle), chevauchement 0,2 (valeur par défaut de SAHI).
- Fusion par défaut : par catégorie, seuil 0,5.
- Tailles d'objets : convention COCO (small < 32², medium < 96², large), comme `dronetrack.data.stats.size_bucket`.
- Vitesse mesurée sur le MacBook Air M1 (MPS), en images par seconde, sur les 548 images de validation.
- Aucun ré-entraînement dans ce plan (voir « Hors périmètre »).

## Review Focus

1. **Image plus petite qu'une tuile** (ex. 480 × 360) → une seule tuile = l'image entière, sans planter (test tâche 2).
2. **Largeur exactement multiple du pas** → pas de tuile en double (test tâche 2).
3. **Objet coupé par le bord d'une tuile** → avec IoS + fusion, la boîte partielle et la boîte entière deviennent une seule boîte couvrant tout l'objet (tests tâches 3 et 4).
4. **Une voiture et une camionnette qui se chevauchent** → la fusion ne supprime jamais une boîte d'une autre catégorie (test tâche 3).
5. **Aucun objet de la taille demandée pour une catégorie** → la catégorie est ignorée dans la mAP par taille, pas comptée 0 (test tâche 5).

## Hors périmètre

Ré-entraîner le modèle sur des tuiles (SAHI « slicing-aided fine-tuning ») : le modèle actuel a appris
sur des images réduites, donc il voit les objets plus petits qu'ils n'apparaissent dans une tuile en
pleine résolution. Si l'expérience 2 montre ce décalage d'échelle, ce sera un plan 2b (~1 USD de GPU).

---

### Task 1: Écart predict() / val() d'Ultralytics (D6), limité à 45 minutes

**Files:**
- Modify (si cause trouvée) : `src/dronetrack/detect/yolo.py`, `tests/test_yolo.py`
- Modify : `docs/DECISIONS.md` (D6)

**Interfaces:**
- Consumes : `YoloDetector`, `scripts/evaluate_detector.py`.
- Produces : rien de nouveau si pas de correctif ; sinon `YoloDetector` garde la même signature.

- [ ] **Step 1:** Sur 5 images de validation, comparer pour `predict()` et pour les prédictions JSON de `val(save_json=True)` : nombre de boîtes, 5 boîtes les plus sûres, taille du tenseur d'entrée (`results[0].orig_shape` et la forme après letterbox, via un callback `on_predict_batch_start` / le `batch` du validateur).
- [ ] **Step 2:** Tester une par une les hypothèses restantes : forme du letterbox (`auto`), demi-précision, ordre des canaux, `max_det` en mode end-to-end, prétraitement `load_image`. Pour chacune, relancer `uv run scripts/evaluate_detector.py --weights models/visdrone_y26n.pt` et noter la mAP@50.
- [ ] **Step 3:** Si une hypothèse ramène notre pipeline à moins de 0,01 de `val()` (0,296) : écrire d'abord un test qui fixe le réglage (ex. `test_yolo_detector_uses_<réglage>`), le voir échouer, corriger `YoloDetector`, le voir passer.
- [ ] **Step 4:** Mettre à jour D6 avec la cause (ou « non trouvée en 45 min » et les hypothèses écartées). Commit `fix: align predict pipeline with val` ou `docs: D6 investigation`.
- [ ] **Step 5:** Noter la mAP@50 de référence « image entière » (10 catégories) qui servira de ligne A à l'expérience 2.

---

### Task 2: Découpage en tuiles

**Files:**
- Create: `src/dronetrack/detect/tiling.py`
- Test: `tests/test_tiling.py`

**Interfaces:**
- Produces : `make_tiles(img_w: int, img_h: int, tile: int = 640, overlap: float = 0.2) -> list[tuple[int, int, int, int]]` : liste de `(x1, y1, x2, y2)` en pixels. Pas = `int(tile * (1 - overlap))` ; la dernière tuile de chaque ligne/colonne est collée au bord droit/bas ; si l'image est plus petite que la tuile dans une dimension, une seule tuile couvre toute cette dimension.

- [ ] **Step 1: Tests qui échouent**

```python
def test_visdrone_det_size_gives_4_by_3_tiles():
    tiles = make_tiles(2000, 1500)
    assert len(tiles) == 12
    assert sorted({t[0] for t in tiles}) == [0, 512, 1024, 1360]
    assert sorted({t[1] for t in tiles}) == [0, 512, 860]

def test_tiles_cover_the_image_and_stay_inside():
    tiles = make_tiles(1344, 756)
    assert all(0 <= x1 < x2 <= 1344 and 0 <= y1 < y2 <= 756 for x1, y1, x2, y2 in tiles)
    covered = np.zeros((756, 1344), bool)
    for x1, y1, x2, y2 in tiles:
        covered[y1:y2, x1:x2] = True
    assert covered.all()

def test_image_smaller_than_tile_is_one_tile():
    assert make_tiles(480, 360) == [(0, 0, 480, 360)]

def test_no_duplicate_tile_when_size_is_a_multiple_of_the_stride():
    tiles = make_tiles(1152, 640)  # 1152 = 512 + 640: second tile already touches the border
    assert len(tiles) == len(set(tiles)) == 2
```

- [ ] **Step 2:** `uv run pytest tests/test_tiling.py -v` → FAIL (import).
- [ ] **Step 3:** Implémenter `make_tiles`.
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** GLOSSAIRE (tuile, chevauchement, pas). Commit `feat: tiling grid`.

---

### Task 3: Fusion des doublons (NMS par IoU ou IoS)

**Files:**
- Modify: `src/dronetrack/geometry.py`, `tests/test_geometry.py`
- Create: `src/dronetrack/detect/merge.py`
- Test: `tests/test_merge.py`

**Interfaces:**
- Consumes : `iou_matrix`, `boxes_to_array`.
- Produces :
  - `ios_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray` : intersection ÷ aire de la plus petite des deux boîtes, (N, M).
  - `nms(dets: Sequence[Detection], threshold: float = 0.5, metric: str = "ios", merge: bool = True) -> list[Detection]` : `metric` ∈ {`"iou"`, `"ios"`} ; par catégorie ; glouton par score décroissant. `merge=False` : suppression classique (NMS). `merge=True` (NMM, comme SAHI) : la boîte gardée devient l'union des boîtes qu'elle absorbe et garde son score. Renvoie les gardées triées par score décroissant.

- [ ] **Step 1: Tests qui échouent**

```python
def test_ios_of_box_inside_another_is_one():
    assert ios_matrix(np.array([[0, 0, 10, 10]], float), np.array([[0, 0, 4, 10]], float))[0, 0] == pytest.approx(1.0)

def test_nms_iou_removes_lower_score_duplicate():
    kept = nms([Detection(0, 0, 10, 10, 0.9, 3), Detection(1, 0, 11, 10, 0.6, 3)], 0.5, "iou", merge=False)
    assert [d.score for d in kept] == [0.9]

def test_partial_box_cut_by_a_tile_border():
    full = Detection(0, 0, 40, 20, 0.8, 3)
    partial = Detection(0, 0, 12, 20, 0.85, 3)  # IoU 0.3, IoS 1.0
    assert nms([full, partial], 0.5, "iou", merge=False) == [partial, full]  # IoU misses the duplicate
    assert nms([full, partial], 0.5, "ios", merge=False) == [partial]  # IoS finds it but keeps the cut box
    assert nms([full, partial], 0.5, "ios", merge=True) == [Detection(0, 0, 40, 20, 0.85, 3)]  # merged: whole object

def test_nms_never_suppresses_another_class():
    car, van = Detection(0, 0, 10, 10, 0.9, 3), Detection(0, 0, 10, 10, 0.8, 4)
    assert len(nms([car, van])) == 2

def test_nms_empty():
    assert nms([]) == []
```

Le test fige les trois comportements : c'est la démonstration, en 3 lignes, du choix de fusion
mesuré dans l'expérience 2.

- [ ] **Step 2:** `uv run pytest tests/test_merge.py tests/test_geometry.py -v` → FAIL.
- [ ] **Step 3:** Implémenter `ios_matrix` puis `nms`.
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** GLOSSAIRE (NMS, NMM / fusion de boîtes, IoS). Commit `feat: class-aware NMS/NMM with IoU or IoS`.

---

### Task 4: Détecteur par tuiles

**Files:**
- Modify: `src/dronetrack/detect/yolo.py` (ajout `detect_batch`), `tests/test_yolo.py`
- Create: `src/dronetrack/detect/tiled.py`
- Test: `tests/test_tiled.py`

**Interfaces:**
- Consumes : `make_tiles`, `nms`, `Detection`.
- Produces :
  - `YoloDetector.detect_batch(images: Sequence[np.ndarray]) -> list[list[Detection]]` (un seul appel `predict` sur la liste) ; `detect(image)` devient `detect_batch([image])[0]`.
  - `class TiledDetector(base, tile: int = 640, overlap: float = 0.2, include_full: bool = True, merge_threshold: float = 0.5, merge_metric: str = "ios", merge_boxes: bool = True)` avec `detect(image: np.ndarray) -> list[Detection]` : découpe, détecte toutes les tuiles en un `detect_batch`, décale chaque boîte de `(x1, y1)` de sa tuile, ajoute les détections de l'image entière si `include_full`, puis `nms(..., merge_threshold, merge_metric, merge_boxes)`. `base` = tout objet ayant `detect_batch`.

- [ ] **Step 1: Tests qui échouent** (avec un faux détecteur écrit dans le test : il renvoie une boîte fixe en coordonnées de tuile et enregistre les tailles reçues)

```python
class FakeDetector:
    """Finds one car at (10, 10, 30, 30) in every image it is given; records the sizes it saw."""
    def __init__(self):
        self.sizes = []
    def detect_batch(self, images):
        self.sizes += [im.shape[:2] for im in images]
        return [[Detection(10, 10, 30, 30, 0.9, 3)] for _ in images]

def test_tile_boxes_are_shifted_to_image_coordinates():
    fake = FakeDetector()
    dets = TiledDetector(fake, tile=640, overlap=0.2, include_full=False).detect(np.zeros((640, 1152, 3), np.uint8))
    assert sorted((d.x1, d.y1) for d in dets) == [(10, 10), (522, 10)]
    assert fake.sizes == [(640, 640), (640, 640)]

def test_full_image_pass_is_added_and_duplicates_merged():
    dets = TiledDetector(FakeDetector(), include_full=True).detect(np.zeros((640, 640, 3), np.uint8))
    assert len(dets) == 1  # the tile and the full image see the same car at the same place

@pytest.mark.slow
def test_yolo_detect_batch_matches_detect():
    det = YoloDetector("yolo26n.pt")
    img = np.zeros((480, 640, 3), np.uint8)
    assert det.detect_batch([img, img]) == [det.detect(img), det.detect(img)]
```

- [ ] **Step 2:** `uv run pytest tests/test_tiled.py -v` → FAIL.
- [ ] **Step 3:** Ajouter `detect_batch` à `YoloDetector` ; implémenter `TiledDetector`.
- [ ] **Step 4:** `uv run pytest -m "slow or not slow" -v` → PASS.
- [ ] **Step 5:** Commit `feat: tiled detector`.

---

### Task 5: mAP par taille d'objet (méthode COCO)

**Files:**
- Modify: `src/dronetrack/eval/det_metrics.py`, `tests/test_det_metrics.py`

**Interfaces:**
- Consumes : `size_bucket` seuils (32², 96²).
- Produces :
  - `SIZE_RANGES: dict[str, tuple[float, float]] = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, inf)}`.
  - `evaluate_detections(..., area_range: tuple[float, float] | None = None)`. Avec un intervalle : une vérité terrain hors intervalle est « ignorée » (ne compte pas dans le rappel) ; une prédiction appariée à une vérité ignorée est ignorée ; une prédiction non appariée dont l'aire est hors intervalle est ignorée ; les prédictions ignorées sortent du calcul (ni vrai ni faux positif). L'appariement préfère une vérité non ignorée. Une catégorie sans vérité non ignorée est sautée.

- [ ] **Step 1: Tests qui échouent**

```python
SMALL, LARGE = Detection(0, 0, 10, 10, 1, 0), Detection(100, 100, 300, 300, 1, 0)

def test_small_range_ignores_large_ground_truth():
    p = {"im": [Detection(0, 0, 10, 10, 0.9, 0)]}  # finds the small one, misses the large one
    assert evaluate_detections(p, {"im": [SMALL, LARGE]}, [0], area_range=SIZE_RANGES["small"]).map50 == pytest.approx(1.0)
    assert evaluate_detections(p, {"im": [SMALL, LARGE]}, [0], area_range=SIZE_RANGES["large"]).map50 == 0.0

def test_prediction_on_ignored_ground_truth_is_not_a_false_positive():
    p = {"im": [Detection(100, 100, 300, 300, 0.95, 0), Detection(0, 0, 10, 10, 0.5, 0)]}
    assert evaluate_detections(p, {"im": [SMALL, LARGE]}, [0], area_range=SIZE_RANGES["small"]).map50 == pytest.approx(1.0)

def test_unmatched_prediction_outside_range_is_ignored():
    p = {"im": [Detection(400, 400, 600, 600, 0.95, 0), Detection(0, 0, 10, 10, 0.5, 0)]}
    assert evaluate_detections(p, {"im": [SMALL]}, [0], area_range=SIZE_RANGES["small"]).map50 == pytest.approx(1.0)

def test_class_without_ground_truth_in_range_is_skipped():
    m = evaluate_detections({"im": []}, {"im": [LARGE]}, [0], area_range=SIZE_RANGES["small"])
    assert m.ap50_per_class == {} and m.map50 == 0.0

def test_no_range_is_unchanged():
    p = {"im": [Detection(0, 0, 10, 10, 0.9, 0)]}
    assert evaluate_detections(p, {"im": [SMALL, LARGE]}, [0]).map50 == pytest.approx(0.5, abs=0.01)
```

- [ ] **Step 2:** `uv run pytest tests/test_det_metrics.py -v` → FAIL.
- [ ] **Step 3:** Étendre `_match_image` pour renvoyer aussi un masque « ignorée » par (prédiction, seuil), et filtrer ces lignes avant le cumul ; compter `n_gt` sur les vérités non ignorées.
- [ ] **Step 4:** `uv run pytest -v` → PASS (tous les anciens tests de mAP inchangés).
- [ ] **Step 5:** GLOSSAIRE (AP small / medium / large). Commit `feat: mAP by object size`.

---

### Task 6: Expérience 2 — précision contre vitesse

**Files:**
- Modify: `scripts/evaluate_detector.py`, `scripts/run_detect.py`, `docs/RESULTATS.md`, `docs/DECISIONS.md`, `README.md`

**Interfaces:**
- Consumes : `TiledDetector`, `evaluate_detections(area_range=...)`, `SIZE_RANGES`.
- Produces :
  - `evaluate_detector.py` gagne `--tiles` (active `TiledDetector`), `--tile 640`, `--overlap 0.2`, `--no-full`, `--merge-metric {iou,ios}`, `--no-merge-boxes`, `--merge-threshold 0.5` ; affiche en plus AP@50 small / medium / large (moyenne sur les catégories) et images/s.
  - `run_detect.py` gagne les mêmes options `--tiles ...`.

- [ ] **Step 1:** Ajouter les options ; vérifier sur `--max-images 20` que les 4 variantes tournent.
- [ ] **Step 2: Expérience 2.** Sur les 548 images, modèle `models/visdrone_y26n.pt`, 10 catégories :
  - A : image entière (référence) ;
  - B : tuiles seules (`--tiles --no-full`) ;
  - C : tuiles + image entière, NMS classique par IoU (`--merge-metric iou --no-merge-boxes`) ;
  - D : tuiles + image entière, IoS + fusion des boîtes (réglage par défaut, comme SAHI).
  Pour chaque variante : mAP@50, mAP@50-95, AP@50 small / medium / large, images/s. Sauver les sorties dans `outputs/exp2_*.txt`.
- [ ] **Step 3:** `RESULTATS.md` : tableau A–D + 3–4 phrases (gain sur les petits objets en %, facteur de ralentissement, effet NMS-IoU contre IoS + fusion, effet sur les grands objets). Remplir la phrase d'entretien « Z % / N fois ».
- [ ] **Step 4:** GIF de la meilleure variante sur la vidéo de l'étape 0 (`docs/media/tiled.gif`, 60 images, 480 px) ; README : ligne cochée « Step 3 », tableau court de l'expérience 2 et le GIF. DECISIONS : D7 (tuile 640, chevauchement 0,2, IoS + fusion) avec les chiffres qui justifient le choix, et note sur le décalage d'échelle si observé (plan 2b).
- [ ] **Step 5:** Commit `feat: experiment 2 - tiling accuracy vs speed` ; push. Proposer à Elie la ligne de CV mise à jour (seulement les chiffres mesurés).
