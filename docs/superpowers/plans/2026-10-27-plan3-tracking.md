# Plan 3 — Suivi multi-objets (étape 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Un tracker multi-objets fait maison (ByteTrack simplifié : filtre de Kalman + algorithme hongrois + association en deux passes), des métriques MOTA / IDF1 faites maison et vérifiées contre un outil officiel, et l'expérience 3 : notre tracker contre ByteTrack, sur les mêmes détections.

**Architecture:** Le tracker reçoit des `list[Detection]` (de `YoloDetector` ou `TiledDetector`, il ne sait pas lequel) et renvoie des `list[Track]`. Trois briques séparées dans `track/` : `kalman.py` (prédire où sera la boîte), `matching.py` (relier boîtes et trajectoires), `byte_tracker.py` (la logique ByteTrack). Les métriques vont dans `eval/mot_metrics.py`. Les détections de chaque vidéo sont calculées une seule fois et mises en cache, pour comparer les trackers sur exactement les mêmes boîtes.

**Tech Stack:** Python 3.11, numpy, scipy (`linear_sum_assignment`, à ajouter), ultralytics (détecteur ré-entraîné `models/visdrone_y26n.pt`, et son `BYTETracker` comme référence), pytest. Vérification des métriques : TrackEval, ou py-motmetrics si TrackEval ne s'installe pas avec numpy 2. Tout tourne sur le Mac M1.

**Spec:** `docs/superpowers/specs/2026-10-25-drone-tracking-design.md` (section 3 « Suivi », section 4 « VisDrone2019-MOT, 5 classes », section 6 étape 4, section 10 expérience 3).

## Global Constraints

- Contraintes globales des plans 1 et 2 inchangées (uv, boîtes x1y1x2y2, classes 0–9, rythme Elie : récapitulatif simple en début de journée, explication avant chaque tâche + accord, questions après, glossaire).
- Données de suivi : `data/VisDrone2019-MOT-val` (7 séquences, 2 846 images), déjà téléchargées.
- Format d'annotation MOT : `frame, id, gauche, haut, largeur, hauteur, score, catégorie, troncature, occlusion` ; catégorie 0 = zone ignorée, 1–10 = classes (YOLO = catégorie − 1), 11 = others.
- Classes évaluées (protocole VisDrone-MOT, spec section 4) : `EVAL_CLASSES = (0, 3, 4, 5, 8)` = pedestrian, car, van, truck, bus.
- Réglages du tracker = ceux du `bytetrack.yaml` d'Ultralytics, pour comparer les algorithmes et non les réglages : seuil haut 0,25, seuil bas 0,1, nouvelle trajectoire 0,25, mémoire 30 images, association passe 1 si IoU ≥ 0,2 (match_thresh 0,8 en coût), passe 2 si IoU ≥ 0,5.
- Détecteur pour le suivi : image entière (rapide), seuil 0,1 (le tracker a besoin des boîtes peu sûres pour sa 2e passe). Les tuiles restent une option (`--tiles`).

## Review Focus

1. **Objet qui disparaît quelques images** (passe sous un arbre) → il retrouve son numéro s'il revient avant 30 images (test tâche 4).
2. **Deux objets qui se croisent** → chacun garde son numéro (test tâche 4).
3. **Une voiture et un piéton qui se recouvrent** → jamais associés l'un à l'autre (association par catégorie, test tâche 3).
4. **Image sans aucune détection** → les trajectoires vieillissent sans planter (test tâche 4).
5. **Prédiction dans une zone ignorée** (foule non annotée) → ni bonne réponse ni fausse alerte (test tâche 5).

---

### Task 1: Lire les annotations de suivi VisDrone-MOT

**Files:**
- Modify: `pyproject.toml` (ajout `scipy`), `src/dronetrack/data/visdrone.py`
- Test: `tests/test_visdrone.py`

**Interfaces:**
- Produces :
  - `EVAL_CLASSES: tuple[int, ...] = (0, 3, 4, 5, 8)` dans `visdrone.py`.
  - `parse_mot_annotation(path: Path) -> tuple[dict[int, list[Track]], dict[int, list[Detection]]]` : (vérité terrain par numéro d'image, zones ignorées par numéro d'image). Garde `score == 1` et catégorie 1–10 (classe = catégorie − 1, `track_id` = id du fichier) ; catégorie 0 → zone ignorée (`Detection` de classe −1) ; catégorie 11 jetée. Boîtes de largeur ou hauteur ≤ 0 jetées.

- [ ] **Step 1: Tests qui échouent**

```python
def test_parse_mot_groups_by_frame_and_keeps_ids(tmp_path):
    p = tmp_path / "seq.txt"
    p.write_text("1,7,10,20,30,40,1,4,0,0\n2,7,12,20,30,40,1,4,0,0\n1,8,0,0,5,5,1,1,0,0\n")
    gt, ignored = parse_mot_annotation(p)
    assert gt[1] == [Track(10, 20, 40, 60, 1.0, 3, track_id=7), Track(0, 0, 5, 5, 1.0, 0, track_id=8)]
    assert gt[2][0].x1 == 12 and ignored == {}

def test_parse_mot_separates_ignored_regions_and_drops_others(tmp_path):
    p = tmp_path / "seq.txt"
    p.write_text("1,0,0,0,50,50,1,0,0,0\n1,3,0,0,5,5,1,11,0,0\n1,4,0,0,5,5,0,4,0,0\n")
    gt, ignored = parse_mot_annotation(p)
    assert gt == {} and ignored[1] == [Detection(0, 0, 50, 50, 1.0, -1)]
```

- [ ] **Step 2:** `uv add scipy` ; `uv run pytest tests/test_visdrone.py -v` → FAIL (import).
- [ ] **Step 3:** Implémenter `parse_mot_annotation` et `EVAL_CLASSES`.
- [ ] **Step 4:** `uv run pytest -v` → PASS. Vérifier sur `uav0000086_00000_v` : nombre d'images annotées, nombre d'identités distinctes, objets par image (affiché, pas de seuil attendu).
- [ ] **Step 5:** GLOSSAIRE (suivi multi-objets / MOT, trajectoire, identité). Commit `feat: VisDrone-MOT annotation reader`.

---

### Task 2: Filtre de Kalman (prédire la prochaine position)

**Files:**
- Create: `src/dronetrack/track/__init__.py`, `src/dronetrack/track/kalman.py`
- Test: `tests/test_kalman.py`

**Interfaces:**
- Produces : `class KalmanBoxFilter` (une instance par trajectoire), état `[cx, cy, w, h, vx, vy, vw, vh]` à vitesse constante.
  - `__init__(box: np.ndarray)` : box = `[x1, y1, x2, y2]`, vitesses initiales 0.
  - `predict() -> np.ndarray` : avance d'une image, renvoie la boîte prédite `[x1, y1, x2, y2]`.
  - `update(box: np.ndarray) -> None` : corrige l'état avec une mesure.
  - `box` (propriété) : boîte actuelle `[x1, y1, x2, y2]`.
  - Bruits proportionnels à la taille de la boîte, comme ByteTrack : écart-type position 1/20 de la hauteur, vitesse 1/160.

- [ ] **Step 1: Tests qui échouent**

```python
def test_learns_constant_velocity():
    kf = KalmanBoxFilter(np.array([0, 0, 10, 10], float))
    for t in range(1, 11):  # the object moves 5 px right per frame
        kf.predict()
        kf.update(np.array([5 * t, 0, 5 * t + 10, 10], float))
    assert kf.predict() == pytest.approx([55, 0, 65, 10], abs=1.0)

def test_predict_without_update_keeps_moving():
    kf = KalmanBoxFilter(np.array([0, 0, 10, 10], float))
    for t in range(1, 11):
        kf.predict(); kf.update(np.array([5 * t, 0, 5 * t + 10, 10], float))
    for _ in range(3):  # occluded for 3 frames: no measurement
        box = kf.predict()
    assert box[0] == pytest.approx(65, abs=2.0)

def test_update_pulls_toward_measurement():
    kf = KalmanBoxFilter(np.array([0, 0, 10, 10], float))
    kf.predict()
    kf.update(np.array([4, 0, 14, 10], float))
    assert 0 < kf.box[0] <= 4
```

- [ ] **Step 2:** `uv run pytest tests/test_kalman.py -v` → FAIL.
- [ ] **Step 3:** Implémenter avec numpy (matrices F, H, Q, R ; équations standard prédiction / correction).
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** GLOSSAIRE (filtre de Kalman, état, prédiction / correction, vitesse constante). Commit `feat: Kalman box filter`.

---

### Task 3: Association détections ↔ trajectoires (algorithme hongrois)

**Files:**
- Create: `src/dronetrack/track/matching.py`
- Test: `tests/test_matching.py`

**Interfaces:**
- Consumes : `iou_matrix`.
- Produces : `associate(track_boxes: np.ndarray, track_cls: np.ndarray, det_boxes: np.ndarray, det_cls: np.ndarray, min_iou: float) -> tuple[list[tuple[int, int]], list[int], list[int]]` : (paires (i_trajectoire, j_détection), trajectoires sans détection, détections sans trajectoire). Coût = 1 − IoU ; paires de catégories différentes ou d'IoU < `min_iou` interdites ; résolu par `scipy.optimize.linear_sum_assignment`.

- [ ] **Step 1: Tests qui échouent**

```python
B = np.array([[0, 0, 10, 10], [100, 0, 110, 10], [200, 0, 210, 10]], float)

def test_three_objects_matched_whatever_the_order():
    dets = B[[2, 0, 1]] + 2  # same objects, shuffled, moved 2 px
    m, ut, ud = associate(B, np.zeros(3), dets, np.zeros(3), 0.2)
    assert sorted(m) == [(0, 1), (1, 2), (2, 0)] and ut == [] and ud == []

def test_different_class_never_matched():
    m, ut, ud = associate(B[:1], np.array([3]), B[:1], np.array([0]), 0.2)
    assert m == [] and ut == [0] and ud == [0]

def test_low_overlap_rejected():
    m, ut, ud = associate(B[:1], np.zeros(1), B[:1] + 9, np.zeros(1), 0.2)  # IoU 1/199
    assert m == [] and ut == [0] and ud == [0]

def test_hungarian_beats_greedy():
    # greedy would give track 0 its best det (IoU .5) and leave track 1 with nothing usable
    tracks = np.array([[0, 0, 10, 10], [6, 0, 16, 10]], float)
    dets = np.array([[3, 0, 13, 10], [-4, 0, 6, 10]], float)
    m, _, _ = associate(tracks, np.zeros(2), dets, np.zeros(2), 0.2)
    assert sorted(m) == [(0, 1), (1, 0)]

def test_empty_inputs():
    assert associate(np.zeros((0, 4)), np.zeros(0), B, np.zeros(3), 0.2) == ([], [], [0, 1, 2])
```

- [ ] **Step 2:** `uv run pytest tests/test_matching.py -v` → FAIL.
- [ ] **Step 3:** Implémenter `associate`.
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** GLOSSAIRE (problème d'affectation, algorithme hongrois). Commit `feat: Hungarian association`.

---

### Task 4: Le tracker (ByteTrack simplifié)

**Files:**
- Create: `src/dronetrack/track/byte_tracker.py`
- Test: `tests/test_byte_tracker.py`

**Interfaces:**
- Consumes : `KalmanBoxFilter`, `associate`, `Detection`, `Track`.
- Produces : `class ByteTracker(high: float = 0.25, low: float = 0.1, new_track: float = 0.25, max_lost: int = 30, first_min_iou: float = 0.2, second_min_iou: float = 0.5)` avec `update(dets: Sequence[Detection]) -> list[Track]` (une image à la fois ; renvoie les trajectoires confirmées vues sur cette image, boîte = boîte corrigée par Kalman, score = celui de la détection). Algorithme, à chaque image :
  1. prédire toutes les trajectoires (Kalman) ;
  2. passe 1 : détections de score ≥ `high` contre toutes les trajectoires (actives et perdues), IoU ≥ `first_min_iou` ;
  3. passe 2 : détections de score dans [`low`, `high`) contre les trajectoires **actives** restées seules, IoU ≥ `second_min_iou` ;
  4. détections fortes non associées de score ≥ `new_track` → nouvelles trajectoires, « à confirmer » ; confirmées quand elles sont revues à l'image suivante (sauf sur la toute première image, confirmées tout de suite) ; une trajectoire à confirmer non revue est supprimée ;
  5. trajectoires non associées → « perdues » ; supprimées après `max_lost` images sans être revues.
  Les numéros commencent à 1 et ne sont jamais réutilisés.

- [ ] **Step 1: Tests qui échouent** (petites vidéos synthétiques faites de listes de `Detection`)

```python
def car(x, score=0.9):
    return Detection(x, 0, x + 20, 20, score, 3)

def run(frames, **kw):
    t = ByteTracker(**kw)
    return [t.update(f) for f in frames]

def test_moving_object_keeps_its_id():
    out = run([[car(5 * i)] for i in range(10)])
    assert {tr.track_id for f in out for tr in f} == {1}

def test_object_reappearing_after_occlusion_keeps_its_id():
    frames = [[car(5 * i)] for i in range(10)] + [[] for _ in range(5)] + [[car(5 * i)] for i in range(15, 20)]
    out = run(frames)
    assert out[10] == [] and {tr.track_id for f in out[15:] for tr in f} == {1}

def test_object_lost_too_long_gets_a_new_id():
    frames = [[car(0)], [car(0)]] + [[] for _ in range(31)] + [[car(0)], [car(0)]]
    assert run(frames)[-1][0].track_id == 2

def test_low_score_detection_keeps_track_alive():
    frames = [[car(5 * i)] for i in range(5)] + [[car(25, score=0.15)]] + [[car(30)]]
    out = run(frames)
    assert len(out[5]) == 1 and out[6][0].track_id == 1

def test_low_score_detection_alone_never_starts_a_track():
    assert run([[car(0, score=0.15)]] * 3) == [[], [], []]

def test_crossing_objects_keep_their_ids():
    frames = [[car(10 * i), Detection(200 - 10 * i, 5, 220 - 10 * i, 25, 0.9, 3)] for i in range(21)]
    out = run(frames)
    first = {tr.track_id: tr.x1 for tr in out[0]}
    last = {tr.track_id: tr.x1 for tr in out[-1]}
    left_id = min(first, key=first.get)
    assert last[left_id] > 150  # the object that started on the left ended on the right

def test_new_track_needs_a_second_sighting_after_first_frame():
    out = run([[car(0)], [car(0), car(300)], [car(0), car(300)]])
    assert len(out[1]) == 1 and len(out[2]) == 2

def test_empty_frames_do_not_crash():
    assert run([[], [], []]) == [[], [], []]
```

- [ ] **Step 2:** `uv run pytest tests/test_byte_tracker.py -v` → FAIL.
- [ ] **Step 3:** Implémenter `ByteTracker` (une petite classe interne `_Track` : filtre de Kalman, id, classe, score, état actif / perdu / à confirmer, nombre d'images sans être vue).
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** GLOSSAIRE (ByteTrack, association en deux passes, trajectoire perdue, changement d'identité). Commit `feat: ByteTrack-style tracker`.

---

### Task 5: MOTA et IDF1 faits maison

**Files:**
- Create: `src/dronetrack/eval/mot_metrics.py`
- Test: `tests/test_mot_metrics.py`

**Interfaces:**
- Consumes : `iou_matrix`, `ios_matrix`, `linear_sum_assignment`.
- Produces :
  - `@dataclass class MotMetrics: mota: float; idf1: float; idsw: int; fp: int; fn: int; n_gt: int; idtp: int`.
  - `evaluate_tracking(gt: Mapping[int, list[Track]], pred: Mapping[int, list[Track]], classes: Sequence[int], ignored: Mapping[int, list[Detection]] | None = None, min_iou: float = 0.5) -> MotMetrics` : clés = numéros d'image. Pour chaque classe séparément puis sommé :
    - **MOTA (CLEAR MOT)** : à chaque image, on garde d'abord les paires de l'image précédente encore valables (même id réel ↔ même id prédit, IoU ≥ `min_iou`), puis on associe le reste par l'algorithme hongrois (coût 1 − IoU). Un objet réel associé à un autre id prédit qu'à sa dernière association = 1 changement d'identité (IDSW). Avant de compter les fausses alertes, une prédiction non associée dont l'IoS avec une zone ignorée atteint 0,5 est retirée. MOTA = 1 − (FN + FP + IDSW) / nombre d'objets réels.
    - **IDF1** : association globale (hongrois) entre identités réelles et identités prédites, en maximisant le nombre d'images où elles se recouvrent (IoU ≥ `min_iou`) ; IDTP = ce nombre ; IDF1 = 2·IDTP / (nombre d'objets réels + nombre de prédictions comptées).

- [ ] **Step 1: Tests qui échouent**

```python
def tr(x, tid, cls=3):
    return Track(x, 0, x + 10, 10, 1.0, cls, track_id=tid)

GT = {1: [tr(0, 1)], 2: [tr(5, 1)], 3: [tr(10, 1)], 4: [tr(15, 1)]}

def test_perfect_tracking():
    m = evaluate_tracking(GT, {f: [tr(b[0].x1, 9)] for f, b in GT.items()}, [3])
    assert (m.mota, m.idf1, m.idsw, m.fp, m.fn) == (1.0, 1.0, 0, 0, 0)

def test_one_identity_switch():
    pred = {1: [tr(0, 9)], 2: [tr(5, 9)], 3: [tr(10, 4)], 4: [tr(15, 4)]}
    m = evaluate_tracking(GT, pred, [3])
    assert m.idsw == 1 and m.mota == pytest.approx(0.75) and m.idf1 == pytest.approx(0.5)

def test_missed_frame_and_false_alarm():
    pred = {1: [tr(0, 9)], 2: [], 3: [tr(10, 9), tr(100, 5)], 4: [tr(15, 9)]}
    m = evaluate_tracking(GT, pred, [3])
    assert (m.fn, m.fp, m.idsw) == (1, 1, 0) and m.mota == pytest.approx(0.5)
    assert m.idf1 == pytest.approx(2 * 3 / (4 + 4))

def test_prediction_in_ignored_region_is_not_a_false_alarm():
    pred = {1: [tr(0, 9), tr(100, 5)]}
    m = evaluate_tracking({1: [tr(0, 1)]}, pred, [3], ignored={1: [Detection(95, 0, 120, 10, 1.0, -1)]})
    assert m.fp == 0 and m.mota == 1.0

def test_other_classes_are_not_evaluated():
    m = evaluate_tracking(GT, {f: [tr(b[0].x1, 9, cls=0)] for f, b in GT.items()}, [3])
    assert (m.fn, m.fp) == (4, 0)
```

- [ ] **Step 2:** `uv run pytest tests/test_mot_metrics.py -v` → FAIL.
- [ ] **Step 3:** Implémenter `evaluate_tracking`.
- [ ] **Step 4:** `uv run pytest -v` → PASS.
- [ ] **Step 5:** GLOSSAIRE (MOTA, IDF1, IDSW, CLEAR MOT). Commit `feat: homemade MOTA and IDF1`.

---

### Task 6: Script de suivi, cache des détections, dessin des numéros

**Files:**
- Create: `src/dronetrack/track/cache.py`, `scripts/run_track.py`
- Modify: `src/dronetrack/viz/draw.py`
- Test: `tests/test_cache.py`, `tests/test_draw.py`

**Interfaces:**
- Consumes : `build_detector`, `add_detector_args`, `iter_frames`, `ByteTracker`.
- Produces :
  - `save_detections(path: Path, dets_per_frame: list[list[Detection]]) -> None` et `load_detections(path: Path) -> list[list[Detection]]` (JSON, une liste par image, dans l'ordre des images).
  - `detect_sequence(detector, source: Path, cache: Path | None) -> list[list[Detection]]` : relit le cache s'il existe, sinon détecte et l'écrit.
  - `draw_tracks(image: np.ndarray, tracks: Sequence[Track], class_names: Sequence[str]) -> np.ndarray` : couleur fixe par numéro (pas par classe), étiquette `#id classe`.
  - `scripts/run_track.py --source SEQ --weights W [--tiles ...] [--tracker ours|bytetrack] --conf 0.1 --gif OUT --max-frames N` : vidéo / GIF avec numéros. Cache : `outputs/dets/<nom-séquence>_<variante>.json`.

- [ ] **Step 1: Tests qui échouent**

```python
def test_detections_round_trip(tmp_path):
    frames = [[Detection(1, 2, 3, 4, 0.5, 3)], []]
    save_detections(tmp_path / "d.json", frames)
    assert load_detections(tmp_path / "d.json") == frames

def test_same_id_same_color_across_frames():
    img = np.zeros((40, 40, 3), np.uint8)
    a = draw_tracks(img, [Track(5, 5, 15, 15, 0.9, 3, track_id=4)], CLASS_NAMES)
    b = draw_tracks(img, [Track(5, 5, 15, 15, 0.9, 0, track_id=4)], CLASS_NAMES)
    assert tuple(a[5, 10]) == tuple(b[5, 10])  # top edge pixel: color depends on the id only
```

- [ ] **Step 2:** `uv run pytest tests/test_cache.py tests/test_draw.py -v` → FAIL.
- [ ] **Step 3:** Implémenter le cache, `draw_tracks`, le script (`--tracker bytetrack` est branché à la tâche 7).
- [ ] **Step 4:** `uv run pytest -v` → PASS ; `uv run scripts/run_track.py --source data/VisDrone2019-MOT-val/sequences/uav0000086_00000_v --weights models/visdrone_y26n.pt --gif outputs/track_smoke.gif --max-frames 60` → GIF créé ; le montrer à Elie.
- [ ] **Step 5:** Commit `feat: tracking script, detection cache and ID drawing`.

---

### Task 7: Expérience 3 — notre tracker contre ByteTrack, et vérification des métriques

**Files:**
- Create: `src/dronetrack/track/reference.py`, `scripts/evaluate_tracker.py`
- Modify: `scripts/run_track.py`, `docs/RESULTATS.md`, `docs/DECISIONS.md`, `README.md`

**Interfaces:**
- Consumes : tout ce qui précède ; `ultralytics.trackers.byte_tracker.BYTETracker`, `ultralytics.engine.results.Boxes`, `ultralytics.cfg.get_cfg` + `bytetrack.yaml`.
- Produces :
  - `class UltralyticsByteTrack` avec la même interface `update(dets: Sequence[Detection]) -> list[Track]` : convertit les `Detection` en `Boxes` (`[x1, y1, x2, y2, score, cls]`), appelle `BYTETracker.update`, reconvertit la sortie `[x1, y1, x2, y2, id, score, cls, idx]`.
  - `scripts/evaluate_tracker.py --weights W [--tiles ...] --trackers ours bytetrack [--sequences ...] [--check-reference]` : pour chaque séquence de `data/VisDrone2019-MOT-val`, détecte une fois (cache), fait tourner chaque tracker sur les mêmes détections, calcule MOTA, IDF1, IDSW, FP, FN (5 classes) et la vitesse du tracker seul (images/s, détection exclue). Affiche un tableau par séquence et le total. `--check-reference` recalcule MOTA et IDF1 avec l'outil officiel sur la même sortie.

- [ ] **Step 1:** `UltralyticsByteTrack` ; test lent `test_reference_tracker_follows_a_moving_car` (même vidéo synthétique que la tâche 4 → un seul numéro). `uv run pytest -m "slow or not slow" -v` → PASS.
- [ ] **Step 2: Vérification des métriques.** Installer TrackEval (`uv add --dev "trackeval @ git+https://github.com/JonathonLuiten/TrackEval"`) ; s'il plante sur numpy 2 (`np.float`, `np.int`), utiliser `motmetrics` à la place et le noter. Sur 2 séquences et la classe car, MOTA et IDF1 maison doivent égaler l'outil officiel à 0,001 près. Au-delà : chercher la cause avant de continuer, la noter dans DECISIONS.md.
- [ ] **Step 3: Expérience 3.** Les 7 séquences, détecteur image entière seuil 0,1 : notre tracker contre ByteTrack d'Ultralytics. Sauver `outputs/exp3_*.txt`. Puis la même chose avec `--tiles` si le temps le permet (≈ 10 fois plus lent à détecter, mais mis en cache).
- [ ] **Step 4:** `RESULTATS.md` : tableau (MOTA, IDF1, IDSW, FP, FN, vitesse) + 3–4 phrases, avec au moins un exemple d'échec de notre tracker regardé dans une vidéo (où et pourquoi il perd un numéro). DECISIONS : D8 (réglages identiques à Ultralytics, pas de fusion score × IoU, outil de vérification utilisé). README : étape 4 cochée, tableau court, GIF avec numéros (`docs/media/tracking.gif`, 60 images, 480 px).
- [ ] **Step 5:** Commit `feat: experiment 3 - our tracker vs ByteTrack` ; push. Proposer à Elie la ligne de CV (seulement les chiffres mesurés).
