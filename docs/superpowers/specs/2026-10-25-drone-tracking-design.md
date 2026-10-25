# Drone Tracking — Design (spec)

Date : 2026-10-25 · Auteur : Elie Inbar (code écrit par Claude)

## 1. Objectif

Projet portfolio pour décrocher un poste junior en computer vision dans la défense tech en Israël
(cibles les plus alignées : Regulus, Sentrycs, D-Fend, Rafael, Mobileye, Elbit).
Elie doit pouvoir **expliquer et défendre chaque brique en entretien**.

**Mission :** un drone filme d'en haut ; le système détecte chaque véhicule et piéton et lui donne
un numéro d'identité qu'il garde d'une image à l'autre (détection + suivi multi-objets).

**Suite prévue (hors de cette spec) :**
- Projet B : anti-drone (caméra au sol, détecter et suivre des drones dans le ciel), en réutilisant
  les briques tuiles / suivi / évaluation / visualisation sans les modifier.
- Benchmark de vitesse Python vs ONNX vs C++.

## 2. Contraintes

- Matériel : MacBook Air M1 (développement, tests, démos sur extraits courts, accélération MPS).
- Entraînement : GPU gratuit Kaggle (~30 h/semaine, sessions de 12 h max) ; Modal en secours.
- Claude écrit tout le code ; Elie comprend. Le goulot d'étranglement est le temps d'entraînement
  et le temps de compréhension, pas le temps de codage.
- Elie n'a pas de background CV (mais a des bases solides en ML et a déjà entraîné un CNN).
- Règle CV : une compétence n'apparaît sur le CV que si l'étape correspondante est terminée.
  Retirer dès maintenant « TensorRT, < 10 ms, INT8 » et « Docker » de la ligne du projet drone.

## 3. Principe de construction : librairie pour le détecteur, fait maison pour le reste

| Brique | Choix | Pourquoi |
|---|---|---|
| Détecteur | YOLO récent d'Ultralytics (nano ou small), ré-entraîné | Réécrire YOLO n'apporte rien en entretien |
| Tuiles pour petits objets | **Fait maison** (inspiré de SAHI) | Question d'entretien probable |
| Suivi | **Fait maison** : ByteTrack simplifié (Kalman + hongrois + association en deux passes) | Cœur des questions algo |
| Évaluation | **Fait maison** : mAP, MOTA, IDF1 | Prouve la compréhension des métriques |
| Vérification | Chaque brique maison est comparée à l'outil officiel (Ultralytics val, TrackEval, ByteTrack officiel) | Répond à « comment sais-tu que c'est juste ? » |

## 4. Données

- **VisDrone2019-DET** (images annotées, 10 classes : pedestrian, people, bicycle, car, van, truck,
  tricycle, awning-tricycle, bus, motor) → entraînement et évaluation du détecteur.
- **VisDrone2019-MOT** (vidéos avec identités) → évaluation du suivi.
- Conversion des annotations VisDrone vers le format YOLO (centre, largeur, hauteur normalisés).
  Les zones « ignored regions » et la classe « others » sont exclues de l'entraînement.
- Licence VisDrone : usage non commercial / académique — compatible portfolio public.

## 5. Architecture du code

Chaque brique fait une seule chose et toutes échangent les mêmes objets simples :

- `Detection` : boîte (x1, y1, x2, y2 en pixels), score de confiance, classe. Produit par le détecteur.
- `Track` : `Detection` + identifiant entier stable dans le temps. Produit par le tracker.

Le tracker ne connaît pas YOLO : il reçoit des `Detection`. C'est ce qui permet de passer au projet B
en changeant seulement le détecteur et les données.

```
drone_tracking/
├── src/dronetrack/
│   ├── types.py          ← Detection, Track
│   ├── data/             ← lecture VisDrone, conversion YOLO, statistiques
│   ├── detect/           ← wrapper YOLO, découpage/recollage en tuiles
│   ├── track/            ← filtre de Kalman, association (IoU + hongrois), tracker
│   ├── eval/             ← mAP, MOTA, IDF1
│   └── viz/              ← dessin des boîtes et identités sur la vidéo
├── scripts/              ← convertir, détecter, suivre, évaluer
├── notebooks/            ← notebook d'entraînement Kaggle
├── tests/                ← pytest
├── docs/
│   ├── GLOSSAIRE.md      ← jargon expliqué simplement, enrichi à chaque étape
│   ├── DECISIONS.md      ← chaque choix technique et sa raison
│   └── RESULTATS.md      ← chiffres des trois expériences
└── README.md
```

Flux : vidéo → image → `detect` → `list[Detection]` → `track` → `list[Track]` → `viz` (vidéo annotée)
et `eval` (scores).

Outils : Python 3.11, `uv`, `pytest`, Git, dépôt GitHub public dès le premier jour.

## 6. Étapes et critères de fin

Avant chaque étape : 3–4 phrases simples sur où on en est et à quoi elle sert, accord d'Elie.
Après : ce qui a changé concrètement, 1–2 questions de compréhension, mise à jour du glossaire.

| Étape | Contenu | Fini quand |
|---|---|---|
| 0. Démarrage | Dépôt, environnement, YOLO pré-entraîné COCO sur une vidéo VisDrone, sur le Mac | Un premier GIF existe (point de départ) |
| 1. Données | Conversion VisDrone → YOLO, statistiques (taille des objets, classes rares) | Tests verts ; une image annotée affichée correspond à la réalité |
| 2. Ré-entraînement | Entraînement sur Kaggle ; mAP maison vérifiée contre Ultralytics | **Expérience 1** : mAP avant/après ré-entraînement |
| 3. Tuiles | Découpage avec chevauchement, détection par tuile, recollage + suppression des doublons | **Expérience 2** : gain sur petits objets vs coût en images/seconde |
| 4. Suivi | Tracker maison ; MOTA/IDF1 maison vérifiés contre TrackEval ; comparaison à ByteTrack officiel | **Expérience 3** : notre tracker vs ByteTrack sur les mêmes vidéos |
| 5. Vitrine | README final, GIF, démo Hugging Face Space (Gradio) | Un inconnu comprend le projet en 30 secondes |

## 7. Tests

Chaque brique maison est testée sur de petits cas vérifiables à la main, par exemple :
- IoU de deux boîtes identiques = 1, de deux boîtes disjointes = 0 ;
- Kalman prédit correctement un objet en mouvement rectiligne uniforme ;
- l'association hongroise relie chaque détection à la bonne trajectoire sur un cas à 3 objets ;
- mAP / MOTA maison égaux (à une tolérance près) aux outils officiels sur le même échantillon.

## 8. Risques et parades

| Risque | Parade |
|---|---|
| Kaggle coupe les sessions (12 h, ~30 h/semaine) | Sauvegarde régulière du modèle et reprise ; commencer par le modèle nano |
| Mac M1 lent sur les longues vidéos | Démos sur extraits courts ; calcul lourd sur Kaggle |
| Métriques maison ≠ métriques officielles | Chercher et expliquer l'écart, le noter dans DECISIONS.md |
| Licences | VisDrone non commercial et Ultralytics AGPL : compatibles avec un dépôt public de portfolio |

## 9. Apprentissage (cours Hugging Face Computer Vision)

À lire avant de commencer (~4 h) :
1. Unit 1 — Image
2. Unit 2 — Introduction to CNNs
3. Unit 2 — Introduction to Transfer Learning
4. Unit 6 — Object Detection (la plus importante : bounding box, IoU, NMS, mAP)
5. Unit 2 — YOLO

Plus tard : Unit 7 — Video Processing Basics (avant l'étape 4) ; Unit 9 — Model Optimization
(avant le benchmark C++). Non couvert par le cours et expliqué dans le projet : Kalman, algorithme
hongrois, ByteTrack, MOTA/IDF1, tuiles SAHI.

## 10. Histoire d'entretien

Trois expériences chiffrées, chacune avec une explication :
1. « Le ré-entraînement sur images de drone fait passer la mAP de X à Y. »
2. « Les tuiles améliorent la détection des petits objets de Z %, mais divisent la vitesse par N. »
3. « Mon tracker fait presque aussi bien que ByteTrack officiel, et je sais expliquer ses échecs. »
Plus l'angle personnel : pilote de drone tactique en réserve.
