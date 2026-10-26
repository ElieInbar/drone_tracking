# Résultats

## Étape 0 — YOLO26n pré-entraîné COCO, sans ré-entraînement
Séquence `uav0000086_00000_v` (terrain de basket filmé en oblique), 150 images, MacBook Air M1 (MPS) :
- 15,4 détections par image, pour 38,7 objets annotés par image (25,6 piétons debout, 11,2 personnes assises, 2 motos).
- 19,4 images par seconde.
- Ce qu'on voit : les personnes proches de la caméra sont bien trouvées ; les personnes lointaines
  (fond de l'image, près du grillage) sont manquées ; deux personnes collées reçoivent parfois une seule boîte.

## Données (VisDrone2019-DET, après conversion)
- Entraînement : 6471 images, 343 204 objets, 53 objets par image en moyenne.
- Validation : 548 images, 38 759 objets (même total que la conversion officielle d'Ultralytics).
- **60,5 % des objets d'entraînement sont petits** (aire < 32 × 32 px), 34,0 % moyens, 5,5 % grands.
  Sur COCO, environ 41 % des objets sont petits : VisDrone est nettement plus difficile.
- Catégories très déséquilibrées : car 42,2 %, pedestrian 23,1 %… mais awning-tricycle 0,9 %,
  tricycle 1,4 %, bus 1,7 %. Il faut s'attendre à de mauvais scores sur les catégories rares.

## Expérience 1 — Gain du ré-entraînement sur images de drone
*(étape 2)*

## Expérience 2 — Tuiles pour les petits objets : précision contre vitesse
*(étape 3)*

## Expérience 3 — Notre tracker contre ByteTrack officiel
*(étape 4)*
