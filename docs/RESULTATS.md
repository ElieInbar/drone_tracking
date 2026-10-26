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
Validation VisDrone-DET (548 images, 38 759 objets), notre mAP (méthode COCO), même pipeline
d'inférence pour les deux modèles (`predict`, 640 px, seuil 0,001). Comparaison sur les 6 catégories
communes à COCO et VisDrone, avec pedestrian + people fusionnés en « personne » (voir D3 et D6).

| Catégorie | YOLO26n COCO (avant) | YOLO26n ré-entraîné (après) |
|---|---|---|
| personne | 0,174 | 0,348 |
| vélo | 0,007 | 0,057 |
| voiture | 0,411 | 0,708 |
| camion | 0,081 | 0,248 |
| bus | 0,135 | 0,354 |
| moto | 0,055 | 0,343 |
| **mAP@50** | **0,144** | **0,343** (× 2,4) |
| **mAP@50-95** | **0,083** | **0,192** (× 2,3) |

Lecture :
- Le ré-entraînement multiplie la mAP par 2,4. Les plus gros gains sont sur les objets que COCO voit
  rarement d'en haut : motos (× 6), vélos (× 8, mais partent de presque rien), camions (× 3).
- Les vélos restent mauvais (0,06) : ce sont les objets les plus petits et ils ne font que 3 % des données.
- Sur les 10 catégories VisDrone, le modèle ré-entraîné obtient mAP@50 = 0,273 (notre pipeline) ;
  les catégories rares (awning-tricycle 0,08, tricycle 0,17) tirent la moyenne vers le bas.
- Sur la vidéo de l'étape 0 : 30 détections par image au lieu de 15, et les personnes lointaines
  près du grillage sont maintenant trouvées.

Entraînement : 50 epochs en 43 min sur un GPU L4 (Modal), environ 0,60 USD. La mAP@50 de validation
plafonne vers 0,30 dès l'epoch 33 ; pas de sur-apprentissage (pertes de validation toujours en baisse).

## Expérience 2 — Tuiles pour les petits objets : précision contre vitesse
*(étape 3)*

## Expérience 3 — Notre tracker contre ByteTrack officiel
*(étape 4)*
