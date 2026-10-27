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
Validation VisDrone-DET (548 images), modèle YOLO26n ré-entraîné, 10 catégories, notre mAP (une
catégorie par boîte, voir D6). Tuiles de 640 × 640 en pleine résolution, chevauchement 0,2.
Tailles COCO : petit < 32 × 32 px, moyen < 96 × 96 px, grand au-delà.

| Version | Doublons | mAP@50 | petits | moyens | grands | mAP@50-95 |
|---|---|---|---|---|---|---|
| A image entière (référence) | — | 0,273 | 0,169 | 0,414 | **0,580** | 0,153 |
| B' tuiles seules | NMS IoU | 0,360 | 0,277 | 0,475 | 0,412 | 0,199 |
| **C tuiles + image entière** | **NMS IoU** | **0,374** | **0,281** | **0,498** | 0,570 | **0,208** |
| E tuiles + image entière | NMS IoS | 0,370 | 0,280 | 0,487 | 0,541 | 0,206 |
| F tuiles + image entière | fusion IoU | 0,323 | 0,241 | 0,415 | 0,503 | 0,158 |
| D tuiles + image entière | fusion IoS (SAHI) | 0,248 | 0,176 | 0,323 | 0,405 | 0,121 |

(B, tuiles seules avec fusion IoS : 0,275 ; même défaut que D.)

Vitesse sur le MacBook Air M1 (seuil 0,25, détection seule, 100 images, 2 mesures) :
A ≈ 60 images/s (51 à 68), C ≈ 6,4 images/s (6,3 à 6,4), soit **environ 10 fois plus lent** :
une image VisDrone donne 8 à 12 tuiles, plus le passage sur l'image entière.

Lecture :
- **Les tuiles améliorent la détection des petits objets de 66 %** (AP@50 0,169 → 0,281) et la mAP
  globale de 37 % (0,273 → 0,374), **au prix d'une vitesse divisée par 10**.
- Les tuiles seules (B') perdent sur les **grands** objets (0,580 → 0,412) : un camion de plus de
  128 px peut n'être vu entier par aucune tuile. Ajouter l'image entière (C) les récupère (0,570).
- **Surprise : la fusion de boîtes de SAHI (D) est pire que de ne rien faire** (0,248 < 0,273).
  Le tableau isole la cause : mesurer le recouvrement par IoS au lieu d'IoU ne change presque rien
  (E ≈ C) ; c'est la **fusion** qui fait chuter la note (F < C, D << E). Dans les parkings et les
  foules de VisDrone, deux objets voisins de même catégorie se touchent ; la fusion les réunit en une
  grande boîte qui ne correspond à aucun des deux, et les deux sont comptés comme manqués. La
  combinaison IoS + fusion est la pire, car l'IoS juge « même objet » des boîtes qui ne se touchent
  qu'un peu (la petite zone commune est grande par rapport à la plus petite boîte).
- Réglage retenu par défaut : **C** (tuiles + image entière, suppression des doublons par IoU, sans fusion).
- Sur la vidéo de l'étape 0, avec C : 48 détections par image (30 sans tuiles, 15 avant ré-entraînement)
  pour 38,7 objets annotés : les tuiles trouvent plus d'objets mais ajoutent aussi des fausses alertes
  au seuil 0,25.
- Les mesures de vitesse varient beaucoup d'une exécution à l'autre (MacBook Air sans ventilateur, qui
  ralentit en chauffant) : seules les mesures faites l'une après l'autre dans les mêmes conditions sont comparées.

## Expérience 3 — Notre tracker contre ByteTrack (Ultralytics)
Validation VisDrone-MOT : 7 vidéos, 2 846 images, 5 catégories évaluées (pedestrian, car, van, truck,
bus), zones ignorées appliquées. Détecteur YOLO26n ré-entraîné, image entière, seuil 0,1 ; détections
calculées une fois et mises en cache : **les deux trackers reçoivent exactement les mêmes boîtes**.
Réglages identiques (bytetrack.yaml d'Ultralytics).

Vérification de nos métriques contre TrackEval (outil de référence des articles), catégorie car, 3 vidéos,
2 trackers : IDF1 identique, MOTA à moins de 0,0005 (départage des égalités).

| Tracker | MOTA | IDF1 | Changements de numéro | Fausses alertes | Ratés | Vitesse (tracker seul) |
|---|---|---|---|---|---|---|
| Notre tracker, première version (sans fuse_score) | 0,070 | 0,403 | 1 385 | 32 485 | 33 718 | — |
| Ultralytics ByteTrack sans fuse_score | 0,044 | 0,383 | 1 944 | 33 225 | 34 333 | — |
| Ultralytics ByteTrack (réglage par défaut) | 0,177 | 0,403 | 1 180 | 21 010 | 37 651 | ≈ 270 images/s |
| **Notre tracker (avec fuse_score)** | **0,193** | **0,416** | **909** | **20 543** | **37 244** | **≈ 350 images/s** |

Lecture :
- La première version perdait nettement en MOTA (0,070 contre 0,177) à cause de 11 000 fausses alertes de
  plus. Tout l'écart venait d'un seul réglage d'Ultralytics, `fuse_score` : avant de relier, il multiplie
  le recouvrement par la confiance de la détection, donc une boîte moyennement sûre doit beaucoup mieux
  recouvrir la trajectoire. Sans ce réglage, Ultralytics tombe à 0,044, sous notre première version.
- Avec ce réglage ajouté, **notre tracker dépasse ByteTrack d'Ultralytics sur les 3 notes** : MOTA 0,193
  contre 0,177, IDF1 0,416 contre 0,403, et **23 % de changements de numéro en moins** (909 contre 1 180).
  Il gagne en MOTA sur 6 vidéos sur 7. Différences de conception qui peuvent l'expliquer (non isolées) :
  association par catégorie (une voiture n'est jamais reliée à un piéton), numéro donné seulement à la
  confirmation, filtre de Kalman sur largeur et hauteur directement.
- **Les erreurs viennent surtout du détecteur** : 57 787 ratés et fausses alertes contre 909 changements
  de numéro. Exemple, vidéo uav0000182 (MOTA −0,15) : 1 970 camionnettes réelles mais 924 boîtes
  « van » prédites, et 8 697 boîtes « car » pour 5 848 voitures. Une camionnette prise pour une voiture
  compte deux fois (fausse alerte « car » + raté « van »). Des voitures annotées sous les arbres, invisibles,
  sont aussi comptées comme ratées.
- **Les tuiles dégradent le suivi** : avec la détection par tuiles, MOTA −0,18 et IDF1 0,34 (contre 0,19 et
  0,42). Elles produisent 2,6 fois plus de fausses alertes au seuil 0,1 (53 598 contre 20 543). Les tuiles
  améliorent la mAP (qui juge le classement des boîtes pour tous les seuils) mais un tracker travaille à un
  seuil fixe, où elles ajoutent surtout du bruit. Ré-régler les seuils du tracker pour les tuiles reste à tester.
- Le tracker seul tourne à plusieurs centaines d'images par seconde : le coût du système est dans le détecteur.
