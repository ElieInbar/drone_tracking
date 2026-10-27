# Glossaire

Chaque terme est expliqué simplement, avec un exemple tiré du projet. Le fichier grandit à chaque étape.

## Pixel
Le plus petit point d'une image. Une image couleur est un tableau de nombres de forme
hauteur × largeur × 3 (rouge, vert, bleu), chaque nombre entre 0 et 255.
*Dans le projet :* une image VisDrone fait environ 1500 × 2000 × 3, soit environ 9 millions de nombres.

## Bounding box (boîte englobante)
Le rectangle le plus serré autour d'un objet. On le note par deux coins :
(x1, y1) en haut à gauche et (x2, y2) en bas à droite, en pixels.
Attention : l'axe y descend (y = 0 est le haut de l'image).
*Dans le projet :* c'est l'objet `Detection` de `src/dronetrack/types.py`.

## IoU (Intersection over Union)
Mesure à quel point deux boîtes se recouvrent : surface commune ÷ surface totale couverte.
1 = boîtes identiques, 0 = elles ne se touchent pas.
*Exemple :* deux carrés de 10 × 10 décalés de 5 pixels partagent 50 px² sur 150 px² au total → IoU = 1/3.
*Dans le projet :* sert à décider si une détection est juste (mAP), à supprimer les doublons entre
tuiles, et à relier une détection à la bonne trajectoire (suivi). Code : `src/dronetrack/geometry.py`.

## COCO
Grand jeu de données d'images de la vie courante (80 catégories : personne, voiture, chien, oiseau…),
photographiées surtout à hauteur d'homme. La plupart des détecteurs publiés ont appris dessus.
*Dans le projet :* YOLO26 « tel quel » a appris sur COCO ; ses catégories ne sont pas celles de
VisDrone, d'où la table de correspondance `COCO_TO_VISDRONE` (6 catégories communes).

## Modèle pré-entraîné
Un modèle dont les poids ont déjà appris sur un grand jeu de données. On peut l'utiliser tel quel,
ou le ré-entraîner (fine-tuning) sur nos propres images pour le spécialiser.
*Dans le projet :* `yolo26n.pt` est pré-entraîné sur COCO ; l'étape 2 le spécialise sur VisDrone.

## Score de confiance et seuil
Pour chaque boîte, le détecteur donne un score entre 0 et 1 : à quel point il est sûr de lui.
On ne garde que les boîtes au-dessus d'un seuil (0,25 par défaut dans la démo).
Seuil bas → on rate moins d'objets mais on a plus de fausses alertes ; seuil haut → l'inverse.

## Faux positif / faux négatif
Faux positif : une détection là où il n'y a pas d'objet (une ombre prise pour un piéton).
Faux négatif : un objet réel que le détecteur n'a pas trouvé.

## Annotation (vérité terrain, ground truth)
Les « bonnes réponses » dessinées à la main par des humains : pour chaque image, la boîte et la
catégorie de chaque objet. Le modèle apprend en comparant ses prédictions à ces réponses.
*Dans le projet :* un fichier texte par image dans `data/VisDrone2019-DET-*/annotations/`.

## Format d'annotation VisDrone
Une ligne par objet : `gauche, haut, largeur, hauteur, score, catégorie, troncature, occlusion`, en pixels.
Le « score » vaut 0 pour une boîte à ignorer. Catégorie 0 = zone ignorée, 11 = « others ».

## Format YOLO et coordonnées normalisées
Une ligne par objet : `classe centre_x centre_y largeur hauteur`, chaque valeur divisée par la
taille de l'image (donc entre 0 et 1). Avantage : la même étiquette reste valable si on redimensionne l'image.
*Exemple :* une boîte de 50 × 100 px en haut à gauche d'une image de 200 × 100 →
`3 0.125000 0.500000 0.250000 1.000000`. Code : `src/dronetrack/data/convert.py`.

## Ignored regions (zones ignorées)
Zones de l'image trop floues ou trop denses pour être annotées objet par objet (une foule lointaine).
On ne les apprend pas : ce ne sont ni des objets, ni de vrais « fonds ».

## Petits / moyens / grands objets
Convention de COCO, utilisée partout : petit = aire < 32 × 32 px, moyen = aire < 96 × 96 px, grand sinon.
*Dans le projet :* 60,5 % des objets de VisDrone sont petits. C'est la principale difficulté du
jeu de données, et la raison de l'étape 3 (tuiles).

## Déséquilibre des classes
Quand certaines catégories sont beaucoup plus fréquentes que d'autres. Le modèle voit peu
d'exemples des catégories rares et les apprend mal.
*Dans le projet :* 42 % de voitures, mais moins de 1 % de tricycles à auvent.

## Vrai positif (TP)
Une détection qui recouvre un objet réel de la bonne catégorie avec un IoU au moins égal au seuil
(0,5 en général), et dont l'objet n'a pas déjà été pris par une détection plus sûre.
Une deuxième boîte sur le même objet est un faux positif (doublon).

## Précision
Parmi mes détections, quelle part est juste ? précision = vrais positifs ÷ nombre de détections.
*Exemple :* 10 boîtes dont 8 sur de vrais objets → précision 0,8.

## Rappel (recall)
Parmi les objets réels, quelle part ai-je trouvée ? rappel = vrais positifs ÷ nombre d'objets réels.
*Exemple :* 8 objets trouvés sur 40 → rappel 0,2.

## Courbe précision-rappel
On classe toutes les détections de la plus sûre à la moins sûre et on les ajoute une à une :
le rappel monte, la précision varie. Chaque seuil de confiance possible est un point de la courbe.

## AP (Average Precision)
L'aire sous la courbe précision-rappel d'une catégorie : un seul chiffre entre 0 et 1 qui résume
la qualité du détecteur pour tous les seuils à la fois. Méthode COCO : on lit la précision à
101 niveaux de rappel (0 ; 0,01 ; … ; 1) et on fait la moyenne.

## mAP@50 et mAP@50-95
mAP = moyenne des AP de toutes les catégories.
@50 : une détection compte si son IoU avec l'objet atteint 0,5 (boîte « à peu près » bien placée).
@50-95 : moyenne sur 10 seuils d'IoU de 0,50 à 0,95 ; récompense les boîtes très bien ajustées.
Code : `src/dronetrack/eval/det_metrics.py`.

## Fine-tuning (ré-entraînement)
Reprendre un modèle pré-entraîné et continuer son apprentissage sur nos propres données, pour le
spécialiser. Bien plus rapide et efficace que de partir de zéro.
*Dans le projet :* YOLO26n (COCO) ré-entraîné 50 epochs sur VisDrone → mAP@50 de 0,14 à 0,34.

## Epoch et batch
Une epoch = un passage complet sur toutes les images d'entraînement (6471 ici).
Un batch = le petit paquet d'images traitées ensemble avant chaque mise à jour des poids (16 ici).

## Perte (loss)
Le nombre que l'entraînement cherche à faire baisser : à quel point les prédictions sont loin des
réponses. YOLO26 en suit trois : `box_loss` (placement des boîtes), `cls_loss` (catégorie),
`l1_loss` (écart des bords en coordonnées).

## Jeu de validation
Images mises de côté, jamais utilisées pour apprendre, qui servent à mesurer honnêtement le modèle.
On juge un modèle sur la validation, jamais sur la perte d'entraînement.

## Sur-apprentissage (overfitting)
Le modèle apprend les images d'entraînement par cœur au lieu de règles générales : la perte
d'entraînement baisse mais les résultats de validation stagnent ou se dégradent.

## Mosaïque (augmentation)
Pendant l'entraînement, Ultralytics colle 4 images en une pour créer de la variété. Arrêtée pour
les 10 dernières epochs. *Dans le projet :* la perte d'entraînement chute à l'epoch 40 quand la
mosaïque s'arrête, sans que la mAP de validation bouge : les images sont plus faciles, le modèle
n'est pas meilleur.

## Tuile (tile)
Un morceau carré de l'image (640 × 640 px ici), pris en pleine résolution, que l'on donne au
détecteur à la place de l'image entière réduite. Les petits objets gardent ainsi tous leurs pixels.
*Exemple :* une image VisDrone de 1920 × 1080 donne 8 tuiles ; une de 2000 × 1500 en donne 12.
Code : `src/dronetrack/detect/tiling.py`.

## Chevauchement (overlap) et pas (stride)
Les tuiles voisines se recouvrent (20 % ici) pour qu'un objet coupé par le bord d'une tuile
apparaisse entier dans la tuile d'à côté. Le pas est la distance entre deux tuiles :
640 × (1 − 0,2) = 512 px, donc 128 px partagés. Tout objet de moins de 128 px est vu entier au moins une fois.

## NMS (Non-Maximum Suppression)
Supprimer les doublons : quand plusieurs boîtes de la même catégorie désignent le même objet, on
garde la plus sûre et on jette celles qui la recouvrent trop (seuil d'IoU ou d'IoS). « Non-maximum » :
on supprime tout ce qui n'est pas le maximum local.
*Dans le projet :* sert à recoller les détections des tuiles. Code : `src/dronetrack/detect/merge.py`.

## NMM (Non-Maximum Merging) ou fusion de boîtes
Variante utilisée par SAHI : au lieu de jeter les doublons, la boîte gardée s'agrandit pour les
couvrir tous. Deux moitiés d'un camion coupé par une tuile redonnent le camion entier.

## IoS (Intersection over Smaller)
Surface commune ÷ surface de la plus petite des deux boîtes. Un morceau d'objet situé à l'intérieur
de la boîte entière a une IoS de 1, alors que son IoU peut être faible.
*Exemple :* boîte entière 40 × 20, morceau 12 × 20 au bord : IoU = 240 / 800 = 0,3 mais IoS = 240 / 240 = 1.

## AP small / medium / large (mAP par taille)
La mAP calculée seulement sur les objets d'une taille donnée (convention COCO : < 32², < 96², au-delà).
Les objets réels d'une autre taille sont « ignorés » : ils ne comptent pas comme manqués, et une
détection posée dessus ne compte ni comme bonne réponse ni comme fausse alerte. Une détection hors
taille qui ne recouvre rien est aussi ignorée.
*Dans le projet :* permet de mesurer le gain des tuiles sur les petits objets sans qu'il soit noyé
dans la moyenne. Code : `area_range` dans `src/dronetrack/eval/det_metrics.py`.

## Suivi multi-objets (MOT, Multi-Object Tracking)
Suivre plusieurs objets dans une vidéo en donnant à chacun un numéro (identité) qu'il garde d'une
image à l'autre. La détection dit « il y a une voiture ici » ; le suivi dit « c'est la voiture n° 7 ».
*Dans le projet :* étape 4, évaluée sur les 7 vidéos de validation VisDrone-MOT (2 846 images).

## Trajectoire (track) et identité
La suite des positions d'un même objet au fil des images, avec son numéro. Dans les annotations
VisDrone-MOT, chaque ligne donne l'image, le numéro de l'objet et sa boîte.
*Dans le projet :* l'objet `Track` = une `Detection` + `track_id`. Code de lecture :
`parse_mot_annotation` dans `src/dronetrack/data/visdrone.py`.

## Filtre de Kalman
Méthode pour suivre une grandeur bruitée qui évolue dans le temps. À chaque image : **prédire**
(« elle avançait de 5 px par image, elle devrait être 5 px plus loin ») puis **corriger** avec la
mesure du détecteur, en faisant plus confiance à celle des deux qui est la moins incertaine.
*Dans le projet :* une boîte par trajectoire, état = centre, taille et leurs vitesses (8 nombres),
hypothèse de vitesse constante. Sans mesure (objet caché sous un arbre), il continue de prédire.
Code : `src/dronetrack/track/kalman.py`.

## État, incertitude et gain de Kalman
L'**état** : ce que le filtre croit (position, taille, vitesses). L'**incertitude** (matrice P) :
à quel point il en est sûr ; elle grandit à chaque prédiction et diminue à chaque mesure.
Le **gain** K : la part de la mesure qu'on accepte (proche de 1 si la prédiction est très
incertaine, proche de 0 si le détecteur l'est).

## Problème d'affectation et algorithme hongrois
Relier deux listes (trajectoires et détections) une à une, de façon à minimiser le coût total
(ici coût = 1 − IoU entre la boîte prédite et la boîte détectée). L'algorithme hongrois trouve la
meilleure répartition pour tous à la fois, là où « chacun prend la plus proche » peut laisser
quelqu'un sans rien.
*Exemple (test `test_hungarian_beats_greedy`) :* la trajectoire 0 préfère la détection 0 (IoU 0,54),
mais si elle la prend, la trajectoire 1 n'a plus rien d'utilisable. Le hongrois donne la détection 1
à la trajectoire 0 (IoU 0,43) et la détection 0 à la trajectoire 1 (IoU 0,54) : deux associations au lieu d'une.
Code : `src/dronetrack/track/matching.py` (avec `scipy.optimize.linear_sum_assignment`).

## ByteTrack et l'association en deux passes
Méthode de suivi (2022) dont l'idée clé est de ne pas jeter les détections peu sûres. Passe 1 :
les détections sûres (score ≥ 0,25) sont reliées aux trajectoires. Passe 2 : les détections
hésitantes (0,1 à 0,25) servent seulement à prolonger les trajectoires restées seules, typiquement
un objet à moitié caché ; elles ne créent jamais de nouvelle trajectoire.
*Dans le projet :* `src/dronetrack/track/byte_tracker.py`, réglages identiques à Ultralytics.

## Trajectoire à confirmer / perdue
Une nouvelle trajectoire doit être revue à l'image suivante pour recevoir un numéro (sinon c'était
une fausse alerte d'une image). Une trajectoire qui n'est plus vue devient « perdue » et reste en
mémoire 30 images, au cas où l'objet réapparaît ; après, elle est oubliée.

## Changement d'identité (ID switch)
Quand un même objet réel change de numéro en cours de vidéo (après une occultation, un croisement,
un mouvement brusque du drone). C'est l'erreur typique d'un tracker.

## MOTA (Multiple Object Tracking Accuracy) et CLEAR MOT
Note globale du suivi : MOTA = 1 − (objets ratés + fausses alertes + changements de numéro) ÷ objets réels.
1 = parfait ; peut devenir négatif si on fait plus d'erreurs qu'il n'y a d'objets. Calcul image par
image (protocole CLEAR MOT) : on garde d'abord les associations de l'image précédente encore valables,
puis on associe le reste par l'algorithme hongrois (IoU ≥ 0,5).
*Limite :* MOTA est dominée par la détection (ratés et fausses alertes) ; un changement de numéro
ne compte qu'une fois, même si l'objet garde ensuite le mauvais numéro toute la vidéo.

## IDF1
Part des objets qui portent le bon numéro sur toute la vidéo. On associe une fois pour toutes
chaque objet réel à au plus un numéro prédit (celui avec lequel il partage le plus d'images), puis
IDF1 = 2 × (images où l'objet a ce numéro) ÷ (objets réels + prédictions). Contrairement à MOTA, un
objet qui change de numéro en milieu de vidéo perd la moitié de sa note.
Code : `src/dronetrack/eval/mot_metrics.py`.
