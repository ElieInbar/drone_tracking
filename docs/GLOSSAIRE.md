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
