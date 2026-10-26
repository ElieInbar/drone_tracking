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
