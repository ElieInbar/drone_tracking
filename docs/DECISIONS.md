# Décisions techniques

Chaque choix et sa raison. Utile pour répondre à « pourquoi as-tu fait comme ça ? » en entretien.

## D1 — Boîtes au format x1, y1, x2, y2 en pixels, partout
VisDrone donne (gauche, haut, largeur, hauteur) et YOLO (centre x, centre y, largeur, hauteur) normalisés.
On convertit une seule fois à l'entrée et à la sortie ; à l'intérieur du code, un seul format.
Les deux coins rendent le calcul de l'IoU direct (min/max des coins).

## D2 — Détecteur YOLO26 nano
Dernière génération de YOLO d'Ultralytics (octobre 2026). La version nano est la plus petite :
elle s'entraîne vite sur le GPU gratuit de Kaggle et tourne sur un Mac M1. Si la précision est trop
faible, on passera à la version small.

## D3 — Correspondance COCO → VisDrone pour le modèle « avant »
Le modèle pré-entraîné parle COCO (person, car…), VisDrone a ses propres catégories. On traduit
les 6 catégories communes : person→pedestrian, bicycle→bicycle, car→car, motorcycle→motor,
bus→bus, truck→truck. Les autres sorties COCO (oiseau, chien…) sont jetées.
Limite connue : COCO ne distingue pas « pedestrian » (debout) de « people » (assis, accroupi) ;
toutes les personnes deviennent « pedestrian ». C'est pourquoi l'expérience 1 compare les modèles
seulement sur ces 6 catégories.

## D4 — mAP calculée à la manière de COCO, pas d'Ultralytics
COCO lit la précision à 101 niveaux de rappel et fait la moyenne. Ultralytics intègre la même
courbe par la méthode des trapèzes, ce qui donne 0,995 au lieu de 1 pour des prédictions parfaites
(défaut connu). On garde la méthode COCO, plus simple à expliquer et exacte sur les cas faits à la
main. Conséquence : nos chiffres peuvent s'écarter légèrement de ceux d'Ultralytics (≈ 0,005) ;
la vérification de la tâche 7 tolère 0,01.
