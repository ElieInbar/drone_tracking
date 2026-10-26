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

## D5 — Ré-entraînement sur Modal : 50 epochs, 640 px, batch 16, GPU L4
Prévu sur Kaggle, fait sur Modal : le compte Modal était déjà prêt, et Modal se pilote depuis le
terminal (pas de notebook à lancer à la main). Les images brutes et les entraînements sont gardés
dans un volume Modal : un entraînement coupé peut reprendre depuis `last.pt`.
50 epochs : la courbe de validation est plate dès l'epoch 33, donc plus d'epochs n'aurait servi à rien
pour ce modèle. 640 px : la taille standard de YOLO ; les petits objets seront traités par les tuiles
(étape 3), pas par une image d'entrée plus grande.

## D6 — Vérification de notre mAP contre Ultralytics
Sur le même modèle, `val()` d'Ultralytics donne mAP@50 = 0,296 et notre script 0,273 (écart 0,024,
au-delà de la tolérance de 0,01). Recherche de la cause :
- mêmes prédictions (celles de notre pipeline) notées par la formule d'Ultralytics : 0,270, par la
  nôtre : 0,273 → les deux formules sont d'accord ;
- prédictions exportées par `val()` (JSON) notées par notre formule : 0,298, contre 0,296 pour
  Ultralytics → écart 0,002 : **notre mAP est validée**.
L'écart vient donc des prédictions : `predict()` et `val()` d'Ultralytics ne produisent pas
exactement les mêmes boîtes. Causes testées et écartées : interpolation de redimensionnement (+0,002),
limite max_det 300 vs 317 (0), regroupement rectangulaire des images (0). Question ouverte, à reprendre
au plan 2 (inférence par tuiles). L'expérience 1 reste équitable : les deux modèles passent par le
même pipeline.
