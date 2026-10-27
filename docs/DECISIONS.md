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
limite max_det 300 vs 317 (0), regroupement rectangulaire des images (0).

**Cause principale trouvée (plan 2, tâche 1) : le « multi-label » de la validation.** `val()` appelle
la suppression des doublons avec `multi_label=True` : une même boîte peut sortir plusieurs fois, avec
plusieurs catégories (26 % des boîtes de `val()` sont de telles copies). Une boîte hésitante entre
voiture et camionnette compte alors comme candidate dans les deux catégories, ce qui augmente la mAP.
`predict()` garde une seule catégorie par boîte. En ne gardant que la catégorie la plus sûre de chaque
boîte de `val()`, sa mAP@50 passe de 0,298 à 0,284, à 0,011 de notre 0,273 ; le reste vient de détails
de prétraitement (interpolation, letterbox).
**Décision : on garde une catégorie par boîte.** Le multi-label est une astuce d'évaluation : un
système réel (et notre tracker) attribue une seule catégorie à chaque objet. Nos chiffres sont donc un
peu plus bas que ceux qu'affiche Ultralytics, mais ils mesurent ce que le système fait vraiment.
La première ligne de l'expérience 2 (image entière, 10 catégories) est mAP@50 = 0,273.

## D7 — Tuiles 640 px, chevauchement 0,2, + image entière, doublons supprimés par IoU (sans fusion)
Le plan prévoyait la fusion de boîtes par IoS, la méthode par défaut de SAHI. L'expérience 2 l'a
réfutée sur VisDrone : mAP@50 0,248, sous la référence sans tuiles (0,273), alors que la suppression
classique par IoU donne 0,374. Les variantes E et F isolent la cause : c'est la fusion, pas l'IoS.
Dans des scènes denses, la fusion réunit des objets voisins de même catégorie en une seule boîte.
Le passage sur l'image entière est gardé : sans lui, les grands objets chutent (0,580 → 0,412),
car un objet de plus de 128 px peut être coupé dans toutes les tuiles.
Coût : environ 10 fois plus lent (≈ 60 → 6,4 images/s sur le Mac M1). À utiliser quand les petits
objets comptent plus que la vitesse ; pour du temps réel embarqué, l'image entière reste l'option rapide.
Hors périmètre, à tester si besoin (plan 2b) : ré-entraîner sur des tuiles, car le modèle a appris
sur des images réduites et voit les objets plus gros dans une tuile qu'à l'entraînement.

## D8 — Tracker : réglages d'Ultralytics, association par catégorie, fuse_score activé
Réglages copiés du `bytetrack.yaml` d'Ultralytics (seuils 0,25 / 0,1, nouvelle trajectoire 0,25,
mémoire 30 images, passes à IoU ≥ 0,2 et 0,5, confirmation à IoU ≥ 0,3) pour comparer les algorithmes,
pas les réglages. Le plan prévoyait de ne pas pondérer le recouvrement par le score (« fuse_score ») ;
l'expérience 3 a montré que c'était tout l'écart avec Ultralytics (MOTA 0,070 contre 0,177). Il est
activé par défaut (passe 1 et confirmation, pas la passe 2, comme Ultralytics) et reste désactivable.
Différences gardées : association seulement entre objets de même catégorie ; numéro donné à la
confirmation (les fausses alertes d'une image ne consomment pas de numéro).
Vérification des métriques : TrackEval, lancé sur nos sorties ; il utilise des alias numpy supprimés
(`np.int`, `np.float`) qu'on recrée seulement pendant la vérification.
Le détecteur pour le suivi reste l'image entière : les tuiles dégradent le suivi au seuil 0,1 (voir RESULTATS).

## D9 — Démo en ligne : Hugging Face Space gratuit (CPU), 300 images au plus, image entière
Matériel gratuit « CPU basic » (2 cœurs, pas de GPU) plutôt qu'un GPU payant : YOLO26n est assez petit
pour l'inférence sur CPU (≈ 14 images/s sur 2 cœurs du M1). Limite de 300 images, réduites à 1280 px de
large, pour qu'un visiteur attende environ une minute. Réglages de l'expérience 3 (image entière, seuil
0,1, notre tracker) ; pas de tuiles, 10 fois plus lentes et moins bonnes pour le suivi (D8).
Sortie en H.264 (seul format que lisent tous les navigateurs). Le Space ne contient que la page
(`space/`) : il installe `dronetrack` depuis GitHub au tag `v1.0` et télécharge les poids depuis la
release `v1.0`, pour qu'il n'existe qu'une version du code et des poids.
