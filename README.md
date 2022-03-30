# Outil de segmentation d'image en Deep Learning : SkyEye :small_airplane::eye::earth_africa:


Ce projet est un *fork* du projet https://github.com/OakenMan/image-segmentation-keras, lui même étant un fork du projet https://github.com/Millasta/image-segmentation-keras, lui-même étant un fork du projet https://github.com/divamgupta/image-segmentation-keras : *Implémentation de Segnet, FCN, UNet, PSPNet et d'autres modèles avec Keras.*

Le prolongement de ce projet a été réalisé par Jean-Malo dans le cadre du **P**rojet **R**echerche et **D**éveloppement (**P**rojet de **F**in d'**É**tude) de la troisième année du cycle ingénieur au département informatique de l'école Polytech Tours, lors de l'année 2021-22. 

L'objectif de ce projet est de rajouter un ensemble de fonctionnalités à l'outil SkyEye, permettant la segmentation automatique des charbonnières sur des images LiDAR.
Le projet initial a été développé par Valentin MAURICE lors de son PRD sur le même sujet, réalisé en 2019-20, puis Tom Suchel a repirs le projet, aussi lors de son PRD, en 2020-21.

Mon role était principalement d'améliorer les résultats de prédictions, qui ne sont toujours pas suffisant pour une réelle utilisation de l'application.

Les differentes modification que j'ai rélaisé sur l'application peuevnt être visibles dans la liste de commit.

J'ai choisi de tester une architecture s'éloigant un peu de la segmentation d'image pour résoudre le problème de reconnaissance des charbonnières : le Faster R-CNN, qui consiste à faire de la détection d'objets. Cette architecture étant assez complexe j'ai finalement principalement travaillé sur la première partie du réseau : le Region Proposal Network que j'ai testé en dehors de l'application SkyEye (https://github.com/Dionyx295/rpn_helpers). En effet, les données d'entrées sont différentes de ce que prévoit pour l'insatnt SkyeEye, j'ai donc préféré tester l'architecture à part en me disant que l'intégration dans SkyEye n'aurait de sens que si les résultats du RPN sont concluants.


## Prérequis

Le projet a été développé avec une distribution Python 3.6.5 (64bit) sur windows, et nécessite les dépendances suivantes (à retrouver dans requirements.txt) :

- **tensorflow==1.9.0**
- **keras==2.2.5**
- **protobuf==3.6.0**
- six (1.14.0)
- pillow (7.0.0)
- matplotlib (3.1.3)
- scikit-image (0.16.2)
- numpy (1.18.1)
- h5py (2.10.0)
- tqdm (4.43.0)
- pyqt5 (5.14.1)
- opencv-python (4.2.0.32)
- imgaug (0.4.0)
- sklearn (0.0)

**Attention**: Qt 5.15 doit être installé sur l'ordinateur afin de pouvoir modifier l'interface utilisateur.

## Lancement

Il suffit d'éxécuter en ligne de commande ```python entrypoint.py```.


## Structure du projet

Le projet est structué de la manière suivante :

- **html/** : contient la documentation générée avec [pdoc](https://pdoc3.github.io/pdoc/) 
- **keras_segmentation/** : sources du projet initial forké
- **log/** : fichiers logs
- **README/** : ressources de ce document
- **skyeye_segmentation/** : source du projet
- **test_keras_segmentation/** : tests unitaires du projet forké
- **test_syeye_segmentation/** : tests unitaires du projet



## Documentation

La documentation disponible dans *html/* a été générée avec [pdoc](https://pdoc3.github.io/pdoc/) : 

```python
>>> pdoc skyeye_segmentation --html
```



## Tests unitaires

Pour lancer les tests unitaires il suffit de se placer dans un des dossier test_... et de taper dans un terminal ```python -m pytest```.
Le test_charb_segmentation serait à revoir, trop d'images sont générées et cela pollue le repertoire de travail.


## Qualimétrie

Pour lancer une analyse statique du code avec [PyLint](https://www.pylint.org/) (fichier de configureation : *.pylintrc*) :

```python
>>> pylint --rcfile=.pylintrc skyeye_segmentation > pylint.txt
```

Le rapport PyLint est alors disponible et donne des indications sur la qualité du code analysé, ainsi qu'une note générale.



## Manuel d'utilisation de la partie réalisée par Tom Suchel

Il a créé une rachitecture correspondant à un VGG de taille réduite. Le travail de segmentation est devenu un travail de prédiction (il découpe les images en patch de petite taille et donne ces patch à son réseau qui predit si oui ou non le patch contient une charbonnière). Les résultats ne sont pas concluants, il faudrait surement plus d'image d'apprentissage.
Voir le [manuel d'utilisation](MANUAL/Manuel.md)


