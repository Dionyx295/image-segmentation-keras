# Guide développeur 
De manière général les fichiers que j'ai le plus dû modifier sont skyeye_segmentation\controller\main_window.py et skyeye_segmentation\controller\skyeye_func.py.
- main_window.py permet de définir les actions de chaque élemnts de l'interface (boutons, inputs...)
- skyeye_func.py contient toutes les fonctions s'éxécutant de manière parrallèle (apprentissage, préditcions...).

## Modification de l’interface utilisateur 
L’interface est créée en Qt, le fichier .ui regroupant toute l’interface se situe à l’adresse suivante \skyeye_segmentation\view\main_window.ui. 
Personnellement j’ai modifié cette interface avec Qt Creator (déplacements et ajouts de widgets). 
Ce fichier .ui peut ensuite être traduit en python avec la commande “pyuic5 main_window.ui > main_window.py” (en ouvrant un terminal là où se situe le fichier .ui). 
Le fichier main_window.py, ainsi créé, est celui qui sera utilisé par skeyeye pour afficher l’interface, on retrouve le controller associé à l’adresse \skyeye_segmentation\controller\main_window.py. 
Ce fichier permet surtout de lier les boutons de l’interface avec le code voulu.
L’outil de création d’interface graphique de Qt creator permet d’utiliser les widgets par défaut de Qt. 
Il arrive cependant qu’on ai besoin de modifier ces objets par défaut. 
Qt repose beaucoup sur le fait de créer des sous classes pour ajouter des fonctionnalités à nos widgets (comme des événements). 
J’ai eu besoin par exemple d’ajouter un event “clicked” sur des images (QGraphicsView), pour cela j’ai utilisé la “promotion” de widget.  
Si je promeus un QGraphicsView dans Qt Creator cela signifie qu’il apparraitra de manière classique dans la visualisation de l’interface mais lorsque l’on lancera l’application, c’est une classe fille qui sera utilisée pour initialiser l’objet. 
C’est pour ça que le fichier \skyeye_segmentation\view \clickablegraphicsview.py existe. 
Il contient juste une classe qui hérite de QGraphicsView et ajoute simplement un event “clicked”. 
On peut ensuite connecter cet événement avec l’action souhaitée dans le controller main_window.py. 

## Sauvegarde des modèles
Après le PRD de Tom Suchel, les modèles étaient sauvegardés à chaque epoch, cela pose problème car cela peut rapidement prendre trop de place en mémoire.
On pourrait penser qu'une solution serait de ne sauvegardé que le modèle avec la meilleure précision mais dans notre cas cela n'est pas possible. 
En effet, les charbonnières ne répresente que moins d'1% des pixels de l'image.
Cela signifie qu'on peut très bien avoir un modèle avec 99% de précision mais qui prédit pourtant que du fond.
Le code pour choisir comment sont sauvegardés les models se situe dans le fichier skyeye_segmentation\controller\skyeye_func.py : l.426 quand aucun jeu d'évaluation n'est renseigné, l.456 quand on utilise un jeu d'évaluation.

## Ajout de poids
L’ajout de poid se fait directement dans le traitement des images, quand on associe un label à un pixel on peut aussi associer un poid à ce pixel (poid qu’on peut déterminer en fonction du label). 
Pour l’instant l’activation et la valeur de ces poids est définie directement dans le code (paramètre de la fonction image_segmentation_generator de keras_segmentation/ data_utils/data_loader.py, utilisé dans skyeye_segmentation/controller/skyeye_func.py : val_gen / train_gen).
