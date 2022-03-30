# Installation de SkyEye sur une nouvelle machine
Ce guide détails toutes les étapes pour lancer l'application sur une nouvelle machine.

## Récupération des sources
### git  
- https://git-scm.com/download/win
- rien à changer dans l’installation
- clic droit dans le dossier voulu, git bash (ouvre un terminal)
- ```git clone https://github.com/Dionyx295/image-segmentation-keras.git```
- désavantages des autres support type clé USB : mise à jour plus difficile (pas de git pull)

## Installation de l’environnement python
### Installation python (3.6)
- https://www.python.org/downloads/release/python-368/
- lancer l’exécutable (clic add to path) pas besoin de custom install
  
### Créer un environnement virtuel
- ```python -m venv skyeye_venv``` (pour créer l’environnement virtuel “skyeye_venv” dans le dossier courant)
 
### Installation des librairies
- Activer l’environnement virtuel (dépend du terminal utilisé : cas de powershell)
- se placer dans le dossier contenant le venv skyeye_venv
- Ctrl Maj ClicDroit pour ouvrir un terminal powershell (attention commandes dépendent du terminal)
- ```set-executionpolicy -Scope CurrentUser RemoteSigned```
- ```skyeye_venv/Scripts/activate.ps1```
- (skyey_venv) doit apparaître au début de la ligne  
- Se placer dans le dossier image-segmentation-keras (pour avoir accès au fichier requirements.txt) en ayant toujours le venv d’activé (cd chemin d’accès)
- ```upgrade pip : python -m pip install --upgrade pip```
- ```pip install -r requirements.txt```
    
(may need to install : https://visualstudio.microsoft.com/visual-cpp-build-tools/ pendant l’installation cocher developpement desktop en C++, nécessite de redémarer l’ordinateur)

## Lancement de l’application

```python entrypoints.py``` (à la racine des sources)
