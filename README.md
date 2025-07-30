# Deep Learning

## Cloner le dépôt
Pour cloner ce dépôt, exécutez la commande suivante :

```bash 
git clone https://github.com/SwerkF/deep_learning_gym.git
```

## Installation

Créez un environnement virtuel Python et activez-le. Par exemple, avec `venv` :

```bash
python -m venv venv
```

### Windows
Pour activer l'environnement virtuel sur Windows, exécutez :

```bash
venv\Scripts\activate
```

### Linux/MacOS
Pour activer l'environnement virtuel sur Linux ou MacOS, exécutez :

```bash 
source venv/bin/activate
```

Pour installer les dépendances, exécutez la commande suivante :

```bash
pip install -r requirements.txt
```

## Entrainement

### CartPole-V1

Pour entraîner le modèle sur l'environnement CartPole-V1, ouvrez le fichier `cartpole_v1_oliwer_mohand.ipynb` et exécutez les cellules.

### LunarLander-v3

Pour entraîner le modèle sur l'environnement LunarLander-v3, ouvrez le fichier `lunar_lander_v3_oliwer_mohand.ipynb` et exécutez les cellules.

Les modèles se sauvegardent automatiquement dans le dossier `models/`. Vous pouvez les charger pour effectuer des prédictions ou pour continuer l'entraînement.

## GUI

### CartPole-V1

Pour lancer l'interface graphique pour l'environnement CartPole-V1, exécutez la commande suivante :

```bash
python gui_cartpole.py
```

### LunarLander-v3  
Pour lancer l'interface graphique pour l'environnement LunarLander-v3, exécutez la commande suivante :

```bash
python gui_lander.py
```