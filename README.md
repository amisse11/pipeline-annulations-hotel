# Projet pipeline : prédire les annulations de réservations d'hôtel

Projet de groupe du cours 030513 IFM, Intelligence artificielle appliquée au prétraitement des données (Collège La Cité, automne 2026).

Évaluation sommative 2 : création des pipelines.

## Équipe

- Akiella Wankoua, Blonde Béatrice
- Essomba, Martin Junior
- Gouater, Rene Eutyche
- Misse, Anthony

## Objectif

Prédire si une réservation d'hôtel sera annulée (variable `is_canceled`) avec un pipeline scikit-learn complet :

- prétraitement selon le type de variable ;
- transformateurs personnalisés ;
- sélection des caractéristiques ;
- modèle de classification.

## Données

- Fichier : `donnees/brutes/hotels.csv` (119 390 réservations, 32 colonnes).
- Contenu : réservations de deux hôtels au Portugal, de 2015 à 2017.
- Source : Antonio, N., de Almeida, A. et Nunes, L. (2019). Hotel booking demand datasets. *Data in Brief*, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126
- Version utilisée : celle du projet TidyTuesday (https://github.com/rfordatascience/tidytuesday/tree/master/data/2020/2020-02-11).

## Organisation du dossier

- `notebooks/` : les 5 notebooks, à exécuter dans l'ordre
    - `01_exploration_preparation.ipynb` : exploration, nettoyage et séparation train / test
    - `02_pipelines_pretraitement.ipynb` : pipelines par type de variable et ColumnTransformer
    - `03_transformateur_personnalise.ipynb` : classes `CreationVariables` et `PlafonnementValeurs`
    - `04_selection_caracteristiques.ipynb` : sélection avec `SelectFromModel` et une forêt aléatoire
    - `05_modele_evaluation.ipynb` : choix du modèle, entraînement et évaluation sur le test
    - `transformateurs.py` : classes écrites par le notebook 03
- `donnees/brutes/` : données de départ
- `figures/` : graphiques produits par les notebooks
- `archives/` : premier notebook de René (parties 1 et 2), point de départ des notebooks 01 et 02

Les dossiers `donnees/preparees/` et `modeles/` sont créés pendant l'exécution. Ils ne sont pas dans le dépôt.

## Lien entre les notebooks

Chaque notebook lit les fichiers créés par le précédent :

- le notebook 01 écrit les fichiers train et test dans `donnees/preparees/` ;
- le notebook 02 écrit `modeles/etape2_pretraitement.joblib` ;
- le notebook 03 écrit `notebooks/transformateurs.py` et `modeles/etape3_pipeline_pretraitement.joblib` ;
- le notebook 04 écrit `modeles/etape4_pipeline_selection.joblib` ;
- le notebook 05 entraîne le modèle final et l'évalue sur le test.

## Installation et exécution

1. Installer les librairies : `pip install -r requirements.txt`
2. Ouvrir le dossier du projet dans VS Code ou Jupyter.
3. Exécuter les notebooks dans l'ordre, de 01 à 05 (« Exécuter tout » dans chaque notebook).

L'exécution complète prend environ 2 minutes. Les résultats affichés ont été obtenus avec Python 3.11, pandas 1.5.3 et scikit-learn 1.9.1.

## Principales décisions

- Suppression des doublons (31 994 lignes) : sans numéro de réservation, une même ligne pouvait se retrouver dans le train et dans le test.
- Retrait de 3 colonnes pour fuite de données :
    - `reservation_status` et `reservation_status_date` donnent le résultat ;
    - `assigned_room_type` est souvent connue seulement à l'arrivée du client.
- Deux transformateurs personnalisés :
    - `CreationVariables` crée `total_nuits`, `total_clients` et `prix_par_personne` ;
    - `PlafonnementValeurs` limite les valeurs extrêmes au 99e centile appris sur le train.
- Sélection : `SelectFromModel` avec une forêt aléatoire et le seuil « médiane » (54 colonnes gardées sur 108).
- Modèle : forêt aléatoire avec `class_weight="balanced"`, choisie contre la régression logistique par validation croisée.

## Résultats sur le test

- F1 : 0,743
- Rappel des annulations : 0,822
- Précision des annulations : 0,677
- ROC AUC : 0,918
