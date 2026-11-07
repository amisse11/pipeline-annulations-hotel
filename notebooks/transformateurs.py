"""
Transformateurs personnalisés du projet (Partie 3).
Ce fichier est créé par le notebook 03. Les notebooks 04 et 05 l'importent.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CreationVariables(BaseEstimator, TransformerMixin):
    """Crée 3 nouvelles variables à partir des colonnes existantes.

    - total_nuits = nuits de week-end + nuits de semaine
    - total_clients = adultes + enfants + bébés
    - prix_par_personne = adr / total_clients
    """

    nouvelles_colonnes = ["total_nuits", "total_clients", "prix_par_personne"]

    def fit(self, X, y=None):
        # Rien à apprendre : on garde seulement le nom des colonnes d'entrée
        self.colonnes_entree_ = list(X.columns)
        return self

    def transform(self, X):
        # On travaille sur une copie pour ne pas modifier le tableau d'origine
        X = X.copy()
        X["total_nuits"] = X["stays_in_weekend_nights"] + X["stays_in_week_nights"]
        X["total_clients"] = X["adults"] + X["children"] + X["babies"]
        # replace(0, 1) évite une division par 0
        X["prix_par_personne"] = X["adr"] / X["total_clients"].replace(0, 1)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.colonnes_entree_ + self.nouvelles_colonnes, dtype=object)


class PlafonnementValeurs(BaseEstimator, TransformerMixin):
    """Limite les valeurs extrêmes de certaines colonnes.

    - fit() : apprend la limite haute (un quantile) de chaque colonne sur le train.
    - transform() : remplace les valeurs au-dessus de la limite par la limite.
    """

    def __init__(self, colonnes, quantile=0.99):
        self.colonnes = colonnes
        self.quantile = quantile

    def fit(self, X, y=None):
        self.colonnes_entree_ = list(X.columns)
        # Limite de chaque colonne, calculée sur les données d'entraînement seulement
        self.limites_ = X[self.colonnes].quantile(self.quantile)
        return self

    def transform(self, X):
        X = X.copy()
        for colonne in self.colonnes:
            X[colonne] = X[colonne].clip(upper=self.limites_[colonne])
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.colonnes_entree_, dtype=object)
