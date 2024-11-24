import numpy as np

'''
Charger les données pour tester l'apprentissage perceptron de la partie 1.2
'''
# Chargement des données
def charger_donnees(file_path):
    """
    Charge les données d'un fichier texte.
    :param file_path: Chemin du fichier
    :return: Matrice des données (2 x n_samples), vecteur des classes
    """
    data = np.loadtxt(file_path)
    x = data[:2, :]  # Les deux premières lignes sont les caractéristiques
    yd = np.array([1] * 25 + [-1] * 25)  # Sorties désirées
    return x, yd