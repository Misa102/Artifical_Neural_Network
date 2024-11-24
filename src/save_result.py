import os
import matplotlib.pyplot as plt
from display_figure import *  # Importation de la fonction de traçage


# Créer un répertoire pour sauvegarder les résultats
def creer_repertoire(save_dir):
    """
    Crée le répertoire de sauvegarde s'il n'existe pas.
    :param save_dir: Chemin du répertoire de sauvegarde
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)  # Crée le répertoire
        print(f"Répertoire créé : {save_dir}")
    else:
        print(f"Répertoire existant : {save_dir}")


def sauvegarder_donnees_et_decision(x, yd, w, title, save_dir, file_name):
    """
    Sauvegarde les données et la frontière de décision.
    :param x: Matrice des données d'entrée (2 x n_samples)
    :param yd: Vecteur des classes
    :param w: Vecteur des poids
    :param title: Titre du graphique
    :param save_dir: Répertoire où enregistrer le graphique
    :param file_name: Nom du fichier à sauvegarder
    """
    # Appeler la fonction pour afficher les données et la frontière de décision
    afficher_apprentissage(x, yd, w, title)

    # Sauvegarder le graphique
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    print(f"Graphique enregistré sous : {save_path}")