import matplotlib.pyplot as plt
import numpy as np
import os

'''
Perceptron_simple
'''

# Fonction pour tracer et sauvegarder
def afficher_perceptron_simple(x_inputs, y_desired, w, activation, save_dir):
    '''
    Trace les points de données et la frontière de décision
    :param x_inputs: Les entrées de données
    :param y_desired: Les étiquettes cibles
    :param w: Les poids du perceptron
    :param activation: Type de fonction d'activation (0: sign, 1: tanh, 2: sin)
    :param save_dir: Répertoire où enregistrer l'image
    '''
    # Définir le titre et l'étiquette en fonction de l'activation
    activation_map = {0: "Sign", 1: "Tanh"}
    activation_name = activation_map.get(activation, "Unknown")

    plt.figure(figsize=(8, 6))
    plt.scatter(x_inputs[y_desired == 1][:, 0], x_inputs[y_desired == 1][:, 1], color='blue', label='Classe 1 (1)')
    plt.scatter(x_inputs[y_desired == 0][:, 0], x_inputs[y_desired == 0][:, 1], color='red', label='Classe 2 (0)')

    # Calculer la frontière de décision
    x_boundary = np.linspace(0, 1)
    y_boundary = -(w[1] / w[2]) * x_boundary - (w[0] / w[2])
    plt.plot(x_boundary, y_boundary, color='green', label='Frontière de décision')

    # Configurer le graphique
    plt.title(f"Visualisation de l'opération OU logique (Activation: {activation_name})")
    plt.xlabel("Entrée x1")
    plt.ylabel("Entrée x2")
    plt.grid()
    plt.legend()

    # Sauvegarder l'image
    save_path = os.path.join(save_dir, f"OU_logic_activation_{activation_name}.png")
    plt.savefig(save_path)
    print(f"Graphique enregistré sous: {save_path}")

    plt.show()
    plt.close()

'''
Etude apprentissage Widrow-hoff perceptron 
'''
# Fonction pour afficher les données et la frontière de décision pour l'apprentissage et sauvegarder
def afficher_apprentissage(x, yd, w, title="", save_dir="../results/perceptron_learning"):
    """
    Affiche les données et la frontière de décision, puis sauvegarde le graphique dans un répertoire.
    :param x: Matrice des données d'entrée (2 x n_samples)
    :param yd: Vecteur des classes
    :param w: Vecteur des poids
    :param title: Titre du graphique
    :param save_dir: Répertoire où sauvegarder le graphique
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(8, 6))
    plt.scatter(x[0, yd == 1], x[1, yd == 1], color='blue', label='Classe 1 (+1)')
    plt.scatter(x[0, yd == -1], x[1, yd == -1], color='red', label='Classe 2 (-1)')

    # Tracer la frontière de décision
    x_boundary = np.linspace(x[0].min(), x[0].max())
    y_boundary = -(w[1] / w[2]) * x_boundary - (w[0] / w[2])
    plt.plot(x_boundary, y_boundary, color='green', label='Frontière de décision')

    # Configurations
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid()
    plt.legend()

    # Sauvegarder le graphique dans le répertoire
    file_name = f"frontiere_apprentissage_{title}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    print(f"Graphique sauvegardé sous : {save_path}")

    plt.show()
    plt.close()

# Fonction pour afficher l'évolution de l'erreur cumulée et sauvegarder
def afficher_evolution_erreur(erreurs, title="Évolution de l'erreur cumulée", xlabel="Époque", ylabel="Erreur cumulée", save_dir="../results/perceptron_learning"):
    """
    Affiche l'évolution de l'erreur cumulée au fil des époques, puis sauvegarde le graphique dans un répertoire spécifié.
    :param erreurs: Liste ou tableau des erreurs cumulées
    :param title: Titre du graphique
    :param xlabel: Label de l'axe des abscisses
    :param ylabel: Label de l'axe des ordonnées
    :param save_dir: Répertoire où sauvegarder le graphique (répertoire par défaut est spécifié)
    """
    # Créer le répertoire de sauvegarde s'il n'existe pas
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Créer le graphique
    plt.figure(figsize=(8, 6))
    plt.plot(erreurs, marker='o', linestyle='-', color='b')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    # Sauvegarder le graphique dans le répertoire
    file_name = f"erreur_cumulee_{title.replace(' ', '_')}.png"  # Remplacer les espaces par des underscores
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path)
    print(f"Graphique d'erreur cumulée enregistré sous : {save_path}")

    plt.show()
    plt.close()

'''
Etude apprentissage mulyiplayer_percptron
'''
# Fonction sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def affichier_multilayer_perceptron(x, yd, w1, w2, sigmoid, save_dir=None):
    """
    Affiche les données d'apprentissage et les frontières de décision de chaque neurone caché et de la couche de sortie.

    :param x: Matrice des données d'entrée (2 x n_samples)
    :param yd: Vecteur des classes
    :param w1: Poids de la couche cachée (y compris le biais)
    :param w2: Poids de la couche de sortie
    :param sigmoid: Fonction d'activation sigmoid
    :param save_dir: Répertoire où sauvegarder l'image (si fourni)
    """
    # Affichage des données d'apprentissage
    plt.figure(figsize=(8, 5))
    plt.scatter(x[0, yd == 0], x[1, yd == 0], c='r', label='Classe 0', edgecolor='k')
    plt.scatter(x[0, yd == 1], x[1, yd == 1], c='b', label='Classe 1', edgecolor='k')

    # Calcul des frontières de décision pour chaque neurone caché
    for i in range(w1.shape[0] - 1):  # -1 pour ignorer le biais
        a, b = w1[i + 1, 0], w1[i + 1, 1]  # On prend les poids du neurone caché
        intercept = -w1[i, 0] / w1[i, 1]  # Interception
        slope = -a / b  # Pente
        x_vals = np.linspace(-0.5, 1.5, 100)
        y_vals = slope * (x_vals + intercept)  # Ligne de séparation

        plt.plot(x_vals, y_vals, label=f'Décision Neurone caché {i + 1}')

    # Création d'une grille pour le tracé des frontières de décision
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 100), np.linspace(-0.5, 1.5, 100))
    grid = np.c_[xx.ravel(), yy.ravel()].T
    grid_bias = np.vstack([np.ones((1, grid.shape[1])), grid])

    # Calcul de la sortie pour chaque point de la grille
    h_input = np.dot(w1.T, grid_bias)
    h_output = sigmoid(h_input)
    h_output_bias = np.vstack([np.ones((1, h_output.shape[1])), h_output])
    y_input = np.dot(w2, h_output_bias)
    y_output = sigmoid(y_input)

    # Tracer la frontière de décision de la couche de sortie
    zz = y_output.reshape(xx.shape)
    plt.contour(xx, yy, zz, levels=[0.5], colors='k', linestyles='solid', label='Décision de la sortie')

    plt.title("Données d'apprentissage et frontières de décision")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.grid()

    # Sauvegarde de l'image si un répertoire est spécifié
    if save_dir:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = "frontiere_apprentissage_multilayer.png"
        save_path = os.path.join(save_dir, file_name)
        plt.savefig(save_path)
        print(f"Graphique enregistré sous : {save_path}")

    # Affichage du graphique
    plt.show()
    plt.close()