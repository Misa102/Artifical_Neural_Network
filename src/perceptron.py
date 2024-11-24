import numpy as np
from display_figure import afficher_apprentissage

# Définition de la fonction perceptron_simple
def perceptron_simple(x, w, active):
    '''
    Fonction de perceptron simple avec fonction d'activation
    :param x: inputs
    :param w: poids
    :param active: fonction d'activation (0: sign, 1: tanh, 2: sin)
    :return: la sortie du perceptron simple
    '''
    # Calcul de la combinaison linéaire y = w[0] + w[1]*x[0] + w[2]*x[1]
    y = w[0] + np.dot(w[1:], x)
    # Application de la fonction d'activation
    if active == 0:  # signe
        return np.sign(y)
    elif active == 1:  # tanh
        return np.tanh(y)
    else:
        raise ValueError("Fonction d'activation non supportée")


# Fonction d'apprentissage Widrow-Hoff
def apprentissage_widrow(x, yd, epoch, batch_size):
    """
    Algorithme d'apprentissage Widrow-Hoff
    :param x: Matrice des données d'entrée (2 x n_samples)
    :param yd: Vecteur des sorties désirées (1 x n_samples)
    :param epoch: Nombre maximal d'itérations
    :param batch_size: Taille des batches pour l'apprentissage
    :return: Poids finaux et historique des erreurs par itération
    """
    n, m = x.shape  # n: dimensions des caractéristiques, m: nombre d'échantillons
    w = np.random.randn(3)  # Poids initiaux aléatoires (inclut le biais)
    alpha = 0.1  # Taux d'apprentissage
    erreurs = []  # Historique des erreurs cumulées par époque

    # Ajouter un biais à la matrice d'entrée
    x_bias = np.vstack((np.ones((1, m)), x))  # Ajout d'une ligne de 1

    for ep in range(epoch):
        err_epoch = 0  # Erreur cumulée pour cette époque

        # Itération sur les batches
        for j in range(0, m, batch_size):
            batch_x = x_bias[:, j:j + batch_size]
            batch_yd = yd[j:j + batch_size]

            # Calcul des sorties pour le batch courant
            y = np.tanh(w @ batch_x)

            # Mise à jour des poids pour chaque échantillon
            for k in range(batch_x.shape[1]):
                erreur_signal = batch_yd[k] - y[k]
                phi_prime = 1 - y[k] ** 2  # Dérivée de tanh
                gradient = -erreur_signal * phi_prime * batch_x[:, k]
                w -= alpha * gradient  # Mise à jour des poids

                # Accumulation de l'erreur quadratique
                err_epoch += erreur_signal ** 2

        # Enregistrer l'erreur de classification de l'époque
        erreurs.append(err_epoch)

        # Affichage des résultats pour cette époque
        print(f"Époque {ep + 1}: Erreur cumulée = {err_epoch}")
        afficher_apprentissage(x, yd, w, f"Époque {ep + 1}")

        # Arrêt si l'erreur est nulle
        if err_epoch == 0:
            print("Erreur nulle atteinte, apprentissage terminé.")
            break

    return w, erreurs
