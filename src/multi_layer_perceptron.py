import numpy as np

"Définition de la fonction d'activation sigmoîg"
# Fonction sigmoïde
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Dérivée de la fonction sigmoïde
def der_sigmoid(x):
    sig = sigmoid(x)
    return sig * (1 - sig)

# # test dérivée du sigmoîde
# x = np.array([0, 2, -2])
# print("Sigmoid:", sigmoid(x))
# print("Dérivée de Sigmoid:", der_sigmoid(x))

'''
1.3.1 Mise en place d’un perceptron multicouche
'''

"Définition de la fonction multiperceptron"
def multiperceptron(x, w1, w2):
    # Ajout de seul égale à 1
    x_bias = np.append(x, 1) # x_bias contient (x1,x2,1)

    # calcul l'entrée de la couche cachée
    # La variable w1 contient les poids synaptiques des 2 neurones de la couche cachée.
    # C’est une matrice à 3 lignes et 2 colonnes.
    # w1[:, 0] et w1[:, 1] sont les vecteurs de poids pour les deux neurones de la couche cachée.

    ' entrée du premier neurone caché '
    # w1[:, 0] = [2, -1, -0.5]
    # h1_input est calculé comme x1.2 + x2.(-1) + 1.(-0.5)
    h1_input = np.dot(x_bias, w1[:, 0])

    ' entrée du second neurone caché '
    # w1[:, 1] = [1, 0.5, 0.5]
    # h2_input est calculé comme x1.1 + x2.0.5 + 1.0.5
    h2_input = np.dot(x_bias, w1[:, 1])

    ' application de la fonction d.activation aux entrées des neurones cachées pour obtenir les sorties de ces neurones'
    ' la fonction sigmoîd transforme toute entrée en une valuer comprise entre 0 et 1 '
    # sortie du 1er neurone caché
    h1_output = sigmoid(h1_input)
    # sortie du 2e neurone caché
    h2_output = sigmoid(h2_input)

    ' preparer des sorties de la couche cachée '
    # tableau hidden_outputs servira d'entrée pour la couche de sortie
    hidden_outputs = np.array([h1_output, h2_output, 1])

    ' calculer l.entrée et la sortie de la couche de sortie '
    # y_input : la somme ppondérée des sorties de la couche cachée et du seuil, qui constitue l'entrée du neurone de sortie
    # avec w2 = [-1, 1, 2]
    # y_output = h1_output.(-1) + h2_output.1 + 1.2
    y_input = np.dot(hidden_outputs, w2)
    y_output = sigmoid(y_input)

    return y_output, h1_output, h2_output

'''
1.3.2 Programmation apprentissage multicouches
'''
# Fonction d'apprentissage du perceptron multicouche avec la règle de Widrow-Hoff
def multiperceptron_widrow(x, yd, epoch, batch_size):
    """
    Apprentissage d'un perceptron multicouche avec la règle de Widrow-Hoff.

    Paramètres :
    - x : matrice (2 x n) représentant l'ensemble d'apprentissage.
    - yd : vecteur (1 x n) indiquant la réponse désirée (0 ou 1) pour chaque élément de x.
    - epoch : nombre d'itérations sur l'ensemble d'apprentissage.
    - batch_size : nombre d'individus traités avant mise à jour des poids.
    - learning_rate : taux d'apprentissage (alpha)

    Résultats :
    - w1 : matrice de poids de la couche cachée (3 x 2).
    - w2 : vecteur de poids de la couche de sortie (3 x 1).
    - erreur : vecteur contenant l'erreur cumulée pour chaque itération.
    """
    learning_rate = 0.5

    # Initialisation aléatoire des poids
    np.random.seed(0)
    w1 = np.random.randn(3, 2)  # Poids de la couche cachée
    w2 = np.random.randn(1, 3)  # Poids de la couche de sortie

    # Ajouter le biais à l'ensemble d'apprentissage
    n_samples = x.shape[1]
    x = np.vstack([np.ones((1, n_samples)), x])  # Ajout de la ligne de biais

    # Stockage de l'erreur cumulée
    erreur = []

    # Boucle d'apprentissage
    for ep in range(epoch):
        erreur_cumulee = 0

        for i in range(0, n_samples, batch_size):
            # Extraire le batch
            x_batch = x[:, i:i + batch_size]
            yd_batch = yd[i:i + batch_size]

            # Propagation avant
            h_input = np.dot(w1.T, x_batch)  # Entrées de la couche cachée
            h_output = sigmoid(h_input)  # Sorties de la couche cachée
            h_output = np.vstack([np.ones((1, h_output.shape[1])), h_output])  # Ajouter le biais

            y_input = np.dot(w2, h_output)  # Entrée de la couche de sortie
            y_output = sigmoid(y_input)  # Sortie de la couche de sortie

            # Calcul de l'erreur pour ce batch
            erreurs = yd_batch - y_output
            erreur_cumulee += np.sum(erreurs ** 2)  # Erreur quadratique cumulée

            # Rétropropagation
            # Gradient pour la couche de sortie
            delta_output = erreurs * der_sigmoid(y_input)  # Erreur multipliée par la dérivée de la sortie
            grad_w2 = np.dot(delta_output, h_output.T)  # Gradient pour les poids de sortie

            # Gradient pour la couche cachée
            delta_hidden = np.dot(w2[:, 1:].T, delta_output) * der_sigmoid(
                h_input)  # Backpropagation vers la couche cachée
            grad_w1 = np.dot(delta_hidden, x_batch.T).T

            # Mise à jour des poids
            w2 += learning_rate * grad_w2 / batch_size
            w1 += learning_rate * grad_w1 / batch_size

        # Stocker l'erreur cumulée de cette itération
        erreur.append(erreur_cumulee / n_samples)

        # Affichage de l'erreur par itération
        print(f"Itération {ep + 1}, Erreur cumulée: {erreur_cumulee:.2f}")

        # Arrêt si l'erreur est nulle
        if erreur_cumulee == 0:
            break

    return w1, w2, erreur