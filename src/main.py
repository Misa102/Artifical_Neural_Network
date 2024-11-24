from perceptron import *
from save_result import *
from display_figure import afficher_perceptron_simple
from upload_data import charger_donnees
from multi_layer_perceptron import multiperceptron, multiperceptron_widrow


'''
Perceptron simple
'''
# Données entrées OU logique
x_inputs = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
y_desired = np.array([-1, 1, 1, 1])

# Initialiser les poids
w = np.array([-0.5, 1, 1])

# Définir le chemin relatif pour enregistrer les résultats du perceptron de la logique OU dans "results/perceptron"
save_dir = "../results/perceptron"
creer_repertoire(save_dir)

# Tracer avec les différentes fonctions d'activation
for activation in [0, 1]:  # 0: Sign, 1: Tanh
    for x in x_inputs:
        perceptron_simple(x, w, active=activation)
    # Affichage perceptron simple
    afficher_perceptron_simple(x_inputs, y_desired, w, activation, save_dir)


'''
Apprentissage perceptron
'''
# Test 1 : Données p2_d1.txt
x1, yd1 = charger_donnees('../data/p2_d1.txt')
w1, erreurs1 = apprentissage_widrow(x1, yd1, epoch=10, batch_size=5)

# Affichage de l'évolution de l'erreur
afficher_evolution_erreur(erreurs1, title="Évolution de l'erreur cumulée pour p2_d1.txt")

# Test 2 : Données p2_d2.txt
x2, yd2 = charger_donnees('../data/p2_d2.txt')
w2, erreurs2 = apprentissage_widrow(x2, yd2, epoch=10, batch_size=5)

# Affichage de l'évolution de l'erreur
afficher_evolution_erreur(erreurs2, title="Évolution de l'erreur cumulée pour p2_d2.txt")


'''
Perceptron multicouche
'''
# Définition des poids et de l'entrée
w1 = np.array([
    [2, 1],  # poids de x1 aux neurones cachés
    [-1, 0.5],  # poids de x2 aux neurones cachés
    [-0.5, 0.5]  # seuil égale à 1
])

# w2 poids pour la couche de sortie
w2 = np.array([-1, 1, 2])

# Définir l'entrée
x = np.array([1, 1])

# Calculer la sortie
y_output, h1_output, h2_output = multiperceptron(x, w1, w2)

print("Sortie du premier neuron de la couche cachée est :", h1_output)
print("Sortie du deuxième neuron de la couche cachée est :", h2_output)
print("Sortie finale du réseau multicouche est :", y_output)


'''
Apprentissage perceptron multicouches pour résoudre la porte XOR
'''
# Création de l'ensemble d'apprentissage pour le XOR
x = np.array([[0, 1, 0, 1],
              [0, 0, 1, 1]])  # Entrées
yd = np.array([0, 1, 1, 0])  # Sorties désirées

# Paramètres d'apprentissage
epoch = 20
batch_size = 4

# Apprentissage du réseau
w1, w2, erreur = multiperceptron_widrow(x, yd, epoch, batch_size)

# Affichage de l'évolution de l'erreur
afficher_evolution_erreur(erreur, "Évolution de l'erreur pour le XOR", "Itération","Erreur cumulée", save_dir="../results/perceptron_multilayers_learning")
affichier_multilayer_perceptron(x, yd, w1, w2, sigmoid, save_dir="../results/perceptron_multilayers_learning")