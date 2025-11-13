import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

'''
def binary_cross_entropy(y, Af):
    m = y.shape[1]
    bce = -1/m * np.sum(y * np.log(Af) + (1 - y) * np.log(1 - Af))
    return bce
'''

def initialisation(dimensions):
    params = {}
    C = len(dimensions)

    np.random.seed(1)

    for c in range(1, C):
        params['W' + str(c)] = np.random.randn(dimensions[c], dimensions[c - 1])
        params['b' + str(c)] = np.random.randn(dimensions[c], 1)

    return params


def forward_propagation(X, params):

    activations = {'A0': X}

    C = len(params) // 2

    for c in range(1, C + 1):
        Z = params['W' + str(c)].dot(activations['A' + str(c - 1)]) + params['b' + str(c)]
        activations['A' + str(c)] = 1 / (1 + np.exp(-Z))

    return activations


def back_propagation(y, params, activations):

    m = y.shape[1]
    C = len(params) // 2

    dZ = activations['A' + str(C)] - y
    gradients = {}

    for c in reversed(range(1, C + 1)):
        gradients['dW' + str(c)] = 1/m * np.dot(dZ, activations['A' + str(c - 1)].T)
        gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
    if c > 1:
        dZ = np.dot(params['W' + str(c)].T, dZ) * activations['A' + str(c - 1)] * (1 - activations['A' + str(c - 1)])

    return gradients


def update(gradients, params, learning_rate):

    C = len(params) // 2

    for c in range(1, C + 1):
        params['W' + str(c)] = params['W' + str(c)] - learning_rate * gradients['dW' + str(c)]
        params['b' + str(c)] = params['b' + str(c)] - learning_rate * gradients['db' + str(c)]

    return params


def predict(X, params):
    activations = forward_propagation(X, params)
    C = len(params) // 2
    Af = activations['A' + str(C)]
    return Af >= 0.5


def deep_neural_network(X, y, hidden_layers = (16, 16, 16), learning_rate = 0.001, n_iter = 3000):

    # initialisation params
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(1)
    params = initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et BCE
    training_history = np.zeros((int(n_iter), 2))

    C = len(params) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X, params)
        gradients = back_propagation(y, params, activations)
        params = update(gradients, params, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du BCE et de l'accuracy
        training_history[i, 0] = (binary_cross_entropy(y.flatten(), Af.flatten()))
        y_pred = predict(X, params)
        training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))

    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()

    return training_history