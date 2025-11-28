from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


def initialisation(dimensions):
    params = {}
    C = len(dimensions)

    for c in range(1, C):
        params[f"W{c}"] = np.random.randn(dimensions[c], dimensions[c - 1]) * 0.01
        params[f"b{c}"] = np.zeros((dimensions[c], 1))

    return params


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def softmax(Z):
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)
    return np.exp(Z_shift) / np.sum(np.exp(Z_shift), axis=0, keepdims=True)


def forward_propagation(X, params):
    activations = {"A0": X}
    C = len(params) // 2

    for c in range(1, C):
        Z = params[f"W{c}"].dot(activations[f"A{c-1}"]) + params[f"b{c}"]
        activations[f"A{c}"] = sigmoid(Z)

    # dernière couche = softmax
    Z = params[f"W{C}"].dot(activations[f"A{C-1}"]) + params[f"b{C}"]
    activations[f"A{C}"] = softmax(Z)

    return activations


def back_propagation(y, params, activations):
    gradients = {}
    m = y.shape[1]
    C = len(params) // 2

    # --- couche finale (softmax) ---
    dZ = activations[f"A{C}"] - y

    for c in reversed(range(1, C + 1)):
        gradients[f"dW{c}"] = 1/m * dZ.dot(activations[f"A{c-1}"].T)
        gradients[f"db{c}"] = 1/m * np.sum(dZ, axis=1, keepdims=True)

        if c > 1:
            W = params[f"W{c}"]
            A_prev = activations[f"A{c-1}"]
            dA_prev = W.T.dot(dZ)
            dZ = dA_prev * A_prev * (1 - A_prev)  # dérivée sigmoïde

    return gradients


def update(params, gradients, lr):
    C = len(params) // 2

    for c in range(1, C + 1):
        params[f"W{c}"] -= lr * gradients[f"dW{c}"]
        params[f"b{c}"] -= lr * gradients[f"db{c}"]

    return params


def predict(X, params):
    activations = forward_propagation(X, params)
    C = len(params) // 2
    A = activations[f"A{C}"]
    return np.argmax(A, axis=0)


def deep_neural_network(X, y, hidden_layers, learning_rate, n_iter):
    
    dimensions = [X.shape[0], *hidden_layers, y.shape[0]]
    params = initialisation(dimensions)

    history = np.zeros((n_iter, 2))
    C = len(params) // 2

    for i in tqdm(range(n_iter)):
        activations = forward_propagation(X, params)
        gradients = back_propagation(y, params, activations)
        params = update(params, gradients, learning_rate)

        A = activations[f"A{C}"]

        history[i, 0] = log_loss(y.T, A.T)
        y_pred = np.argmax(A, axis=0)
        y_true = np.argmax(y, axis=0)
        history[i, 1] = accuracy_score(y_true, y_pred)

    # Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history[:, 0], label='loss')
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history[:, 1], label='acc')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    np.save("model.npy", params)
    return history