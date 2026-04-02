from tqdm import tqdm
import numpy as np
import pandas as pd


# Initialize model parameters randomly for all layers.
def initialisation(dimensions):  
    parameters = {}
    L = len(dimensions)

    # set each parameter to a random value
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l-1]) * np.sqrt(1 / dimensions[l-1]) # Xavier initialization
        parameters['b' + str(l)] = np.random.randn(dimensions[l], 1)

    return parameters, L - 1


# Calculate the sigmoid activation function.
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


# Calculate the softmax activation function with shifted values for numerical stability.
def softmax(Z):
    Z_shifted = Z - np.max(Z, axis=0, keepdims=True)
    exps = np.exp(Z_shifted)
    return exps / np.sum(exps, axis=0, keepdims=True)


# Perform forward propagation to calculate activations for each layer.
def forward_propagation(X, parameters, layer_number):
    activations = {'A0' : X}

    for l in range(1, layer_number + 1):
        # w * x + b
        Z = parameters['W' + str(l)].dot(activations['A' + str(l - 1)]) + parameters['b' + str(l)]
        
        if l == layer_number:
            activations['A' + str(l)] = softmax(Z)
        else:
            activations['A' + str(l)] = sigmoid(Z)

    return activations


# Perform back propagation to compute gradients for parameter updates.
def back_propagation(y, parameters, activations, layer_number):
    m = y.shape[0]
    gradients = {}

    # output layer activation
    A_L = activations['A' + str(layer_number)]
    dZ = A_L.copy()
    dZ[y, np.arange(m)] -= 1

    for l in reversed(range(1, layer_number + 1)): # end to start
        gradients['dW' + str(l)] = 1 / m * np.dot(dZ, activations['A' + str(l - 1)].T)
        gradients['db' + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            # formula to go from layer l to layer l-1
            dZ = np.dot(parameters['W' + str(l)].T, dZ) * activations['A' + str(l - 1)] * (1 - activations['A' + str(l - 1)])
    
    return gradients


# Update weights and biases using computed gradients and a learning rate.
def update(gradients, parameters, learning_rate, layer_number):
    for l in range(1, layer_number + 1):
        # new weight = old weight - learning rate * slope
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * gradients['dW' + str(l)] 
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * gradients['db' + str(l)]
    
    return parameters


# Calculate sparse categorical cross-entropy loss.
def sparse_categorical_cross_entropy(y, A):
    epsilon = 1e-15
    m = y.shape[0]
    return -np.mean(np.log(A[y, np.arange(m)] + epsilon))


# Calculate binary F1 score for predicted labels.
def f1_score(y_true, y_pred):
    true_positive = np.sum((y_true == 1) & (y_pred == 1))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    false_negative = np.sum((y_true == 1) & (y_pred == 0))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator > 0 else 0.0


# Train a deep neural network, managing epochs, validation, and early stopping.
# X = Features (matrix)
# y = Labels (vector)
# hidden_layers = size of each layer (tuple)
# X_val, y_val = validation set
def deep_neural_network(X, y, hidden_layers, learning_rate, epochs, X_val, y_val, patience=200, min_delta=1e-4, epochs_print=100):
    n_classes = np.max(y) + 1
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])    # set the input layer
    dimensions.append(n_classes)        # set the output layer

    parameters, layer_number = initialisation(dimensions)

    training_history = np.zeros((int(epochs), 4))
    best_val_loss = np.inf
    best_parameters = None
    wait = 0
    early_stopping_epoch = None

    for i in tqdm(range(epochs)):
        # training step
        activations = forward_propagation(X, parameters, layer_number)
        gradients = back_propagation(y, parameters, activations, layer_number)
        parameters = update(gradients, parameters, learning_rate, layer_number)

        final_activation = activations['A' + str(layer_number)]
        train_preds = np.argmax(final_activation, axis=0)
        training_history[i, 0] = sparse_categorical_cross_entropy(y, final_activation)  # training loss
        training_history[i, 2] = np.mean(train_preds == y)                            # training accuracy
        train_f1 = f1_score(y, train_preds)

        # validation step
        val_activations = forward_propagation(X_val, parameters, layer_number)
        val_final_activations = val_activations['A' + str(layer_number)]
        val_preds = np.argmax(val_final_activations, axis=0)

        training_history[i, 1] = sparse_categorical_cross_entropy(y_val, val_final_activations) # val loss
        training_history[i, 3] = np.mean(val_preds == y_val)                                  # val accuracy
        val_f1 = f1_score(y_val, val_preds)

        # early stopping
        if training_history[i, 1] < best_val_loss - min_delta:
            best_val_loss = training_history[i, 1]
            best_parameters = {k: v.copy() for k, v in parameters.items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                early_stopping_epoch = i + 1
                tqdm.write(f"\n Early stopping triggered at epoch {early_stopping_epoch}")
                parameters = best_parameters
                break

        # print progress
        if (i + 1) % epochs_print == 0:
            tqdm.write(
                f"Epoch {i+1}/{epochs} - "
                f"Loss: {training_history[i,0]:.4f} - "
                f"Val Loss: {training_history[i,1]:.4f} - "
                f"Accuracy: {training_history[i,2]:.4f} - "
                f"Val Accuracy: {training_history[i,3]:.4f} - "
                f"F1: {train_f1:.4f} - "
                f"Val F1: {val_f1:.4f}"
            )

    np.save("model_params.npy", parameters)
    pd.DataFrame(training_history, columns=['Train Loss', 'Val Loss', 'Train Accuracy', 'Val Accuracy']).to_csv("training_history.csv", index=False)

    print("Model parameters and training history saved in training_history.csv and model_params.npy")

    epochs_ran = i + 1
    training_history = training_history[:epochs_ran]

    return training_history, early_stopping_epoch