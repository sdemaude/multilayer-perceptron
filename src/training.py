from tqdm import tqdm
import numpy as np

def initialisation(dimensions):  
    
    parameters = {}
    L = len(dimensions)

    np.random.seed(1)

    # set each parameter to a random value
    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1])
        parameters['b' + str(l)] = np.random.randn(dimensions[l], 1)

    return parameters, L - 1


def forward_propagation(X, parameters, layer_number):
    
    activations = {'A0' : X}

    for l in range(1, layer_number + 1):
        # w * x + b
        Z = parameters['W' + str(l)].dot(activations['A' + str(l - 1)]) + parameters['b' + str(l)]
        #if fin :
        #    softmax
        #else :
            # Sigmoid
        activations['A' + str(l)] = 1 / (1 + np.exp(-Z))

    return activations


def back_propagation(y, parameters, activations, layer_number):

    m = y.shape[1]

    # final layer activation
    dZ = activations['A' + str(layer_number)] - y # plus ca du tout
    gradients = {}

    for l in reversed(range(1, layer_number + 1)): # end to start
        gradients['dW' + str(l)] = 1 / m * np.dot(dZ, activations['A' + str(l - 1)].T)
        gradients['db' + str(l)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
        if l > 1:
            # formula to go from layer l to layer l-1
            dZ = np.dot(parameters['W' + str(l)].T, dZ) * activations['A' + str(l - 1)] * (1 - activations['A' + str(l - 1)]) # a recalculer en fonction de softmax
    
    return gradients


def update(gradients, parameters, learning_rate, layer_number):

    for l in range(1, layer_number + 1):
        # nouveau_poids = ancien_poids - learning_rate * pente
        parameters['W' + str(l)] = parameters['W' + str(l)] - learning_rate * gradients['dW' + str(l)] 
        parameters['b' + str(l)] = parameters['b' + str(l)] - learning_rate * gradients['db' + str(l)]
    
    return parameters


def log_loss(y, A):
    epsilon = 1e-15
    return 1 / len(y) * np.sum(-y * np.log(A + epsilon) - (1 - y) * np.log(1 - A + epsilon))


def accuracy_score(y, y_predicted):
    return np.mean(y_predicted == y)


# X = Features (matrix)
# y = Labels (vector)
# hidden_layers = size of each layer (tuple)
# X_val, y_val = validation set
def deep_neural_network(X, y, hidden_layers, learning_rate, epochs, X_val, y_val, patience=20, min_delta=1e-4):

    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])    # set the input layer
    dimensions.append(y.shape[0])       # set the output layer
    
    parameters, layer_number = initialisation(dimensions)
    
    training_history = np.zeros((int(epochs), 4))
    best_val_loss = np.inf
    best_parameters = None
    wait = 0

    for i in tqdm(range(epochs)):
        # training step
        activations = forward_propagation(X, parameters, layer_number)
        gradients = back_propagation(y, parameters, activations, layer_number)
        parameters = update(gradients, parameters, learning_rate, layer_number)

        final_activation = activations['A' + str(layer_number)]
        training_history[i, 0] = log_loss(y.flatten(), final_activation.flatten()) # pas forcement log loss ?
        training_history[i, 2] = accuracy_score(y.flatten(), final_activation.flatten() >= 0.5)

        # validation step
        val_activations = forward_propagation(X_val, parameters, layer_number)
        val_final_activations = val_activations['A' + str(layer_number)]

        training_history[i, 1] = log_loss(y_val.flatten(), val_final_activations.flatten())
        training_history[i, 3] = accuracy_score(y_val.flatten(), val_final_activations.flatten() >= 0.5)

        # early stopping
        if training_history[i, 1] < best_val_loss - min_delta:
            best_val_loss = training_history[i, 1]
            best_parameters = parameters.copy()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"\n Early stopping triggered at epoch {i}")
                parameters = best_parameters
                break

        if (i + 1) % 100 == 0:
            print(f"Epoch {i + 1}/{epochs} - Loss: {training_history[i, 0]:.4f} - Val Loss: {training_history[i, 1]:.4f} - Accuracy: {training_history[i, 2]:.4f} - Val Accuracy: {training_history[i, 3]:.4f}")

    # save model params + training history
    np.save("model_params.npy", parameters)
    np.save("training_history.npy", training_history)

    return training_history