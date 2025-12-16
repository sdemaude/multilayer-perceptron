from training import forward_propagation

def predict(X, parameters):
    # predictions
    layer_number = len(parameters) // 2
    activations = forward_propagation(X, parameters, layer_number)
    final_activation = activations['A' + str(layer_number)]
    predictions = (final_activation >= 0.5).astype(int)

    return predictions