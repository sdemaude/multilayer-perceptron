from training import forward_propagation
import numpy as np
import pandas as pd


def binary_cross_entropy(y, p):
        epsilon = 1e-15
        return -np.mean(y * np.log(p + epsilon) + (1 - y) * np.log(1 - p + epsilon))


def predict_and_display(X, y, parameters, class_names=['B', 'M']):
    
    L = len(parameters) // 2

    # forward propagation
    probabilities = forward_propagation(X, parameters, L)['A' + str(L)]
    predictions = np.argmax(probabilities, axis=0) # keep the one with highest probability

    # display results
    df = pd.DataFrame({'Prediction': [class_names[i] for i in predictions]})
    df['True Label'] = [class_names[i] for i in y.flatten()]
    for i, name in enumerate(class_names):
        df[f'Probability_{name}'] = probabilities[i, :]

    print(f"\nSample predictions (first 15):\n{df.head(15)}")

    # compute accuracy
    acc = np.mean(predictions == y) * 100
    print(f"\nAccuracy: {acc:.2f}%")

    # save predictions to CSV file
    df.to_csv('predictions.csv', index=False)
    print(f"\nPredictions saved to 'predictions.csv'")

    return predictions, probabilities