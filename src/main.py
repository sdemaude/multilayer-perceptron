import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

from visualization import data_visualization
from separation import data_split
from training import deep_neural_network
from prediction import predict_and_display, binary_cross_entropy


def init_parser():
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('visualize', help='Display data visualization')
    subparsers.add_parser('split', help='Split the dataset into training and testing sets')
    train_parser = subparsers.add_parser('train', help='Train the model')

    train_parser.add_argument('-l', '--layers', nargs='+', type=int)
    train_parser.add_argument('-e', '--epochs', type=int)
    train_parser.add_argument('-r', '--learning_rate', type=float)
    train_parser.add_argument('--epochs_print', type=int)

    subparsers.add_parser('predict', help='Make predictions with the trained model')

    return parser.parse_args()


def data_preparation():
    # Load the dataset and add column names
    DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "data.csv"
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Couldn't find the dataset file: data.csv\n")

    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ['id', 'diagnosis',
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']

    # Normalize the dataset
    for column in df.drop(columns=['id', 'diagnosis']).columns:
        df[column] = (df[column] - df[column].min()) / (df[column].max() - df[column].min())
    
    return df


def display_plots(training_history):
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.plot(training_history[:, 1], label='val loss')
    plt.legend()
    plt.title("Loss")

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 2], label='train acc')
    plt.plot(training_history[:, 3], label='val acc')
    plt.legend()
    plt.title("Accuracy")

    plt.show()


def load_dataset(file_name):
    DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / file_name
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Couldn't find the dataset file: {file_name}\n")

    dataset = pd.read_csv(DATA_PATH)

    # Vectorize the data
    X = dataset.drop(columns=['id', 'diagnosis']).T.to_numpy()
    y = dataset['diagnosis'].map({'M': 1, 'B': 0}).to_numpy()
    y = y.reshape(1, y.shape[0])

    return X, y


def main():
    args = init_parser()

    match args.command:
        case 'visualize':
            df = data_preparation()
            data_visualization(df)

        case 'split':
            df = data_preparation()
            train, test = data_split(df)

            train.to_csv('datasets/train.csv', index=False)
            test.to_csv('datasets/test.csv', index=False)

        case 'train':
            X, y = load_dataset('train.csv')
            X_val, y_val = load_dataset('test.csv')

            y = y.reshape(-1)
            y_val = y_val.reshape(-1)

            hidden_layers = tuple(args.layers) if args.layers else (48, 64)
            learning_rate = args.learning_rate if args.learning_rate else 0.1
            epochs = args.epochs if args.epochs else 3000
            epochs_print = args.epochs_print if args.epochs_print else 100

            training_history = deep_neural_network(X, y, hidden_layers, learning_rate, epochs, X_val, y_val, patience=50, epochs_print=epochs_print)
            display_plots(training_history)

        case 'predict':
            X, y = load_dataset('test.csv')
            parameters = np.load("model_params.npy", allow_pickle=True).item()

            predictions, probabilities = predict_and_display(X, y, parameters)

            # calculate loss
            loss = binary_cross_entropy(y, probabilities[1, :]) # use the probabilities of the positive class (malignant)
            print(f"Test Loss using BCE: {loss:.4f}")

            # confusion matrix
            cm = confusion_matrix(y.flatten(), predictions.flatten())
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['B', 'M'])
            disp.plot(cmap='winter')
            plt.show()


if __name__== '__main__':
    main()