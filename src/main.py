import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization import data_visualization
from separation import data_split
from training import deep_neural_network
from prediction import predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

# argument parser
parser = argparse.ArgumentParser(description='')
subparsers = parser.add_subparsers(dest='command', required=True)

subparsers.add_parser('visualize', help='Display data visualization')
subparsers.add_parser('split', help='Split the dataset into training and testing sets')
train_parser = subparsers.add_parser('train', help='Train the model')

train_parser.add_argument('-l', '--layers', nargs='+', type=int)
train_parser.add_argument('-e', '--epochs', type=int)
train_parser.add_argument('-c', '--loss', type=str, choices=['CCE', 'BCE'])
train_parser.add_argument('-b', '--batch_size', type=int)
train_parser.add_argument('-r', '--learning_rate', type=float)

predict_parser = subparsers.add_parser('predict', help='Make predictions with the trained model')
# predict_parser.add_argument('--input', type=str, required=True)

args = parser.parse_args()


def data_preparation():
    # Récupération du dataset et ajout des noms de colonnes
    DATA_PATH = Path(__file__).resolve().parent.parent / "datasets" / "data.csv"

    # TODO add error handling
    #if not DATA_PATH.exists():
    #    raise FileNotFoundError(f"Couldn't find the dataset file\n")

    df = pd.read_csv(DATA_PATH, header=None)
    df.columns = ['id', 'diagnosis',
        'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
        'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
        'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
        'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
        'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
        'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']

    # Normalisation du dataset
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

    # TODO add error handling
    #if not DATA_PATH.exists():
    #    raise FileNotFoundError(f"Couldn't find the dataset file, please run the separation step first\n")

    dataset = pd.read_csv(DATA_PATH)

    # Vectorisation des donnees
    X = dataset.drop(columns=['id', 'diagnosis']).T.to_numpy()
    y = dataset['diagnosis'].map({'M': 1, 'B': 0}).to_numpy()
    y = y.reshape(1, y.shape[0])

    return X, y

def main():
    df = data_preparation()

    match args.command:

        case 'visualize':
            data_visualization(df)

        case 'split':
            train, test = data_split(df)

            # save datasets
            train.to_csv('train.csv', index=False)
            test.to_csv('test.csv', index=False)

        case 'train':
            X, y = load_dataset('train.csv')
            X_val, y_val = load_dataset('test.csv')

            hidden_layers = tuple(args.layers) if args.layers else (16, 16, 16)
            epochs = args.epochs if args.epochs else 3000
            #loss = args.loss if args.loss else 'CCE'
            #batch_size = args.batch_size if args.batch_size else 32
            learning_rate = args.learning_rate if args.learning_rate else 0.05

            training_history = deep_neural_network(X, y, hidden_layers, learning_rate, epochs, X_val, y_val)
            display_plots(training_history)

        case 'predict':

            X, y = load_dataset('test.csv')
            parameters = np.load("model_params.npy", allow_pickle=True).item()

            predictions = predict(X, parameters)

            # confusion matrix
            cm = confusion_matrix(y.flatten(), predictions.flatten())
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['B', 'M'])
            disp.plot(cmap='winter')
            plt.show()


if __name__== '__main__':
    main()