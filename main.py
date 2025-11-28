import argparse
import pandas as pd
import numpy as np
from visualization import data_visualization
from separation import data_split
from training import deep_neural_network


parser = argparse.ArgumentParser(description='')
subparsers = parser.add_subparsers(dest='command', required=True)

# Visualize Command
subparsers.add_parser('visualize', help='Display data visualization')

# Separate Command
subparsers.add_parser('split', help='Split the dataset into training and testing sets')

# Train Command
train_parser = subparsers.add_parser('train', help='Train the model')

train_parser.add_argument('-l', '--layers', nargs='+', type=int)#, help='List of hidden layer sizes, default : ?')
train_parser.add_argument('-e', '--epochs', type=int)
train_parser.add_argument('-c', '--loss', type=str, choices=['categoricalCrossEntropy', 'binaryCrossEntropy'])
train_parser.add_argument('-b', '--batch_size', type=int)
train_parser.add_argument('-r', '--learning_rate', type=float)

# Predict Command
predict_parser = subparsers.add_parser('predict', help='Make predictions with the trained model')
# predict_parser.add_argument('--input', type=str, required=True)

args = parser.parse_args()


def data_preparation():
    # Récupération du dataset et ajout des noms de colonnes
    df = pd.read_csv('data.csv', header=None)
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


def to_one_hot(y, num_classes=2):
    m = y.shape[1]
    Y = np.zeros((num_classes, m))
    for i in range(m):
        Y[y[0, i], i] = 1
    return Y


def main():
    df = data_preparation()

    match args.command:
        case 'visualize':
            data_visualization(df)

        case 'split':
            train, test = data_split(df)

            # Sauvegarde des datasets
            train.to_csv('train.csv', index=False)
            test.to_csv('test.csv', index=False)

        case 'train':
            if 'train.csv' not in pd.io.common.get_handle('train.csv', 'r').handle.name:
                print("Training dataset 'train.csv' not found. Please run the separation step first.")
                return

            train = pd.read_csv('train.csv')

            # Vectorisation des donnees
            X = train.drop(columns=['id', 'diagnosis']).T.to_numpy()
            y = train['diagnosis'].map({'M': 1, 'B': 0}).to_numpy().reshape(1, -1)
            y = to_one_hot(y, num_classes=2)

            hidden_layer_size = tuple(args.layers) if args.layers else (16, 16, 16)
            epochs = args.epochs if args.epochs else 3000
            #loss = args.loss if args.loss else 'categoricalCrossEntropy'
            #batch_size = args.batch_size if args.batch_size else 32
            learning_rate = args.learning_rate if args.learning_rate else 0.01

            #print(f"Hidden layers: {hidden_layer_size}")
            #print(f"Learning rate: {learning_rate}")

            deep_neural_network(X, y, hidden_layer_size, learning_rate, epochs)

        case 'predict':
            pass


if __name__== '__main__':
    main()