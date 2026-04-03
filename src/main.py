import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

from visualize import data_visualization
from split import data_split
from train import deep_neural_network
from predict import predict_and_display, binary_cross_entropy

columns = ['id', 'diagnosis',
    'mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness',
    'mean_compactness', 'mean_concavity', 'mean_concave_points', 'mean_symmetry', 'mean_fractal_dimension',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se',
    'worst_radius', 'worst_texture', 'worst_perimeter', 'worst_area', 'worst_smoothness',
    'worst_compactness', 'worst_concavity', 'worst_concave_points', 'worst_symmetry', 'worst_fractal_dimension']


# Initialize and parse command-line arguments.
def init_parser():
    parser = argparse.ArgumentParser(description='')
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('visualize', help='Display data visualization')
    subparsers.add_parser('split', help='Split the dataset into training and testing sets')
    subparsers.add_parser('predict', help='Make predictions with the trained model')
    subparsers.add_parser('all', help='Run the entire mandatory pipeline: split, train, and predict')

    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('-l', '--layers', nargs='+', type=int)
    train_parser.add_argument('-e', '--epochs', type=int)
    train_parser.add_argument('-r', '--learning_rate', type=float)
    train_parser.add_argument('--epochs_print', type=int)

    return parser.parse_args()


# Display loss and accuracy plots from the training history.
def display_plots(training_history, early_stopping_epoch=None):
    plt.figure(figsize=(12, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.plot(training_history[:, 1], label='val loss')
    if early_stopping_epoch is not None:
        plt.axvline(early_stopping_epoch - 1, color='red', linestyle='--', linewidth=1.5)
        plt.text(early_stopping_epoch - 1, plt.ylim()[1] * 0.95, 'Early stopping ', color='red', ha='right', va='top')
    plt.legend()
    plt.title("Loss")

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 2], label='train acc')
    plt.plot(training_history[:, 3], label='val acc')
    if early_stopping_epoch is not None:
        plt.axvline(early_stopping_epoch - 1, color='red', linestyle='--', linewidth=1.5)
        plt.text(early_stopping_epoch - 1, plt.ylim()[1] * 0.95, 'Early stopping ', color='red', ha='right', va='top')
    plt.legend()
    plt.title("Accuracy")

    plt.show()


# Load a specific dataset file and prepare features (X) and labels (y).
def load_dataset(file_name):
    DATA_PATH = Path(__file__).resolve().parent.parent / file_name
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Couldn't find the dataset file: {file_name}\n")
    dataset = pd.read_csv(DATA_PATH)

    # check if the dataset contains the expected columns if not add them
    expected_columns = columns
    if not all(col in dataset.columns for col in expected_columns):
        dataset.columns = expected_columns

    return dataset


def data_preparation(dataset):
    # Standardize the dataset
    for column in dataset.drop(columns=['id', 'diagnosis']).columns:
        dataset[column] = (dataset[column] - dataset[column].mean()) / dataset[column].std()

    # Vectorize the data
    X = dataset.drop(columns=['id', 'diagnosis']).T.to_numpy()
    y = dataset['diagnosis'].map({'M': 1, 'B': 0}).to_numpy()
    y = y.reshape(1, y.shape[0])

    return X, y


# Main entry point to handle commands: visualize, split, train, and predict.
def main():
    try:
        args = init_parser()

        match args.command:
            case 'visualize':
                data_visualization(load_dataset('data.csv'))

            case 'split':
                data_split(load_dataset('data.csv'))
    
            case 'train':
                X, y = data_preparation(load_dataset('data_training.csv'))
                X_val, y_val = data_preparation(load_dataset('data_test.csv'))

                y = y.reshape(-1) # convert to a vector
                y_val = y_val.reshape(-1)
    
                # set hyperparameters with defaults if not provided
                hidden_layers = tuple(args.layers) if args.layers else (32, 32)
                learning_rate = args.learning_rate if args.learning_rate else 0.01
                epochs = args.epochs if args.epochs else 5000
                epochs_print = args.epochs_print if args.epochs_print else 1
    
                training_history, early_stop_epoch = deep_neural_network(X, y, hidden_layers, learning_rate, epochs, X_val, y_val, patience=500, epochs_print=epochs_print)
                display_plots(training_history, early_stop_epoch)
    
            case 'predict':
                X, y = data_preparation(load_dataset('data_test.csv'))
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

            case 'all':
                # split
                data_split(load_dataset('data.csv'))
                
                # train
                X, y = data_preparation(load_dataset('data_training.csv'))
                X_val, y_val = data_preparation(load_dataset('data_test.csv'))

                y = y.reshape(-1)
                y_val = y_val.reshape(-1)
    
                hidden_layers = (32, 32)
                learning_rate = 0.01
                epochs = 10000
                epochs_print = 100
    
                training_history, early_stop_epoch = deep_neural_network(X, y, hidden_layers, learning_rate, epochs, X_val, y_val, patience=500, epochs_print=epochs_print)
                display_plots(training_history, early_stop_epoch)
                
                # predict
                X, y = data_preparation(load_dataset('data_test.csv'))
                parameters = np.load("model_params.npy", allow_pickle=True).item()
    
                predictions, probabilities = predict_and_display(X, y, parameters)
    
                loss = binary_cross_entropy(y, probabilities[1, :])
                print(f"Test Loss using BCE: {loss:.4f}")
    
                cm = confusion_matrix(y.flatten(), predictions.flatten())
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['B', 'M'])
                disp.plot(cmap='winter')
                plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__== '__main__':
    main()