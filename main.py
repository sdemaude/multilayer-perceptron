import argparse
import pandas as pd

from visualization import data_visualization
from separation import data_split
from training import deep_neural_network


parser = argparse.ArgumentParser(description='')
parser.add_argument('-v', '--visualization', help='Display data visualization', action='store_true')
parser.add_argument('-s', '--separation', help='Separate the dataset into training and testing sets', action='store_true')
parser.add_argument('-t', '--training', help='Train the model', action='store_true')
parser.add_argument('-p', '--prediction', help='Make predictions with the trained model', action='store_true')


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


def main():
    df = data_preparation()

    if parser.parse_args().visualization:
        data_visualization(df)

    if parser.parse_args().separation:
        train, test = data_split(df)

        # Sauvegarde des datasets
        train.to_csv('train.csv', index=False)
        test.to_csv('test.csv', index=False)

    if parser.parse_args().training:
        if 'train.csv' not in pd.io.common.get_handle('train.csv', 'r').handle.name:
            print("Training dataset 'train.csv' not found. Please run the separation step first.")
            return
        
        train = pd.read_csv('train.csv')

        # Normalisation des donnees
        X = train.drop(columns=['id', 'diagnosis']).T.to_numpy()
        y = train['diagnosis'].map({'M': 1, 'B': 0}).to_numpy().reshape(1, -1)

        deep_neural_network(X, y,
            hidden_layers = (16, 16, 16),
            learning_rate = 0.001,
            n_iter = 3000
        )

    if parser.parse_args().prediction:
        pass


    # TODO:

    # Crer un programme d'entrainement
        # Vectoriser les donnees
        # Implementer la fonction d'activation 'softmax' sur la couche de sortie
        # Implementer deux 'learning curve graphs' afficher a la fin de l'entrainement

    # Afficher les courbes d'apprentissage
    # Afficher les metrics d'entrainement et de validation a chaque epoch pour visualiser les performances du model
    
    # Crer un programme de prediction
    # Ajouter les arguments du programmes : python train.py --layer 24 24 24 --epochs 84 --loss categoricalCrossEntropy --batch_size 8 --learning_rate 0.0314


if __name__== '__main__':
    main()