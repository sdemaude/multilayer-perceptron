import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt

from itertools import combinations
from pandas.plotting import scatter_matrix
from scipy.stats import wasserstein_distance


parser = argparse.ArgumentParser(description='')
parser.add_argument('-v', '--visualization', help='Display data visualization', action='store_true')


def wasserstein(groups, features):
    wasserstein_score = pd.DataFrame(columns=features, dtype=float)
    for (nameA, groupA), (nameB, groupB) in combinations(groups, 2):
        row = pd.Series(index=features)
        for group in features:
            dist = wasserstein_distance(groupA[group].dropna(), groupB[group].dropna())
            row[group] = dist
        wasserstein_score = pd.concat([wasserstein_score, row.to_frame().T], ignore_index=True)
    
    return wasserstein_score.mean()


def wasserstein_plot(groups, features):
    fig, ax = plt.subplots(figsize=(15, 10))
    plt.subplots_adjust(top=0.9, bottom=0.25)

    values = wasserstein(groups, features).values

    ax.set(title='Mean Wasserstein Distance per Groups per Features',
           ylabel='Mean Wasserstein Distance', xlabel='Features')

    cmap = mpl.colormaps['summer']
    norm = (values - values.min()) / (values.max() - values.min())
    colors = cmap(norm)

    bars = ax.bar(features, values, color=colors)

    for bar, h in zip(bars, values):
        ax.annotate(f'{h:.2f}', (bar.get_x() + bar.get_width() / 2, h), ha='center', va='bottom')

    ax.tick_params(axis='x', rotation=80)

    plt.draw()
    plt.show()


def pair_plot(df, features):
    # Sélection des 6 features avec la plus grande distance de Wasserstein
    features = wasserstein(df.groupby('diagnosis'), features).sort_values(ascending=False).index[:6].tolist()

    # Création du pair plot
    scatter_matrix(
        df[features + ['diagnosis']],
        figsize=(15, 10),
        diagonal='kde',
        color=df['diagnosis'].map({'B': 'darkseagreen', 'M': 'orangered'})
    )

    plt.suptitle('Pair plot des features clefs du dataset', y=1.02)
    plt.show()


def data_visualization(df):
    # Affichage du graphique Wasserstein
    wasserstein_plot(df.groupby('diagnosis'), df.drop(columns=['diagnosis', 'id']).columns)

    # Affichage du pair plot
    pair_plot(df, df.drop(columns=['id', 'diagnosis']).columns)


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
    
    # TODO:
    # Creer un programme pour separer le dataset en deux parties (entrainement et validation)
        # Separer les données en train et test (80/20)
    # Crer un programme d'entrainement
        # Implementer la fonction d'activation 'softmax' sur la couche de sortie
        # Implementer deux 'learning curve graphs' afficher a la fin de l'entrainement
    # Afficher les courbes d'apprentissage
    # Afficher les metrics d'entrainement et de validation a chaque epoch pour visualiser les performances du model
    # Crer un programme de prediction
    # Ajouter les arguments du programmes : python train.py --layer 24 24 24 --epochs 84 --loss categoricalCrossEntropy --batch_size 8 --learning_rate 0.0314

if __name__== '__main__':
    main()