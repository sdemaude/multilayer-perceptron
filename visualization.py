import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import combinations
from pandas.plotting import scatter_matrix
from scipy.stats import wasserstein_distance


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