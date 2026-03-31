import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from itertools import combinations
from pandas.plotting import scatter_matrix
from scipy.stats import wasserstein_distance


# Calculate the average Wasserstein distance for pairs of groups per feature.
def wasserstein(groups, features):
    wasserstein_score = pd.DataFrame(columns=features, dtype=float)
    for (nameA, groupA), (nameB, groupB) in combinations(groups, 2):
        row = pd.Series(index=features)
        for group in features:
            dist = wasserstein_distance(groupA[group].dropna(), groupB[group].dropna())
            row[group] = dist
        wasserstein_score = pd.concat([wasserstein_score, row.to_frame().T], ignore_index=True)
    
    return wasserstein_score.mean()


# Show a bar chart displaying the mean Wasserstein distance for each feature.
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


# Generate a scatter matrix pair plot with the highest Wasserstein-scored features.
def pair_plot(df, features):
    features = wasserstein(df.groupby('diagnosis'), features).sort_values(ascending=False).index[:6].tolist()   # the 6 features with largest Wasserstein distance

    scatter_matrix(
        df[features + ['diagnosis']],
        figsize=(15, 10),
        diagonal='kde',
        color=df['diagnosis'].map({'B': 'darkseagreen', 'M': 'orangered'})
    )

    plt.suptitle('Pair plot des features clefs du dataset', y=1.02)
    plt.show()


# Execute the overall visualization routine (Wasserstein and pair plots).
def data_visualization(df):
    wasserstein_plot(df.groupby('diagnosis'), df.drop(columns=['diagnosis', 'id']).columns)
    pair_plot(df, df.drop(columns=['id', 'diagnosis']).columns)