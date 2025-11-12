import pandas as pd

df = pd.read_csv('data.csv', header=None)
df.drop(index=0)
print(df.describe())