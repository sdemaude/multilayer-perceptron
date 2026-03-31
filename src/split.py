# Split the dataset into training and test sets (80/20 ratio).
def data_split(df):
    train = df.sample(frac=0.8, random_state=42) # seed for reproducibility
    test = df.drop(train.index)

    return train, test