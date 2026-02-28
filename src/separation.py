def data_split(df):
    # Split the data into train and test sets (80/20)
    train = df.sample(frac=0.8, random_state=42)
    test = df.drop(train.index)
    return train, test