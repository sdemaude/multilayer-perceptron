import numpy as np
import pandas as pd


def convert_training_history():
    try:
        training_history = np.load("training_history.npy")
        df = pd.DataFrame(training_history, columns=['train_loss', 'val_loss', 'train_acc', 'val_acc'])
        df.to_csv("training_history.csv", index=False)
        print("Training history successfully converted to training_history.csv")
    except Exception as e:
        print(f"Error converting training history: {e}")


if __name__ == "__main__":
    convert_training_history()