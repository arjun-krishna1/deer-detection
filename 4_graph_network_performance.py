import pickle5 as pickle
import tensorflow
import matplotlib.pyplot as plt

if __name__ == "__main__":
    with open("data/checkpoint/history.pickle", "rb") as f:
        history = pickle.load(f)

    print(history)

    acc = history.history["acc"]
    val_acc = history.history["acc"]
    acc = history.history["acc"]
    val_loss = history.history["acc"]

    epochs = range(len(acc))