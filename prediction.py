# %%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from train import get_predictions, forward_prop

test_data = pd.read_csv('/home/boostdvd/workspace/neural_network/data/mnist_test.csv')
test_data = np.array(test_data).T
y_test = test_data[0]
x_test = test_data[1:] / 255.0

def model(index_zd, W1, b1, W2, b2, ma_zd, ma_odp):
    zd = ma_zd[:, index_zd, None]
    odp = ma_odp[index_zd]
    _,_,_, wynik = forward_prop(W1, b1, W2, b2, zd)
    predykcja = get_predictions(wynik)
    aktualne_zd = zd.reshape((28,28))
    plt.imshow(aktualne_zd, cmap="gray")
    plt.title(f"Predykcja: {predykcja.item()} | Wynik: {odp}" )
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    weights = np.load('data/model_weights.npz')

    W1 = weights["W1"]
    b1 = weights["b1"]
    W2 = weights["W2"]
    b2 = weights["b2"]

    index = input("jaki index chcesz sprawdzić? ")

    while index != "stop":
        model(int(index), W1, b1, W2, b2, x_test, y_test)
        index = input("jaki index chcesz sprawdzić? (wpisz 'stop' żeby zakończyć)")
# %%
