import matplotlib.pyplot as plt
import numpy as np
from utilities import data_visualize as dv

# a = np.load("../SVM/rndScore.npy")
# dv.plot_roc(a[0,:], a[1,:])
# plt.show()
def plotactivation():
    x = np.linspace(-10, 10, 1000)
    y = 1 / (1 + np.exp(-x))
    y2 = np.fmax(0, x)

    plt.figure(figsize=(5, 4))
    plt.figure(1)
    plt.plot(x, y, 'k')
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$\sigma (x)$", fontsize=14)
    # plt.grid(True, linestyle='--', color="#93a1a1", alpha=0.3)
    plt.legend([r"$\sigma (x) = \frac{1}{ 1 + e ^{-x}}$"], prop={'size': 12})

    plt.figure(figsize=(5, 4))
    plt.figure(2)
    plt.plot(x, y2, 'k')
    plt.xlabel(r"$x$", fontsize=14)
    plt.ylabel(r"$R (x)$", fontsize=14)
    # plt.grid(True, linestyle='--', color="#93a1a1", alpha=0.3)
    plt.legend([r"$R(x) = \max (0, x)$"], prop={'size': 14})

    plt.show()

plotactivation()
