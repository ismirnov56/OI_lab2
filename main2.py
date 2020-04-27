import numpy as np
import matplotlib.pyplot as plt
from scipy import special

def myfr(x, y):
    m = 5
    k = 0.3
    r = np.zeros((len(x), len(y)), dtype=complex)
    for i in range(len(x)):
        for j in range(len(y)):
            r[i][j] = np.sqrt(x[i]**2 + y[j]**2)
    return r, np.complex(0, 1) * np.exp(-np.complex(0, 1) * m * r) * np.exp(-k * (r**2)) * (np.cos(r**2) ** 2)

def myFr(fr, r, ro, hx):
    F = 2 * np.pi * fr * special.jv(0, 2 * np.pi * r * ro) * r * (hx**2)
    return F

def show(f, F):
    a = -4
    b = 4

    fig, axs = plt.subplots(2, 2)

    ax = axs[0, 0]
    ax.imshow(np.abs(f), extent=[a, b, a, b])
    ax = axs[0, 1]
    ax.imshow(np.angle(f), extent=[a, b, a, b])

    ax = axs[1, 0]
    ax.imshow(np.abs(F), extent=[a, b, a, b])
    ax = axs[1, 1]
    ax.imshow(np.angle(F), extent=[a, b, a, b])

    plt.show()


def main():
    a = 4
    N = 50
    hx = 2 * a / N
    x = np.linspace(-a, a, N)
    y = np.linspace(-a, a, N)
    r, f = myfr(x, y)
    Fr = myFr(f, r, r, hx)
    show(f, Fr)


if __name__ == '__main__':
    main()