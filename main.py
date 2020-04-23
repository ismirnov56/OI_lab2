import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt



def myfft(fx, M, N, hx):
    newfx = np.zeros(M, dtype=complex)
    index = (M - N) / 2
    newfx[int(index):int(M - index)] = fx
    temp = np.copy(newfx)
    newfx[:int(M / 2)], newfx[int(M / 2):] = temp[int(M / 2):], temp[:int(M / 2)]
    dpf = fft(newfx) * hx
    temp = np.copy(dpf)
    dpf[:int(M / 2)], dpf[int(M / 2):] = temp[int(M / 2):], temp[:int(M / 2)]
    #temp = dpf[int(index):int(M - index)]
    return dpf

def gausf(x, hx, u, N):
    f = 0
    for i in range(1, N+1):
        f += np.exp(-(x[i-1]+hx/2)**2) * np.exp(-2*np.pi*np.complex(0,1)*u*(x[i-1]+hx/2))
    return f * hx

def finite(N, hx, x, u, M):
    F = np.zeros(M, dtype=np.complex)
    for i in range(M):
        F[i] = gausf(x, hx, u[i], N)
    return F

def myf(x, hx, u, N):
    f = 0
    for i in range(1, N + 1):
        f += (1 / (4 + (x[i - 1] + hx / 2) ** 2)) * np.exp(-2 * np.pi * np.complex(0, 1) * u * (x[i - 1] + hx / 2))
    return f * hx

def myfinite(N, hx, x, u, M):
    F = np.zeros(M, dtype=np.complex)
    for i in range(M):
        F[i] = myf(x, hx, u[i], N)
    return F

def show(f, ffft, finit, u, x):

    fig1, ax1 = plt.subplots()
    ax1.plot(x, abs(f), color="blue", label="input")
    ax1.plot(u, abs(ffft), color="red", label="ffft")
    ax1.plot(u, abs(finit), color="green", label="finte")
    ax1.legend()
    plt.grid()
    fig2, ax2 = plt.subplots()
    ax2.plot(x, np.angle(f), color="blue", label="input")
    ax2.plot(u, np.angle(ffft), color="red", label="ffft")
    ax2.plot(u, np.angle(finit), color="green", label="finite")
    ax2.legend()
    plt.grid()

    plt.show()

def main1():
    a = 4
    N = 100
    M = 2000
    hx = (a + a) / N

    x = np.linspace(-a, a, N)
    gausP = np.exp(-x**2)
    furGausP = myfft(gausP, M, N, hx)
    b = (N**2) / (4 * a * M)
    u = np.linspace(-a, a, M)
    hu = (b + b) / N
    gausP = np.exp(-x**2)
    finit = finite(N, hx, x, u, M)
    show(gausP, furGausP, finit, u, x)

    fx = 1 / (4 + x ** 2)
    myffftfx = myfft(fx, M, N, hx)
    myfinfx = myfinite(N, hx, x, u, M)
    show(fx, myffftfx, myfinfx, u, x)

def show2(f):
    a = -4
    b = 4

    fig0, ax0 = plt.subplots()
    fig1, ax1 = plt.subplots()

    ax0.imshow(np.abs(f), extent=[a, b, a, b])
    ax1.imshow(np.angle(f), extent=[a, b, a, b])

    plt.show()

def gaus2(N, x, y):
    f = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            f[i][j] = np.exp(-x[j] ** 2 - y[i]**2)
    return f

def myf2(N, x, y):
    f = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            f[i][j] = (1 / (4 + x[j] ** 2))*(1 / (4 + y[i] ** 2))
    return f

def myfft2(fx, M, N, hx, hy):
    F_str = np.zeros((N, M), dtype=complex)
    for i in range(0, N):
        F_str[i] = myfft(fx[i], M, N, hx)
    print(F_str.shape)
    F = np.zeros((M, M), dtype=complex)
    for i in range(0, M):
        F[:][i] = myfft(F_str[:,i], M, N, hy)
    return F

def main2():
    a = 4
    N = 100
    M = 2000
    hx = (a + a)/N
    hy = (a + a)/N
    x = np.linspace(-a, a, N)
    y = np.linspace(-a, a, N)
    gausP = gaus2(N, x, y)
    show2(gausP)
    gausffft2 = myfft2(gausP, M, N, hx, hy)
    show2(gausffft2)
    myfx2 = myf2(N, x, y)
    show2(myfx2)
    myfxffft2 = myfft2(myfx2, M, N, hx, hy)
    show2(myfxffft2)

if __name__ == '__main__':
    main1()
    main2()