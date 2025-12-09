import numpy as np
from math import log, exp, ceil
import matplotlib.pyplot as plt
from functools import cache

c = 3e10  # см/с
k_data_1 = [
     8.200e-3,
     2.768e-2,
     6.560e-2,
     1.281e-1,
     2.214e-1,
     3.516e-1,
     5.248e-1,
     7.472e-1,
     1.025e0,
 ]

k_data_2 = [
    1.600e00,
    5.400e00,
    1.280e01,
    2.500e01,
    4.320e01,
    6.860e01,
    1.024e02,
    1.458e02,
    2.000e02,
]

T_data = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

# Параметры
T_w = 2000
T_0 = 1e4
R = 0.35
p_exp = 4

var = 1

T = lambda r: ((T_w - T_0) * (r / R) ** p_exp + T_0)

Up = lambda r: 3.084e-4 / (exp(4.799e4 / T(r)) - 1)

@cache
def k(r):
    xi = np.log(T_data)
    if var == 1:
        eta = np.log(k_data_1)
    elif var == 2:
        eta = np.log(k_data_2)

    x = np.log(T(r))

    idx = np.searchsorted(xi, x)
    if idx == 0:
        return np.exp(eta[0])
    if idx == len(xi):
        return np.exp(eta[-1])

    x0, x1 = xi[idx - 1], xi[idx]
    y0, y1 = eta[idx - 1], eta[idx]
    return np.exp(y0 + (x - x0) * (y1 - y0) / (x1 - x0))


def p_f(r):
    return 3 * k(r)


def f(r):
    return 3 * k(r) * Up(r)


def lambda_f(r):
    return 1 / k(r)


def kappa(r1, r2):
    return (2 * lambda_f(r1) * lambda_f(r2)) / (lambda_f(r1) + lambda_f(r2))


def V(zl, zr):
    return (zr ** 2 - zl ** 2) / 2


def u_thomas(steps):
    N = steps - 1
    z = np.linspace(0, 1, steps)
    h = z[1] - z[0]

    ak = np.zeros(N + 1)
    bk = np.zeros(N + 1)
    ck = np.zeros(N + 1)
    dk = np.zeros(N + 1)

    for i in range(1, N):
        zlh = (z[i - 1] + z[i]) / 2
        zrh = (z[i] + z[i + 1]) / 2

        ak[i] = zlh * kappa(z[i - 1] * R, z[i] * R) / (R ** 2 * h)
        ck[i] = zrh * kappa(z[i] * R, z[i + 1] * R) / (R ** 2 * h)
        bk[i] = ak[i] + ck[i] + p_f(z[i] * R) * V(zlh, zrh)
        dk[i] = f(z[i] * R) * V(zlh, zrh)

    # Прямой ход
    alpha = np.zeros(N + 1)
    beta = np.zeros(N + 1)

    alpha[1] = 1
    beta[1] = 0
    for i in range(1, N):
        denom = bk[i] - ak[i] * alpha[i]
        alpha[i + 1] = ck[i] / denom
        beta[i + 1] = (ak[i] * beta[i] + dk[i]) / denom

    # Обратный ход
    u = np.zeros(N + 1)

    zn = (z[N - 1] + z[N]) / 2
    Mn = (zn * kappa(z[N - 1] * R, z[N] * R)) / (R ** 2 * h)
    kN = - (z[N] * 3 * 0.39 / R + (zn * kappa(z[N - 1] * R, z[N] * R)) / (R ** 2 * h) + p_f(R) * z[N] * h / 2)
    pN = -f(R) * z[N] * h / 2
    u[N] = (pN - Mn * beta[N]) / (Mn * alpha[N] + kN)

    for i in range(N - 1, -1, -1):
        u[i] = alpha[i + 1] * u[i + 1] + beta[i + 1]

    F = np.zeros(N + 1)

    F[0] = 0
    for i in range(1, N):
        F[i] = - c / (3 * R) * ((u[i + 1] - u[i - 1]) / (2 * h)) * lambda_f(z[i] * R)
    F[N] = 0.39 * c * u[N]

    divF = [c * k(z[i] * R) * (Up(z[i] * R) - u[i]) for i in range(N + 1)]

    F_integr = np.zeros(N + 1)

    for i in range(1, N + 1):
        F_integr[i] = F_integr[i - 1] + R * ((divF[i - 1] * z[i - 1] + divF[i] * z[i]) / 2) * h

    F_integr = [0] + list(F_integr[1:] / z[1:])

    return u, F, F_integr, divF

import json
if False:
    import comparison_lab2
    # Получаем данные из lab2
    with open("lab_2.json", "w") as fout:
        r, u1, F1, up1, k1 = comparison_lab2.lab2()
        json.dump({"r": r, "u1": u1, "F1": F1, "up1": up1, "k1": k1}, fout)
else:
    with open("lab_2.json", "r") as fin:
        data = json.load(fin)
        r, u1, F1, up1, k1 = data["r"], data["u1"], data["F1"], data["up1"], data["k1"]

for i in range(len(r)):
    r[i] = r[i] / R  # Нормируем радиус

u2, F2, F_int, divF = u_thomas(len(r))

plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(r, F2, label='F(z) lab3')
plt.plot(r, F1, label='F(z) lab2', linestyle='--')
plt.xlabel('r/R')
plt.ylabel('F(z)')
plt.title('Поток излучения F(z)')
plt.legend()

# Второй подграфик: Объемная плотность энергии u(r)
plt.subplot(2, 2, 2)
plt.plot(r, u2, label='u(z) lab3')
plt.plot(r, u1, label='u(z) lab2', linestyle='--')
plt.xlabel('r/R')
plt.ylabel('u(z)')
plt.title('Объемная плотность энергии u(z)')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(r, F_int, label='Интегр. F(z)')
plt.xlabel('r/R')
plt.ylabel('F(z)')
plt.title('Поток излучения F(z) (интегр.)')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(r, divF, label='divF(z)')
plt.xlabel('r/R')
plt.ylabel('divF')
plt.title('Дивергенция потока излучения F(z)')
plt.legend()

plt.tight_layout()
plt.show()