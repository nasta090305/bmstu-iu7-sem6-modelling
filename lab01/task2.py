import numpy as np
from matplotlib import pyplot as plt
import sympy
import math

def func(u, x):
    return 2 * x * u + u ** 3

def picar_alg(min_x, max_x, h, u0, n, f):
    x_symb = sympy.Symbol('x')
    u = list(0 for _ in range(n + 1))
    u[0] = u0
    n_x = math.ceil((max_x - min_x) / h) + 1
    x_arr = np.linspace(min_x, max_x, n_x)
    res = [[] for _ in range(n)]
    for i in range(n):
        for x in x_arr:
            u[i + 1] = u[0] + sympy.integrate(f(x_symb, u[i]), (x_symb, 0, x_symb))
            res[i].append(float(u[i + 1].evalf(subs={x_symb: x})))
    return np.array(res)

min_u = 0
max_u = 1.5
h = 1e-2
x0 = 0.5
picar1, picar2, picar3, picar4 = picar_alg(min_u, max_u, h, x0, 4, func)
u_arr = np.linspace(min_u, max_u, math.ceil((max_u - min_u) / h) + 1)
analyt = []
for u in u_arr:
    analyt.append(-(u ** 2 + 1) / 2 + math.exp(u ** 2))

plt.plot(picar1, u_arr, label='Метод Пикара (1-е приближение)')
plt.plot(picar2, u_arr, label='Метод Пикара (2-е приближение)')
plt.plot(picar3, u_arr, label='Метод Пикара (3-е приближение)')
plt.plot(picar4, u_arr, label='Метод Пикара (4-е приближение)')
plt.plot(analyt, u_arr, label='Аналитическое решение')
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()