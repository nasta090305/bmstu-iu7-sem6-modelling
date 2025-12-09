import numpy as np
from matplotlib import pyplot as plt
import sympy
import math
from task2 import picar_alg

def func(u, x):
    return x + u**3

def find_xmax(min_x, max_x, h, u0):
    x = min_x
    prev_u = u0 - 10
    u = u0
    h = 1e-5
    while x < 1.5:
        prev_u = u
        u += h * func(u, x)
        x += h
    h = 1e-6
    while x < 1.647263:
        u += h * func(u, x)
        x += h
    h = 1e-10
    try:
        while x < max_x:
            prev_u = u
            u += h * func(u, x)
            x += h
    except:
        return x, prev_u
    return x, u

min_x = 0
u0 = 0
max_x, max_u = find_xmax(min_x, 2, 1e-2, u0)
h = 1e-2
n_x = math.ceil((max_x - min_x) / h) + 1
x_arr = np.linspace(min_x, max_x, n_x)
xmax, max_u = find_xmax(min_x, max_x, 1e-2, u0)
print("xmax = ", xmax, "max_u = ", max_u)
picar1, picar2, picar3, picar4 = picar_alg(min_x, max_x, h, u0, 4, func)
euler_y = []
x, u = min_x, u0
for i in range(n_x):
    u += h * func(u, x)
    euler_y.append(u)
    x += h

print("\nТаблица значений:")
print(f"{'x':<8}{'Пикар 1':<12}{'Пикар 2':<12}{'Пикар 3':<12}{'Пикар 4':<12}{'Эйлер':<12}")
for i in range(len(x_arr)):
    print(
        f"{x_arr[i]:<8.4f}{picar1[i]:<12.6f}{picar2[i]:<12.6f}{picar3[i]:<12.6f}{picar4[i]:<12.6f}{euler_y[i]:<12.6f}")

plt.plot(x_arr, picar1, label='Метод Пикара (1-е приближение)')
plt.plot(x_arr, picar2, label='Метод Пикара (2-е приближение)')
plt.plot(x_arr, picar3, label='Метод Пикара (3-е приближение)')
plt.plot(x_arr, picar4, label='Метод Пикара (4-е приближение)')
plt.plot(x_arr, euler_y, label='Метод Эйлера')
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()