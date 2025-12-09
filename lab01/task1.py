import math

import numpy as np
from matplotlib import pyplot as plt
from math import cos, ceil

min_x = 0
max_x = 6
h = 1e-2
n_x = ceil((max_x - min_x) / h) + 1
x = np.linspace(min_x, max_x, n_x)
analit_y = np.empty_like(x)
exp_y = np.empty_like(x)
euler_y = np.empty_like(x)
euler_y1 = np.empty_like(x)
euler_y[0] = 2
euler_y1[0] = -0.4

for i in range(len(x)):
    exp_y[i] = (1 - 0.5 * x[i]
                + 1 / 24 * (x[i] ** 2)
                - 1 / 720 * (x[i] ** 3))

    analit_y[i] = cos(x[i] ** 0.5)

    if i != 0:
        if i != 1:
            euler_y1[i] = euler_y1[i - 1] + h * ((-2 * euler_y1[i - 1] - euler_y[i - 1]) / (4 * x[i - 1]))
        else:
            euler_y1[i] = -0.4
        euler_y[i] = euler_y[i - 1] + h * euler_y1[i - 1]

#plt.plot(x, exp_y, label='Разложение в ряд')
#plt.plot(x, analit_y, label='Аналитическое решение')
plt.plot(x, euler_y, label='Метод Эйлера')
plt.legend(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
