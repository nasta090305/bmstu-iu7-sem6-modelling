import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Метод Галеркина
x = sp.symbols('x')
C1, C2, C3 = sp.symbols('C1 C2 C3')

u0 = x ** 2 / 2
u1 = x * (x - 1) ** 2
u2 = x**2 * (x - 1) ** 2
u3 = x**3 * (x - 1) ** 2

# print(sp.diff(u3, x, 1))

y = u0 + C1 * u1 + C2 * u2 + C3 * u3  # Приближенное решение

# Вычисление оператора Ly = y'' - 2x*y' + 2y
Ly = sp.diff(y, x, 2) - 2 * x * sp.diff(y, x) + 2 * y
R = Ly - x  # Невязка

# Интегралы для метода Галеркина
eq1 = sp.integrate(R * u1, (x, 0, 1))
eq2 = sp.integrate(R * u2, (x, 0, 1))
eq3 = sp.integrate(R * u3, (x, 0, 1))

# print(eq1)
# print(eq2)
# print(eq3)
# Решение системы уравнений для C1 и C2
solution = sp.solve([eq1, eq2, eq3], (C1, C2, C3))
C1_val = solution[C1].evalf()
C2_val = solution[C2].evalf()
C3_val = solution[C3].evalf()

# Аналитическое решение
y_approx = y.subs({C1: C1_val, C2: C2_val, C3: C3_val})
y_approx_func = sp.lambdify(x, y_approx, 'numpy')

# Параметры сетки
N = 100
h = 1.0 / N

# Инициализация коэффициентов
A = np.zeros(N + 1)
B = np.zeros(N + 1)
C = np.zeros(N + 1)
D = np.zeros(N + 1)

B[0] = 1

# Заполнение коэффициентов для внутренних узлов (i=1 до N-1)
for i in range(1, N):
    x_i = i * h
    A[i] = 1 + x_i * h
    B[i] = -2 + 2 * h**2
    C[i] = 1 - x_i * h
    D[i] = h**2 * x_i

# Краевое условие на правом конце (i=N)
A[N] = -1  # коэффициент при u_{N-1}
B[N] = 1    # коэффициент при u_N
D[N] = h    # правая часть

# Метод прогонки
alpha = np.zeros(N + 1)
beta = np.zeros(N + 1)

# Прямой ход
# i=1
alpha[1] = -C[1] / B[1]
beta[1] = D[1] / B[1]

# i=2 до N-1
for i in range(2, N + 1):
    denominator = A[i] * alpha[i-1] + B[i]
    alpha[i] = -C[i] / denominator
    beta[i] = (D[i] - A[i] * beta[i-1]) / denominator

# Обратный ход
u = np.zeros(N + 1)
u[N] = (h + beta[N]) / (1 - alpha[N])

for i in range(N-1, 0, -1):
    u[i] = alpha[i] * u[i+1] + beta[i]

# Учет левого краевого условия
# u[0] = 0
x_vals = np.linspace(0, 1, N + 1)

# Построение графиков
x_plot = np.linspace(0, 1, 100)
y_galerkin = y_approx_func(x_plot)

plt.figure(figsize=(10, 6))
plt.plot(x_plot, y_galerkin, label='Метод Галеркина (n=3)')
plt.plot(x_vals, u, '--', label='Численное решение (прогонка)')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.title('Сравнение аналитического и численного решений')
plt.grid(True)
plt.show()