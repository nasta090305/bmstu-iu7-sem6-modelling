import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import bisect
from decimal import Decimal, getcontext

# Физические константы и параметры
c = Decimal(3e10)  # скорость света (см/с)
T0 = Decimal(10000)  # температура в центре (K)
Tp = Decimal(2000)  # температура на поверхности (K)
R = Decimal(0.35)  # радиус цилиндра (см)
w = Decimal(4)  # параметр распределения температуры

EPS = Decimal(1e-4) # точность метода стрельбы

# Данные для коэффициента поглощения
T_values = np.array([Decimal(i) for i in [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]])
k_values_var2 = np.array([Decimal(i) for i in [8.2e-3, 2.768e-2, 6.56e-2, 1.281e-1, 2.214e-1,
                          3.516e-1, 5.248e-1, 7.472e-1, 1.025]])
k_values_var1 = np.array([Decimal(i) for i in [1.6, 5.4, 12.8, 25, 43.2, 68.6, 102.4, 145.8, 200]])


def T(r):
    """Температурное распределение в цилиндре"""
    return Decimal(T0 + (Tp - T0) * ((r / R) ** w))

def k_interp(T, variant=1):
    """Интерполяция коэффициента поглощения в логарифмическом масштабе"""
    T_loc = float(T)
    T_values_loc = [float(i) for i in T_values]
    k_values_var1_loc = [float(i) for i in k_values_var1]
    k_values_var2_loc = [float(i) for i in k_values_var2]
    if variant == 1:
        k_log = interp1d(np.log(T_values_loc), np.log(k_values_var1_loc),
                         bounds_error=False, fill_value="extrapolate")
    else:
        k_log = interp1d(np.log(T_values_loc), np.log(k_values_var2_loc),
                         bounds_error=False, fill_value="extrapolate")
    return Decimal(np.exp(k_log(np.log(T_loc))))

def up(r):
    """Равновесная плотность энергии излучения (функция Планка)"""
    with np.errstate(over='ignore'):
        return Decimal(Decimal(3.084e-4) / (np.exp(Decimal(47990.0) / T(r)) - Decimal(1)))


def ode_system(r, u, F):
    """Формулирует систему дифференциальных уравнений для метода Рунге-Кутта."""
    k_r = k_interp(T(r))  # Вычисление коэффициента поглощения
    du_dr = -F * 3 * k_r / c  # Производная u по r

    if r == 0:
        dF_dr = c * k_r * (up(r) - u) / 2
    else:
        dF_dr = c * k_r * (up(r) - u) - F / r  # Производная F по r

    return du_dr, dF_dr


def runge_kutta(u0, F0, h=Decimal(1e-4)):
    """Численное решение системы ОДУ методом Рунге-Кутта."""
    r_vals = np.arange(0, R + h, h)  # Создание массива значений r
    u_vals, F_vals = [u0], [F0]  # Начальные условия

    for r in r_vals[:-1]:
        u, F = u_vals[-1], F_vals[-1]  # Последние рассчитанные значения

        # Вычисление коэффициентов метода Рунге-Кутта 4 порядка
        k1, q1 = ode_system(r, u, F)
        k2, q2 = ode_system(r + h / 2, u + h * k1 / 2, F + h * q1 / 2)
        k3, q3 = ode_system(r + h / 2, u + h * k2 / 2, F + h * q2 / 2)
        k4, q4 = ode_system(r + h, u + h * k3, F + h * q3)
        # Итоговые значения на следующем шаге
        u_next = u + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        F_next = F + (h / 6) * (q1 + 2 * q2 + 2 * q3 + q4)

        # # Вычисление коэффициентов метода Рунге-Кутта 2 порядка при alpha = 1
        # k1, q1 = ode_system(r, u, F)
        # k2, q2 = ode_system(r + h / 2, u + h * k1 / 2, F + h * q1 / 2)

        # # Итоговые значения на следующем шаге
        # u_next = u + h * (k2)
        # F_next = F + h * (q2)

        u_vals.append(u_next)
        F_vals.append(F_next)

    return r_vals, np.array(u_vals), np.array(F_vals)

def shooting_method(chi_min=Decimal(0.01), chi_max=Decimal(1.0)):
    """Подбор оптимального начального условия u(0) методом половинного деления."""
    chi_mid = (chi_min + chi_max) / 2
    u0 = chi_min * up(0)
    _, u_vals, F_vals = runge_kutta(u0, 0)
    psi_min = F_vals[-1] - Decimal(0.39) * c * u_vals[-1]

    u0 = chi_max * up(0)
    _, u_vals, F_vals = runge_kutta(u0, 0)
    psi = F_vals[-1] - Decimal(0.39) * c * u_vals[-1]

    while True:
        chi_mid = (chi_min + chi_max) / 2       # Среднее значение chi
        u0 = chi_mid * up(0)                    # Начальное условие для u
        _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
        psi = F_vals[-1] - Decimal(0.39) * c * u_vals[-1]
        print(chi_min, chi_max, psi)
        if abs(psi) < EPS:
            break
        if psi > 0:
            chi_min = chi_mid
        else:
            chi_max = chi_mid
    print("bin search end")
    step = (chi_max - chi_min) / 100
    u0 = chi_min * up(0)  # Начальное условие для u
    _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
    psi = F_vals[-1] - Decimal(0.39) * c * u_vals[-1]
    while abs(psi) > EPS:
        chi_min += step
        u0 = chi_min * up(0)  # Начальное условие для u
        _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
        psi = F_vals[-1] - Decimal(0.39) * c * u_vals[-1]
    u0 = chi_max * up(0)  # Начальное условие для u
    _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
    psi = F_vals[-1] - Decimal(0.39) * c * u_vals[-1]
    print("min search end")
    while abs(psi) > EPS:
        chi_max -= step
        u0 = chi_max * up(0)  # Начальное условие для u
        _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
        psi = F_vals[-1] - Decimal(0.39) * c * u_vals[-1]
    print("max search end")
    print(f'chi_min = {chi_min}, chi_max = {chi_max}')
    return (chi_max + chi_min) / 2

getcontext().prec = 100
chi_opt = shooting_method()  # Поиск оптимального параметра chi (методом стрельбы)
u0_opt = chi_opt * up(0)  # Вычисление оптимального начального условия
r_vals, u_vals, F_vals = runge_kutta(u0_opt, 0)  # Численное решение системы методом Рунге-Кутта

u_p_vals = [up(r) for r in r_vals]

# Построение графиков
plt.figure(figsize=(12, 12))

# График u(r)
plt.subplot(3, 1, 1)
plt.plot(r_vals, u_vals, 'b-', linewidth=2)
plt.title(f'Решение u(r) при chi = {chi_opt:.6f}')
plt.xlabel('r (см)')
plt.ylabel('u(r)')
plt.grid(True)

# График F(r)
plt.subplot(3, 1, 2)
plt.plot(r_vals, F_vals, 'r-', linewidth=2)
plt.title('Функция F(r)')
plt.xlabel('r (см)')
plt.ylabel('F(r)')
plt.grid(True)

# График u_p(r)
plt.subplot(3, 1, 3)
plt.plot(r_vals, u_p_vals, 'g-', linewidth=2)
plt.title('Функция u_p(r)')
plt.xlabel('r (см)')
plt.ylabel('u_p(r)')
plt.grid(True)

plt.tight_layout()
plt.show()