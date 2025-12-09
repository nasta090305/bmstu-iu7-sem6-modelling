import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import bisect

# Физические константы и параметры
c = 3e10  # скорость света (см/с)
T0 = 10000  # температура в центре (K)
Tp = 2000  # температура на поверхности (K)
R = 0.35  # радиус цилиндра (см)
w = 4  # параметр распределения температуры

EPS = 1e-4 # точность метода стрельбы

# Данные для коэффициента поглощения
T_values = np.array([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
k_values_var1 = np.array([8.2e-3, 2.768e-2, 6.56e-2, 1.281e-1, 2.214e-1,
                          3.516e-1, 5.248e-1, 7.472e-1, 1.025])
k_values_var2 = np.array([1.6, 5.4, 12.8, 25, 43.2, 68.6, 102.4, 145.8, 200])

def T(r):
    """Температурное распределение в цилиндре"""
    return T0 + (Tp - T0) * ((r / R) ** w)

def k_interp(T, variant=1):
    """Интерполяция коэффициента поглощения в логарифмическом масштабе"""
    if variant == 1:
        k_log = interp1d(np.log(T_values), np.log(k_values_var1),
                         bounds_error=False, fill_value="extrapolate")
    else:
        k_log = interp1d(np.log(T_values), np.log(k_values_var2),
                         bounds_error=False, fill_value="extrapolate")
    return np.exp(k_log(np.log(T)))

def up(r):
    """Равновесная плотность энергии излучения (функция Планка)"""
    with np.errstate(over='ignore'):
        return 3.084e-4 / (np.exp(47990.0 / T(r)) - 1)


def ode_system(r, u, F):
    """Формулирует систему дифференциальных уравнений для метода Рунге-Кутта."""
    k_r = k_interp(T(r))  # Вычисление коэффициента поглощения
    du_dr = -F * 3 * k_r / c  # Производная u по r

    if r == 0:
        dF_dr = c * k_r * (up(r) - u) / 2
    else:
        dF_dr = c * k_r * (up(r) - u) - F / r  # Производная F по r

    return du_dr, dF_dr


def runge_kutta(u0, F0, h=1e-2):
    """Численное решение системы ОДУ методом Рунге-Кутта."""
    r_vals = np.arange(0, R + h, h)  # Создание массива значений r
    u_vals, F_vals = [u0], [F0]  # Начальные условия

    for r in r_vals[:-1]:
        u, F = u_vals[-1], F_vals[-1]  # Последние рассчитанные значения

        ##Вычисление коэффициентов метода Рунге-Кутта 2 порядка при alpha = 1
        k1, q1 = ode_system(r, u, F)
        k2, q2 = ode_system(r + h / 2, u + h * k1 / 2, F + h * q1 / 2)

        # # Итоговые значения на следующем шаге
        u_next = u + h * (k2)
        F_next = F + h * (q2)

        u_vals.append(u_next)
        F_vals.append(F_next)

    return r_vals, np.array(u_vals), np.array(F_vals)

def shooting_method(chi_min=0.01, chi_max=1.0):
    """Подбор оптимального начального условия u(0) методом половинного деления."""
    chi_mid = (chi_min + chi_max) / 2
    u0 = chi_min * up(0)
    _, u_vals, F_vals = runge_kutta(u0, 0)
    psi_min = F_vals[-1] - 0.39 * c * u_vals[-1]

    u0 = chi_max * up(0)
    _, u_vals, F_vals = runge_kutta(u0, 0)
    psi = F_vals[-1] - 0.39 * c * u_vals[-1]

    while True:
        chi_mid = (chi_min + chi_max) / 2       # Среднее значение chi
        u0 = chi_mid * up(0)                    # Начальное условие для u
        _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
        psi = F_vals[-1] - 0.39 * c * u_vals[-1]
        if abs(psi) < EPS:
            break
        if psi > 0:
            chi_min = chi_mid
        else:
            chi_max = chi_mid
    step = (chi_max - chi_min) / 100
    u0 = chi_min * up(0)  # Начальное условие для u
    _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
    psi = F_vals[-1] - 0.39 * c * u_vals[-1]
    while abs(psi) > EPS:
        chi_min += step
        u0 = chi_min * up(0)  # Начальное условие для u
        _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
        psi = F_vals[-1] - 0.39 * c * u_vals[-1]
    u0 = chi_max * up(0)  # Начальное условие для u
    _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
    psi = F_vals[-1] - 0.39 * c * u_vals[-1]
    while abs(psi) > EPS:
        chi_max -= step
        u0 = chi_max * up(0)  # Начальное условие для u
        _, u_vals, F_vals = runge_kutta(u0, 0)  # Решение системы
        psi = F_vals[-1] - 0.39 * c * u_vals[-1]
    print(f'chi_min = {chi_min}, chi_max = {chi_max}')
    return (chi_max + chi_min) / 2

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


### Исследование влияния параметров ###
def run_param_study(param_name, values, update_func):
    results = []
    original_value = globals()[param_name]

    for val in values:
        update_func(param_name, val)  # Изменяем параметр
        try:
            chi_opt = shooting_method()
            u0_opt = chi_opt * up(0)
            r_vals, u_vals, F_vals = runge_kutta(u0_opt, 0)
            results.append((r_vals.copy(), u_vals.copy(), F_vals.copy(), f'{param_name}={val}'))
        except:
            print(f"Ошибка для {param_name}={val}")

    globals()[param_name] = original_value  # Восстанавливаем
    return results


def update_param(name, value):
    globals()[name] = value


# Параметры для исследования
param_studies = {
    'k(T)': {
        'values': [0.5, 1.0, 2.0],
        'update': lambda _, scale: globals().__setitem__('k_values', original_k_values * scale),
        'original': globals()['k_values_var1'].copy()
    },
    'Tp': [1500, 2000, 2500],
    'T0': [8000, 10000, 12000],
    'w': [2, 4, 6],
    'R': [0.2, 0.35, 0.5]
}

# Создаем директорию для графиков
import os

os.makedirs('param_studies', exist_ok=True)

# Исследуем влияние k(T)
original_k_values = k_values_var1.copy()
results_k = []
for scale in [0.5, 1.0, 2.0]:
    k_values_var1 = original_k_values * scale
    chi_opt = shooting_method()
    u0_opt = chi_opt * up(0)
    r_vals, u_vals, F_vals = runge_kutta(u0_opt, 0)
    results_k.append((r_vals, u_vals, F_vals, f'k×{scale}'))
k_values_var1 = original_k_values

# Исследуем остальные параметры
for param in ['Tp', 'T0', 'w', 'R']:
    results = run_param_study(param, param_studies[param], update_param)

    # Рисуем графики
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    for r, u, F, label in results:
        plt.plot(r, u, label=label)
    plt.title(f'Влияние {param} на u(r)')
    plt.legend()

    plt.subplot(2, 1, 2)
    for r, u, F, label in results:
        plt.plot(r, F, label=label)
    plt.title(f'Влияние {param} на F(r)')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'param_studies/{param}_influence.png')
    plt.close()

# Графики для k(T)
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
for r, u, F, label in results_k:
    plt.plot(r, u, label=label)
plt.title('Влияние коэффициента поглощения k(T) на u(r)')
plt.legend()

plt.subplot(2, 1, 2)
for r, u, F, label in results_k:
    plt.plot(r, F, label=label)
plt.title('Влияние коэффициента поглощения k(T) на F(r)')
plt.legend()
plt.tight_layout()
plt.savefig('param_studies/k_influence.png')
plt.close()