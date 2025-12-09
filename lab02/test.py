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

# Данные для коэффициента поглощения
T_values = np.array([2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])
k_values_var1 = np.array([8.2e-3, 2.768e-2, 6.56e-2, 1.281e-1, 2.214e-1,
                          3.516e-1, 5.248e-1, 7.472e-1, 1.025])
k_values_var2 = np.array([1.6, 5.4, 12.8, 25, 43.2, 68.6, 102.4, 145.8, 200])


def T_r(r):
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


def u_p(r):
    """Равновесная плотность энергии излучения (функция Планка)"""
    with np.errstate(over='ignore'):
        return 3.084e-4 / (np.exp(47990.0 / T_r(r)) - 1)


def system(r, U, variant):
    """Система ОДУ для метода Рунге-Кутты"""
    u, F = U
    k = k_interp(T_r(r), variant)

    # Уравнение для du/dr
    du_dr = - (3 * k * F) / c

    # Уравнение для dF/dr с обработкой особенности при r=0
    if np.isclose(r, 0):
        dF_dr = -c * k * (u - u_p(0))
    else:
        dF_dr = - (F / r) - c * k * (u - u_p(r))

    return np.array([du_dr, dF_dr])


def runge_kutta(U0, r_values, variant):
    """Метод Рунге-Кутты 4-го порядка"""
    U = np.zeros((len(r_values), 2))
    U[0] = U0
    h = r_values[1] - r_values[0]

    for i in range(1, len(r_values)):
        r = r_values[i - 1]
        k1 = h * system(r, U[i - 1], variant)
        k2 = h * system(r + h / 2, U[i - 1] + k1 / 2, variant)
        k3 = h * system(r + h / 2, U[i - 1] + k2 / 2, variant)
        k4 = h * system(r + h, U[i - 1] + k3, variant)

        U[i] = U[i - 1] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return U


def boundary_condition_error(u0_guess, r_values, variant):
    """Функция для проверки граничного условия"""
    U = runge_kutta(np.array([u0_guess, 0]), r_values, variant)
    u_R, F_R = U[-1]
    return F_R - 0.39 * c * u_R

def solve_problem(variant=1, n_points=100):
    """Основная функция решения задачи"""
    r_values = np.linspace(0, R, 100)
    chi_range = np.linspace(0.1, 1, 20)
    chi = 0.1
    u0_opt = chi * u_p(0)
    start, end = 0.1 * u_p(0), 1.0 * u_p(0)
    #n_max = 100

    for _ in range(n_points):
        mid = (start + end) / 2
        #dif = shooting_method(r_values, variant, np.array([mid, 0]))
        dif = boundary_condition_error(mid, r_values, variant)
        if abs(dif) < 1e-8:
            u0_opt = mid
            break
        elif dif > 0:
            start = mid
        else:
            end = mid
    U = runge_kutta(np.array([u0_opt, 0]), r_values, variant)

    return r_values, U, u0_opt


def plot_separate_u_and_up(r_values, U):
    """Отдельные графики для u(r) и u_p(r)"""
    plt.figure(figsize=(12, 8))
    # График u(r) и u_p(r)
    plt.subplot(2, 2, 1)
    plt.plot(r_values, U[:, 0], label=f'u(r)', color='blue')
    plt.plot(r_values, [u_p(r) for r in r_values], label=f'up(r)', color='red', linestyle='--')
    #plt.yscale('log', base=10)  # Логарифмическая шкала с основанием 2
    plt.xlabel('Радиус r, см', fontsize=12)
    plt.ylabel('Значения, Дж/см³', fontsize=12)
    plt.title('Объемная плотность энергии u(r) и функция Планка up(r)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, R])

    # Дополнительные графики (F(r) и k(r))
    plt.subplot(2, 2, 2)
    plt.plot(r_values, U[:, 1], 'g-', linewidth=2)
    plt.xlabel('Радиус r, см', fontsize=12)
    plt.ylabel('Поток излучения F(r), Вт/см²', fontsize=12)
    plt.title('Распределение потока излучения F(r)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0, R])

    plt.subplot(2, 2, 3)
    plt.semilogy(r_values, [k_interp(T_r(r)) for r in r_values], 'm-', linewidth=2)
    plt.xlabel('Радиус r, см', fontsize=12)
    plt.ylabel('Коэффициент поглощения k(r), 1/см', fontsize=12)
    plt.title('Распределение коэффициента поглощения k(r)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, which='both')
    plt.xlim([0, R])

    plt.tight_layout()
    plt.show()


# Решение задачи
r_values, U, u0_opt = solve_problem(variant=1)
print(f"Оптимальное начальное значение u(0): {u0_opt:.6e}")
print(f"Отношение u(0)/u_p(0): {u0_opt / u_p(0):.6f}")

# Построение отдельных графиков
plot_separate_u_and_up(r_values, U)


