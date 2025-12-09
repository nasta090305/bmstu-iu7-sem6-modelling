import constants
import numpy as np
import math
from functools import cache

import matplotlib.pyplot as plt


def log_interp(T, T_table, Y_table):
    T = max(T, T_table[0])
    log_T = np.log(T)
    log_T_table = np.log(T_table)
    log_Y_table = np.log(Y_table)

    if log_T > log_T_table[-1] and len(T_table) > 1:
        k = (log_Y_table[-1] - log_Y_table[-2]) / (log_T_table[-1] - log_T_table[-2])
        log_Y = k * (log_T - log_T_table[-1]) + log_Y_table[-1]
    else:
        log_Y = np.interp(log_T, log_T_table, log_Y_table)
    return np.exp(log_Y)

@cache
def f_sigma(T):
    return log_interp(T, constants.T_table, constants.sigma_table)

@cache
def f_lambda(T):
    return log_interp(T, constants.T_table, constants.lambda_table)

@cache
def f_c(T):
    return np.interp(T, constants.T_table, constants.c_table)

def f_k(T):
    # print(len(constants.T_table), len(constants.k_abs_table))
    return log_interp(T, constants.T_k_table, constants.k_abs_table)

def f_I(t) -> float:
    # return constants.I_MAX
    return (constants.I_MAX / constants.TOK_MAX) * t * math.exp(1 - t / constants.TOK_MAX)

def f_E(r, T, t) -> float:
    sum_E = 0.0
    h = r[1] - r[0]
    for i in range(len(r) - 1):
        r_mid = (r[i] + r[i + 1]) / 2.0
        T_mid = (T[i] + T[i + 1]) / 2.0
        sum_E += f_sigma(T_mid) * r_mid * h
    return f_I(t) / (2 * np.pi * sum_E)

def f_u_p(T) -> float:
    T_safe = max(T, 300.0)
    return 3.084 * 1e-4 / (math.exp(4.799 * 1e4 / T_safe) - 1)

def f_q(R, u, T):
    return [constants.c * f_k(T[i]) * (f_u_p(T[i]) - u[i]) for i in range(len(R))]

def edge_1(r, R):
    return constants.T_0 + (constants.T_W - constants.T_0) * (r / R) ** constants.p

def edge_3(k, T):
    return -(1 + k * 0.39 / (3 * f_k(T[-1])))

##########################################################################################

def thomas_algo(A, B, C, F) -> list[float]:
    n = len(F)
    x = [0.0] * n
    alpha = [0.0] * n
    beta = [0.0] * n

    # forward pass
    alpha[0] = -C[0] / B[0]
    beta[0] = F[0] / B[0]
    for i in range(n - 1):
        alpha[i + 1] = -C[i] / (A[i] * alpha[i] + B[i])
        beta[i + 1] = (F[i] - A[i] * beta[i]) / (A[i] * alpha[i] + B[i])

    # backward pass
    x[-1] = (F[-1] - A[-1] * beta[-1]) / (B[-1] + A[-1] * alpha[-1])
    for i in range(n - 1, 0, -1):
        x[i - 1] = alpha[i] * x[i] + beta[i]
    return np.array(x)

def solve_T(R, T, Q, E_cur) -> list[float]:
    n = len(R)
    A = [0.0] * n
    B = [0.0] * n
    C = [0.0] * n
    D = [0.0] * n

    B[0] = 1.0
    C[0] = -1.0
    r_step = R[1] - R[0]
    t_step = constants.t_max / (constants.M + 1)
    for i in range(1, len(R) - 1):
        r_i_left = (R[i - 1] + R[i]) / 2
        r_i_right = (R[i] + R[i + 1]) / 2
        T_i_left = (T[i - 1] + T[i]) / 2
        T_i_right = (T[i] + T[i + 1]) / 2

        A[i] = r_i_left * f_lambda(T_i_left) / r_step
        B[i] = -(f_c(T[i]) * R[i] * r_step / t_step + r_i_right * f_lambda(T_i_right) / r_step + r_i_left * f_lambda(T_i_left) / r_step)
        C[i] = r_i_right * f_lambda(T_i_right) / r_step
        D[i] = -R[i] * r_step * (f_c(T[i]) * T[i] / t_step + f_sigma(T[i]) * E_cur * E_cur - Q[i])

    A[-1] = 0
    B[-1] = 1
    C[-1] = 0
    D[-1] = constants.T_W

    return thomas_algo(A, B, C, D)


def solve_U(R, T) -> list[float]:
    n = len(R)
    A = [0.0] * n
    B = [0.0] * n
    C = [0.0] * n
    D = [0.0] * n
    B[0] = -1.0

    r_step = R[1] - R[0]
    for i in range(1, len(R) - 1):
        r_i_left = (R[i - 1] + R[i]) / 2.0
        r_i_right = (R[i] + R[i + 1]) / 2.0
        T_i_left = (T[i - 1] + T[i]) / 2.0
        T_i_right = (T[i] + T[i + 1]) / 2.0

        A[i] = r_i_left / (f_k(T_i_left) * r_step)
        B[i] = -(r_i_right / (f_k(T_i_right) * r_step) + r_i_left / (f_k(T_i_left) * r_step) + 3 * r_step * R[i] * f_k(T[i]))
        C[i] = r_i_right / (f_k(T_i_right) * r_step)
        D[i] = -3 * r_step * R[i] * f_k(T[i]) * f_u_p(T[i])

    A[-1] = -1
    B[-1] = edge_3(r_step, T)
    C[-1] = 0
    D[-1] = 0

    return thomas_algo(A, B, C, D)

def solve(r_max, r_step, t_max, t_step, iters, eps):
    R = [i for i in np.arange(0, r_max, r_step)]
    T = [[edge_1(r, r_max) for r in R]]
    U = []
    E = []
    for t in np.arange(0, t_max, t_step):
        T_cur = T[-1]
        for i in range(iters):
            E_cur = f_E(R, T_cur, t)
            U_cur = solve_U(R, T_cur)
            q_cur = f_q(R, U_cur, T_cur)
            T_next = solve_T(R, T_cur, q_cur, E_cur)
            for r in range(1, len(R)):
                # print(R[r], T_next[r] - T_cur[r])
                if (T_next[r] - T_cur[r]) / T_cur[r] > eps:
                    T_cur = T_next
                    break
            else:
                break
        U.append(U_cur)
        T.append(T_cur)
        E.append(E_cur)
    return R, T, U, E


if __name__ == '__main__':
    R, T, U, E = solve(constants.R, constants.step_r, constants.t_max, constants.step_t, constants.iters, constants.EPS)
    colors = plt.cm.plasma(np.linspace(0, 1, len(T)))
    for i in range(0, len(T), 50):
        plt.plot(R, T[i], color=colors[i], linewidth=2, label=f't = {(constants.step_t * i) * 1e6:.1f} мкс')
    plt.legend()
    plt.show()
