""" Singly Diagonally Implicit Runge-Kutta Method """

import numpy as np
import sympy as sp


def calculate_value(problem_data):
    calculate_value.title = 'Singly Diagonally Implicit Runge-Kutta Method'

    def set_gamma(trigger):
        if trigger == 'Plus':
            return (3 + 3 ** (1 / 2)) / 6
        else:
            return (3 - 3 ** (1 / 2)) / 6

    gamma = set_gamma('Plus')

    number_of_equations = problem_data["number_of_equations"]
    M = problem_data["M"]
    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_x = current_value[0]
    current_y = current_value[1:]
    step = problem_data["step"]

    approximate_solution = f(current_value)

    def calculate_k():
        # first stage
        k1 = np.array([sp.symbols('k1_' + str(i + 1)) for i in range(number_of_equations)])

        args = np.concatenate(([current_x + gamma * step], current_y + gamma * step * k1))

        intermediate_f = f(args)

        system = [sp.Eq(np.sum(M[i] * k1), intermediate_f[i])
                  for i in range(number_of_equations)]

        k1 = np.array(sp.nsolve(system, k1, approximate_solution))[:, 0]

        # second stage
        k2 = np.array([sp.symbols('k2_' + str(i + 1)) for i in range(number_of_equations)])

        args = np.concatenate(([current_x + (1 - gamma) * step],
                               current_y + (1 - 2 * gamma) * step * k1 + gamma * step * k2))

        intermediate_f = f(args)

        system = [sp.Eq(np.sum(M[i] * k2), intermediate_f[i])
                  for i in range(number_of_equations)]

        k2 = np.array(sp.nsolve(system, k2, approximate_solution))[:, 0]

        return k1, k2

    k1, k2 = calculate_k()

    next_y = current_y + step * (0.5 * k1 + 0.5 * k2)

    return next_y
