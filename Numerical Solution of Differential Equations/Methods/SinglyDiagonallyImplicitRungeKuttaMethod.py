""" Singly Diagonally Implicit Runge-Kutta Method """

import numpy as np
import sympy as smp


def set_gamma(trigger):
    if trigger == 'Plus':
        return (3 + 3 ** (1 / 2)) / 6
    else:
        return (3 - 3 ** (1 / 2)) / 6


gamma = set_gamma('Plus')

a = np.array([[gamma, 0.],
              [1 - 2 * gamma, gamma]])

c = np.sum(a, axis=1)

b = np.array([0.5, 0.5])

stages_number = b.size


def calculate_value(problem_data):
    calculate_value.title = 'Singly Diagonally Implicit Runge-Kutta Method'

    equations_number = problem_data["equations_number"]
    M = problem_data["M"]
    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_x = current_value[0]
    current_y = current_value[1:]
    step = problem_data["step"]

    approximate_solution = f(current_value)

    def calculate_k():
        k = np.array([smp.symbols('k' + str(i + 1) + '_' + str(j + 1))
                      for i in range(stages_number) for j in range(equations_number)]) \
            .reshape(stages_number, equations_number)

        for i in range(stages_number):
            args = np.concatenate(([current_x + c[i] * step],
                                   current_y + step *
                                   (np.array([a[i] * k[:, j] for j in range(equations_number)]))
                                   .sum(axis=1)))

            intermediate_f = f(args)

            system = [smp.Eq(np.sum(M[j] * k[i]), intermediate_f[j])
                      for j in range(equations_number)]

            k[i] = np.array(smp.nsolve(system, k[i], approximate_solution, verify=False))[:, 0]

        return k

    k = calculate_k()

    next_y = current_y + step * (np.array([b * k[:, i] for i in range(equations_number)])).sum(axis=1)

    return next_y
