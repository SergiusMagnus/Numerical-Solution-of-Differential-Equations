""" Implicit Trapezoidal Rule Method """

import numpy as np
import sympy as sp


def calculate_value(problem_data):
    calculate_value.title = 'Implicit Trapezoidal Rule Method'

    equations_number = problem_data["equations_number"]
    M = problem_data["M"]
    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_y = approximate_solution = current_value[1:]
    step = problem_data["step"]
    next_x = problem_data["next_x"]

    next_y = np.array([sp.symbols('next_y' + str(i + 1)) for i in range(equations_number)])
    next_value = np.concatenate(([next_x], next_y))
    next_f = f(next_value)

    current_f = f(current_value)

    system = [sp.Eq(np.sum(M[i] * next_y),
                    np.sum(M[i] * current_y) + 0.5 * step * (current_f[i] + next_f[i]))
              for i in range(equations_number)]

    next_y = np.array(sp.nsolve(system, next_y, approximate_solution))[:, 0]

    return next_y
