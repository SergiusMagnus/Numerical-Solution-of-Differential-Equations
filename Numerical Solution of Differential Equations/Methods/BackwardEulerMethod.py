""" Backward Euler Method """

import numpy as np
import sympy as smp


def calculate_value(problem_data):
    calculate_value.title = 'Backward Euler Method'

    equations_number = problem_data["equations_number"]
    M = problem_data["M"]
    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_y = approximate_solution = current_value[1:]
    step = problem_data["step"]
    next_x = problem_data["next_x"]

    next_y = np.array([smp.symbols('next_y' + str(i + 1)) for i in range(equations_number)])
    next_value = np.concatenate(([next_x], next_y))
    next_f = f(next_value)

    system = [smp.Eq(np.sum(M[i] * next_y),
                     np.sum(M[i] * current_y) + step * next_f[i])
              for i in range(equations_number)]

    next_y = np.array(smp.nsolve(system, next_y, approximate_solution))[:, 0]

    return next_y
