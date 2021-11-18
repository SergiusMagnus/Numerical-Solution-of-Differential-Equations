""" Backward Euler Method """

import numpy as np
import sympy as sp


def calculate_value(problem_data):
    number_of_equations = problem_data["number_of_equations"]
    M = problem_data["M"]
    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_y = approximate_solution = current_value[1:]
    step = problem_data["step"]
    next_x = problem_data["next_x"]

    next_y = [sp.symbols('next_y' + str(i + 1)) for i in range(number_of_equations)]
    next_value = [next_x] + next_y
    next_f = f(next_value)

    system = [sp.Eq(np.sum(M[i] * next_y),
                    np.sum(M[i] * current_y) + step * next_f[i])
              for i in range(number_of_equations)]

    next_y = sp.nsolve(system, next_y, approximate_solution)

    return next_y
