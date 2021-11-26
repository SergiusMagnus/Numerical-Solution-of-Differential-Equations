""" Stiff Equation """

import numpy as np
import sympy as sp

M = np.array([1.])


def get_equations_number():
    return 1


def SE(args):
    x, y = args

    f = -50 * (y - sp.cos(x))

    equations = np.array([f])

    return equations


def get_initial_condition():
    y = 0
    return [y]


def get_problem_data():
    return {"equations_number": get_equations_number(),
            "M": M,
            "f": SE,
            "get_initial_condition": get_initial_condition}
