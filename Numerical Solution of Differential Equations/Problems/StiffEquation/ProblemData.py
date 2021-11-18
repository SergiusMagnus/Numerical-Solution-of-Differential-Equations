""" Stiff Equation """

import numpy as np

M = np.array([1.])


def get_number_of_equations():
    return 1


def SE(args):
    x, y = args

    f = -50 * (y - np.cos(x))

    equations = np.array([f])

    return equations


def get_initial_condition():
    y = 0
    return y


def get_problem_data():
    return {"number_of_equations": get_number_of_equations(),
            "M": M,
            "f": SE,
            "get_initial_condition": get_initial_condition}
