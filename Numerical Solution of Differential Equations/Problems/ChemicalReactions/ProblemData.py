""" System of Differential Equations """

import numpy as np

M = np.array([[1., 0., 0.],
              [0., 1., 0.],
              [0., 0., 1.]])


def get_number_of_equations():
    return 3


def SDE(args):
    x, y1, y2, y3 = args

    f1 = -0.04 * y1 + 10**4 * y2 * y3
    f2 = 0.04 * y1 - 10**4 * y2 * y3 - 3 * 10**7 * y2**2
    f3 = 3 * 10**7 * y2**2

    equations = np.array([f1, f2, f3])

    return equations


def get_initial_condition():
    y1 = 1
    y2 = 0
    y3 = 0
    return y1, y2, y3


def get_problem_data():
    return {"number_of_equations": get_number_of_equations(),
            "M": M,
            "f": SDE,
            "get_initial_condition": get_initial_condition}
