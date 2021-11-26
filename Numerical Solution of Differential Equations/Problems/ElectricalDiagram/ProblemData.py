""" System of Differential Equations """

import numpy as np

m = 500

M = np.array([[1., 0.],
              [0., 1.]])


def get_equations_number():
    return 2


def SDE(args):
    x, y1, y2 = args

    f1 = y2
    f2 = m ** 2 * ((1 - y1 ** 2) * y2 - y1)

    equations = np.array([f1, f2])

    return equations


def get_initial_condition():
    y1 = 2
    y2 = 0

    return y1, y2


def get_problem_data():
    return {"equations_number": get_equations_number(),
            "M": M,
            "f": SDE,
            "get_initial_condition": get_initial_condition}
