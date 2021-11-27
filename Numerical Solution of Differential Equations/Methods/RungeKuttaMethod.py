""" Runge-Kutta Method """

import numpy as np

a = np.array([[0., 0., 0., 0.],
              [0.5, 0., 0., 0.],
              [0., 0.5, 0., 0.],
              [0., 0., 1., 0.]])

c = np.sum(a, axis=1)

b = np.array([1/6, 1/3, 1/3, 1/6])

stages_number = b.size


def calculate_value(problem_data):
    calculate_value.title = 'Classical Runge-Kutta Method'

    equations_number = problem_data["equations_number"]
    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_x = current_value[0]
    current_y = current_value[1:]
    step = problem_data["step"]

    def calculate_k():
        k = np.zeros(stages_number * equations_number).reshape(stages_number, equations_number)

        for i in range(stages_number):
            args = np.concatenate(([current_x + c[i] * step],
                                   current_y + step *
                                   (np.array([a[i] * k[:, j] for j in range(equations_number)]))
                                   .sum(axis=1)))
            k[i] = f(args)

        return k

    k = calculate_k()

    next_y = current_y + step * (np.array([b * k[:, i] for i in range(equations_number)])).sum(axis=1)

    return next_y
