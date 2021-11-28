""" Rosenbrock Method """

import numpy as np
import sympy as smp

alfa = np.array([[1., 0.],
                 [0.5, 0.5]])

gamma = np.array([[1., 0.],
                  [-1., 1.]])

b = np.array([-0.5, 1.5])

alfa_sum = np.array([np.sum(alfa[i][:i]) for i in range(len(alfa[0]))])

gamma_sum = np.sum(gamma, axis=1)

stages_number = b.size

jacobian = None


def calculate_value(problem_data):
    calculate_value.title = 'Rosenbrock Method'

    equations_number = problem_data["equations_number"]
    M = problem_data["M"]
    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_x = current_value[0]
    current_y = current_value[1:]
    step = problem_data["step"]

    global jacobian

    def calculate_jacobian():
        args = np.array([smp.symbols('arg_' + str(i + 1)) for i in range(equations_number + 1)])

        jacobian = smp.Matrix(f(args)).jacobian(args)

        args_value = current_value
        args_value_tuples = [(args[i], args_value[i]) for i in range(equations_number + 1)]

        jacobian_at_point = np.array(jacobian.subs(args_value_tuples))

        return jacobian_at_point

    if jacobian is None:
        jacobian = calculate_jacobian()

    approximate_solution = f(current_value)

    def calculate_k():
        k = np.array([smp.symbols('k' + str(i + 1) + '_' + str(j + 1))
                      for i in range(stages_number) for j in range(equations_number)]) \
                     .reshape(stages_number, equations_number)

        for i in range(stages_number):
            args = np.concatenate(([current_x + alfa_sum[i] * step],
                                   current_y +
                                   (np.array([alfa[i][:i] * k[:i, j] for j in range(equations_number)]))
                                   .sum(axis=1)))

            intermediate_f = f(args)

            system = [smp.Eq(np.sum(M[j] * k[i]), step * intermediate_f[j]
                             + gamma_sum[i] * (step ** 2) * jacobian[:, 0][j]
                             + step * np.sum(jacobian[:, 1:][j]) *
                             (np.sum(gamma[i] * k[:, j])))
                      for j in range(equations_number)]

            k[i] = np.array(smp.nsolve(system, k[i], approximate_solution))[:, 0]

        return k

    k = calculate_k()

    next_y = current_y + np.array([b * k[:, i] for i in range(equations_number)]).sum(axis=1)

    return next_y
