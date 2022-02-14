""" Rosenbrock Method """

import numpy as np
import sympy as smp
import pathlib

# alfa = np.array([[-1., 0.],
#                  [1.5, -0.5]])
#
# gamma = np.array([[1., 0.],
#                   [-1., 1.]])
#
# b = np.array([0.5, 0.5])


# 3
# alfa = np.array([[-0.4714, 0.],
#                  [1.4714, -0.4714]])
#
# gamma = np.array([[0.80474, 0.],
#                   [-0.80474, 0.80474]])
#
# b = np.array([0.66666, 0.33334])

# 2
# alfa = np.array([[1/4, 0.],
#                  [3/4, 1/4]])
#
# gamma = np.array([[0.0428932, 0.],
#                   [-0.0428932, 0.0428932]])
#
# b = np.array([0.707107, 0.292893])

path = pathlib.Path('./Numerical Solution of Differential Equations/RODAS_coefficients/alfa.csv')
alfa = np.loadtxt(path)
path = pathlib.Path('./Numerical Solution of Differential Equations/RODAS_coefficients/gamma.csv')
gamma = np.loadtxt(path)
path = pathlib.Path('./Numerical Solution of Differential Equations/RODAS_coefficients/b.csv')
b = np.loadtxt(path)


alfa_sum = np.array([np.sum(alfa[i][:i]) for i in range(len(alfa[0]))])

gamma_sum = np.sum(gamma, axis=1)

stages_number = b.size

symbolic_jacobian = None


def calculate_value(problem_data):
    calculate_value.title = 'Rosenbrock Method'

    equations_number = problem_data["equations_number"]
    M = problem_data["M"]
    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_x = current_value[0]
    current_y = current_value[1:]
    step = problem_data["step"]
    need_to_calculate_jacobian = problem_data["need_to_calculate_jacobian"]

    global symbolic_jacobian

    args = np.array([smp.symbols('arg_' + str(i + 1)) for i in range(equations_number + 1)])

    if need_to_calculate_jacobian:
        symbolic_jacobian = smp.Matrix(f(args)).jacobian(args)

    args_value = current_value
    args_value_tuples = zip(args, args_value)

    jacobian_at_point = np.array(symbolic_jacobian.subs(args_value_tuples))

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

            gamma_times_k_sum = np.dot(gamma[i], k)

            system = [smp.Eq(np.sum(M[j] * k[i]), step * intermediate_f[j]
                             + gamma_sum[i] * (step ** 2) * jacobian_at_point[:, 0][j]
                             + step * np.sum(jacobian_at_point[j, 1:] * gamma_times_k_sum))
                      for j in range(equations_number)]

            k[i] = np.array(*smp.linsolve(system, *k[i]))

        return k

    k = calculate_k()

    next_y = current_y + np.array([b * k[:, i] for i in range(equations_number)]).sum(axis=1)

    return next_y
