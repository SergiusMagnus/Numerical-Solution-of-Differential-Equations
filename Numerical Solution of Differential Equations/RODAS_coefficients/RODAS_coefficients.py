import pathlib
import numpy as np
import sympy as smp

gamma = 0.25
s = 6
b = np.array([np.nan for i in range(s)])
alfa_i = np.array([np.nan for i in range(s)])
alfa_ij = np.array([np.nan for i in range(s * s)]).reshape(s, s)
beta_i = np.array([np.nan for i in range(s)])
beta_ij = np.array([np.nan for i in range(s * s)]).reshape(s, s)
omega_ij = np.array([np.nan for i in range(s * s)]).reshape(s, s)
gamma_ij = np.array([np.nan for i in range(s * s)]).reshape(s, s)


def set_initial_values():
    for i in range(s):
        for j in range(i, s):
            alfa_ij[i, j] = 0.
            if j != s:
                beta_ij[i, j] = 0.

    alfa_i[0] = 0.
    alfa_i[1] = 0.386
    alfa_i[2] = 0.21
    alfa_i[3] = 0.63
    alfa_i[4] = 1.
    alfa_i[5] = 1.

    alfa_ij[1, 0] = alfa_i[1]

    beta_i[0] = 0.
    beta_i[1] = 0.0317
    beta_i[2] = 0.0635
    beta_i[3] = 0.3438

    beta_ij[1, 0] = beta_i[1]
    for i in range(s):
        beta_ij[i, i] = gamma


def step_1():
    b[5] = gamma

    # b0, b1, b2, b3, b4
    b0, b1, b2, b3, b4 = smp.symbols("b0, b1, b2, b3, b4")

    eqns = [b0 + b1 + b2 + b3 + b4 + b[5] - 1,
            b1 * beta_i[1] + b2 * beta_i[2] + b3 * beta_i[3] + (b4 + b[5]) * (1 - gamma) - 1/2 + gamma,
            b1 * alfa_i[1] ** 2 + b2 * alfa_i[2] ** 2 + b3 * alfa_i[3] ** 2 + b4 + b[5] - 1/3,
            b1 * alfa_i[1] ** 3 + b2 * alfa_i[2] ** 3 + b3 * alfa_i[3] ** 3 + b4 + b[5] - 1/4,
            b0 * alfa_i[0] ** 4 + b1 * alfa_i[1] ** 4 + b2 * alfa_i[2] ** 4 +
            b3 * alfa_i[3] ** 4 + b4 * alfa_i[4] ** 4 + b[5] * alfa_i[5] ** 4 - 1/5]

    b[:-1] = np.array(*smp.linsolve(eqns, b0, b1, b2, b3, b4))

    # beta5i
    beta_ij[5] = b

    # beta5
    beta_i[5] = beta_ij[5, :-1].sum()


def step_2():
    # b2beta21+b3beta31, b3beta32
    b2beta21b3beta31, b3beta32 = smp.symbols("b2beta21b3beta31, b3beta32")

    eqns = [b2beta21b3beta31 * beta_i[1] + b3beta32 * beta_i[2] + b[3] * beta_ij[3, 3] * beta_i[3] +
            (b[4] + b[5]) * (1/2 - 2 * gamma + gamma ** 2) - 1/6 + gamma - gamma ** 2,
            b2beta21b3beta31 * alfa_i[1] ** 2 + b3beta32 * alfa_i[2] ** 2 + b[3] * beta_ij[3, 3] * alfa_i[3] ** 2 +
            (b[4] + b[5]) * (1/3 - gamma) - 1/12 + gamma/3]

    b2beta21b3beta31, b3beta32 = np.array(*smp.linsolve(eqns, b2beta21b3beta31, b3beta32))

    # beta21
    beta21 = smp.symbols("beta21")

    eqns = [beta21 * b3beta32 * beta_i[1] + (b[4] + b[5]) * (1/6 - 3/2 * gamma + 3 * gamma ** 2 - gamma ** 3) -
            1/24 + gamma/2 - 3/2 * gamma ** 2 + gamma ** 3]

    beta_ij[2, 1] = np.array(*smp.linsolve(eqns, beta21))[0]

    # beta31, beta32
    beta_ij[3, 2] = b3beta32 / b[3]
    beta_ij[3, 1] = (b2beta21b3beta31 - b[2] * beta_ij[2, 1]) / b[3]

    # beta20, beta30
    beta_ij[2, 0] = beta_i[2] - beta_ij[2, 1]
    beta_ij[3, 0] = beta_i[3] - beta_ij[3, 1] - beta_ij[3, 2]


def step_3():
    alfa_ij[5, 4] = gamma
    # alfa51, alfa52, alfa53
    alfa51, alfa52, alfa53 = smp.symbols("alfa51, alfa52, alfa53")

    eqns = [alfa51 * beta_i[1] + alfa52 * beta_i[2] + alfa53 * beta_i[3] - 1/2 + 2 * gamma - gamma ** 2,
            alfa51 * alfa_i[1] ** 2 + alfa52 * alfa_i[2] ** 2 + alfa53 * alfa_i[3] ** 2 - 1/3 + gamma,
            alfa52 * beta_ij[2, 1] * beta_i[1] + alfa53 * (beta_ij[3, 1:4] * beta_i[1:4]).sum() -
            1/6 + 3/2 * gamma - 3 * gamma ** 2 + gamma ** 3]

    alfa_ij[5, 1:4] = np.array(*smp.linsolve(eqns, alfa51, alfa52, alfa53))

    # alfa50
    alfa_ij[5, 0] = alfa_i[5] - alfa_ij[5, 1:5].sum()

    # beta4j, beta4
    beta_ij[4, :4] = alfa_ij[5, :4]
    beta_i[4] = beta_ij[4, :-2].sum()

    # omega_ij
    global omega_ij
    omega_ij = np.linalg.inv(beta_ij)


def step_4():
    # alfa40, alfa41, alfa42, alfa43
    alfa40, alfa41, alfa42, alfa43 = smp.symbols("alfa40, alfa41, alfa42, alfa43")

    eqns = [alfa41 * beta_i[1] + alfa42 * beta_i[2] + alfa43 * beta_i[3] - 1/2 + gamma,
            alfa41 * omega_ij[1, 1] * alfa_i[1] ** 2 +
            alfa42 * (omega_ij[2, 1] * alfa_i[1] ** 2 + omega_ij[2, 2] * alfa_i[2] ** 2) +
            alfa43 *
            (omega_ij[3, 1] * alfa_i[1] ** 2 + omega_ij[3, 2] * alfa_i[2] ** 2 + omega_ij[3, 3] * alfa_i[3] ** 2) - 1,
            alfa40 + alfa41 + alfa42 + alfa43 - 1,
            alfa40 * omega_ij[0, :1].sum() + alfa41 * omega_ij[1, :2].sum() + alfa42 * omega_ij[2, :3].sum() +
            alfa43 * omega_ij[3, :4].sum() - 1]

    alfa_ij[4, 0:4] = np.array(*smp.linsolve(eqns, alfa40, alfa41, alfa42, alfa43))


def step_5():
    # alfa21, alfa31, alfa32
    alfa21, alfa31, alfa32 = smp.symbols("alfa21, alfa31, alfa32")

    eqns = [b[2] * alfa_i[2] * alfa21 * beta_i[1] + b[3] * alfa_i[3] * (alfa31 * beta_i[1] + alfa32 * beta_i[2]) +
            (b[4] + b[5]) * (1/2 - gamma) - 1/8 + gamma/3,
            b[2] * alfa_i[2] * alfa21 * omega_ij[1, 1] * alfa_i[1] ** 2 +
            b[3] * alfa_i[3] * (alfa31 * (omega_ij[1, 1] * alfa_i[1] ** 2)
                                + alfa32 * (omega_ij[2, 1] * alfa_i[1] ** 2 + omega_ij[2, 2] * alfa_i[2] ** 2)) +
            (b[4] + b[5]) - 1/4,
            b[2] * alfa_i[2] ** 2 * alfa21 * beta_i[1] +
            b[3] * alfa_i[3] ** 2 * (alfa31 * beta_i[1] + alfa32 * beta_i[2]) +
            b[4] * alfa_i[4] ** 2 * (alfa_ij[4] * beta_i).sum() +
            b[5] * alfa_i[5] ** 2 * (alfa_ij[5] * beta_i).sum() - 1/10 + gamma/4]

    alfa_ij[2, 1], alfa_ij[3, 1], alfa_ij[3, 2] = np.array(*smp.linsolve(eqns, alfa21, alfa31, alfa32))

    # alfa20, alfa30
    alfa_ij[2, 0] = alfa_i[2] - alfa_ij[2, 1]
    alfa_ij[3, 0] = alfa_i[3] - alfa_ij[3, 1] - alfa_ij[3, 2]

    # gamma_ij
    global gamma_ij
    gamma_ij = beta_ij - alfa_ij


def save_RODAS_coefficients():
    path = pathlib.Path('./Numerical Solution of Differential Equations/RODAS_coefficients/alfa.csv')
    np.savetxt(path, alfa_ij)
    path = pathlib.Path('./Numerical Solution of Differential Equations/RODAS_coefficients/gamma.csv')
    np.savetxt(path, gamma_ij)
    path = pathlib.Path('./Numerical Solution of Differential Equations/RODAS_coefficients/b.csv')
    np.savetxt(path, b)


def checking():
    inv_gamma_ij = np.linalg.inv(gamma_ij)
    # for i in range(s):
    #     for j in range(s):
    #         if j > i:
    #             inv_gamma_ij[i, j] = 0.

    new_alfa_ij = np.dot(alfa_ij, inv_gamma_ij)
    c_ij = np.diag(np.diag(gamma_ij) ** -1) - inv_gamma_ij

    gamma_i = np.array([gamma_ij[:4].sum(axis=1)])


def calculate_RODAS_coefficients():
    set_initial_values()
    step_1()
    step_2()
    step_3()
    step_4()
    step_5()
    # save_RODAS_coefficients()
    # checking()

