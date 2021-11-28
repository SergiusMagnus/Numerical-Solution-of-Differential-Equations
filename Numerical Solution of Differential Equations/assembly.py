import pathlib

from calculation import calculate_solution
from visualization import visualize_solution

from Methods import Euler_Method, \
                    Runge_Kutta_Method, \
                    Backward_Euler_Method, \
                    Implicit_Trapezoidal_Rule_Method,\
                    Singly_Diagonally_Implicit_Runge_Kutta_Method, \
                    Rosenbrock_Method

from Problems import get_SE_data,\
                     get_SDE_1_data, \
                     get_SDE_2_data, \
                     get_DAE_data

methods = {"EM": Euler_Method,
           "RKM": Runge_Kutta_Method,
           "BEM": Backward_Euler_Method,
           "ITRM": Implicit_Trapezoidal_Rule_Method,
           "SDIRKM": Singly_Diagonally_Implicit_Runge_Kutta_Method,
           "ROSM": Rosenbrock_Method}


def solve_SE(start, end, step, method):
    x_data = {"start": start, "end": end, "step": step}
    x, y = calculate_solution(get_SE_data(), methods[method], x_data)

    data_for_visualization = {"x": x,
                              "x_name": 'x',
                              "y": y[0],
                              "y_name": 'y',
                              "step": x_data["step"],
                              "title": methods[method].title}

    path_to_directory = pathlib.Path('./Numerical Solution of Differential Equations/Problems/'
                                     'StiffEquation/Solutions')
    file_name = pathlib.Path(''.join([method, ' ', str(step), '.png']))
    file_path = path_to_directory / file_name
    visualize_solution(data_for_visualization, file_path)


def solve_SDE_1(start, end, step, method):
    x_data = {"start": start, "end": end, "step": step}
    x, y = calculate_solution(get_SDE_1_data(), methods[method], x_data)

    data_for_visualization = {"x": x,
                              "x_name": 'x',
                              "y": y[1],
                              "y_name": 'y2',
                              "step": x_data["step"],
                              "title": methods[method].title}

    path_to_directory = pathlib.Path('./Numerical Solution of Differential Equations/Problems/'
                                     'ChemicalReactions/Solutions')
    file_name = pathlib.Path(''.join([method, ' ', str(step), '.png']))
    file_path = path_to_directory / file_name
    visualize_solution(data_for_visualization, file_path)


def solve_SDE_2(start, end, step, method):
    x_data = {"start": start, "end": end, "step": step}
    x, y = calculate_solution(get_SDE_2_data(), methods[method], x_data)

    data_for_visualization = {"x": y[0],
                              "x_name": 'y1',
                              "y": y[1],
                              "y_name": 'y2',
                              "step": x_data["step"],
                              "title": methods[method].title}

    path_to_directory = pathlib.Path('./Numerical Solution of Differential Equations/Problems/'
                                     'ElectricalDiagram/Solutions')
    file_name = pathlib.Path(''.join([method, ' ', str(step), '.png']))
    file_path = path_to_directory / file_name
    visualize_solution(data_for_visualization, file_path)


def solve_DAE(start, end, step, method):
    x_data = {"start": start, "end": end, "step": step}
    x, y = calculate_solution(get_DAE_data(), methods[method], x_data)

    data_for_visualization = {"x": x,
                              "x_name": 't',
                              "y": y[4],
                              "y_name": 'U5',
                              "step": x_data["step"],
                              "title": methods[method].title}

    path_to_directory = pathlib.Path('./Numerical Solution of Differential Equations/Problems/'
                                     'TransistorAmplifier/Solutions')
    file_name = pathlib.Path(''.join([method, ' ', str(step), '.png']))
    file_path = path_to_directory / file_name
    visualize_solution(data_for_visualization, file_path)
