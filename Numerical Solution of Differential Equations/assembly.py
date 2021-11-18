import pathlib

from calculation import calculate_solution
from visualization import visualize_solution

from Methods.BackwardEulerMethod import calculate_value as Backward_Euler_Method
from Methods.ImplicitTrapezoidalRuleMethod import calculate_value as Implicit_Trapezoidal_Rule_Method

from Problems.TransistorAmplifier.ProblemData import get_problem_data as get_DAE_data


methods = {"BEM": Backward_Euler_Method,
           "ITRM": Implicit_Trapezoidal_Rule_Method}


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
