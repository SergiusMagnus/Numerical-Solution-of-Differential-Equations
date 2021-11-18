from calculation import calculate_solution
from visualization import visualize_solution

from Methods.BackwardEulerMethod import calculate_value as Backward_Euler_Method

from Problems.TransistorAmplifier.ProblemData import get_problem_data as get_DAE_data

methods = {"BEM": [Backward_Euler_Method, "Backward Euler Method"]}


def solve_DAE(start, end, step, method):
    x_data = {"start": start, "end": end, "step": step}
    x, y = calculate_solution(get_DAE_data(), methods[method][0], x_data)

    data_for_visualization = {"x": x,
                              "x_name": 't',
                              "y": y[4],
                              "y_name": 'U5',
                              "step": x_data["step"],
                              "title": methods[method][1]}
    visualize_solution(data_for_visualization, None)
