import numpy as np


def calculate_solution(problem_data, method, x_data):
    equations_number = problem_data["equations_number"]
    get_initial_condition = problem_data["get_initial_condition"]

    start_x = x_data["start"]
    end_x = x_data["end"]
    step_x = x_data["step"]

    x = np.arange(start_x, end_x, step_x)
    y = np.zeros((equations_number, len(x)))
    args = np.zeros(equations_number + 1)

    y[:, 0] = get_initial_condition()

    for i in range(len(x) - 1):
        args[0] = x[i]
        args[1:] = y[:, i]

        problem_data_for_method = {"equations_number": equations_number,
                                   "M": problem_data["M"],
                                   "f": problem_data["f"],
                                   "current_value": args,
                                   "step": step_x,
                                   "next_x": x[i + 1]}
        next_y = method(problem_data_for_method)

        y[:, i + 1] = next_y

    return x, y
