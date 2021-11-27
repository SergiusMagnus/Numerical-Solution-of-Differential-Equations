""" Euler Method """

def calculate_value(problem_data):
    calculate_value.title = 'Euler Method'

    f = problem_data["f"]
    current_value = problem_data["current_value"]
    current_y = current_value[1:]
    step = problem_data["step"]

    next_y = current_y + step * f(current_value)

    return next_y
