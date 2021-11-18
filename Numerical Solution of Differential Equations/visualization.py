import matplotlib.pyplot as plt
import pathlib


def visualize_solution(visualization_data, file_path):
    x = visualization_data["x"]
    x_name = visualization_data["x_name"]
    y = visualization_data["y"]
    y_name = visualization_data["y_name"]
    step = visualization_data["step"]
    title = visualization_data["title"]

    plt.plot(x, y)
    plt.title(title)
    x_label = ''.join([x_name, ' (step = ', str(step), ')'])
    plt.xlabel(x_label)
    plt.ylabel(y_name)

    # plt.savefig(file_path)
    plt.show()
