from os.path import join
import matplotlib.pyplot as plt


def visualize_logs(exp_path: str, log_file_name: str, metric_name: str, bottom: float,
                   top: float, color: str, title: str, plot_file_name: str):
    with open(join(exp_path, log_file_name), 'r') as log_file:
        data = [float(line.strip()) for line in log_file.readlines()]
        plt.ion()
        plt.clf()
        plt.plot(range(1, len(data) + 1), data, c=color)
        if bottom is not None and top is not None:
            plt.ylim(bottom, top)
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(title)
        plt.savefig(join(exp_path, f"{plot_file_name}.png"))
        plt.draw()
        plt.pause(0.001)