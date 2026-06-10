import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_completed_deliveries_plot(
    completed_deliveries: list[int],
    path: Path,
    filename: str,
    save_plot_data: bool = False,
    window_size: int = 20,
    start_eps: float = None,
    epsilon_decay: float = None,
):
    filename = filename.removesuffix(".pth")
    path = path / f"{filename}_completed_deliveries_w{window_size}.png"
    n = len(completed_deliveries)
    completed_deliveries_sum = [0] * n
    for i in range(n):
        completed_deliveries_sum[i] = sum(
            completed_deliveries[i - window_size + 1 : i + 1]
        )
    plt.figure(figsize=(10, 6))
    x = range(n)
    plt.plot(x, completed_deliveries_sum)
    # if start_eps is not None and epsilon_decay is not None:
    #     y = [start_eps * (epsilon_decay**i) for i in x]
    #     plt.plot(x, y, label="eps")
    plt.title(f"Completed deliveries in last {window_size} episodes")
    plt.xlabel("episode")
    plt.ylabel("deliveries")
    plt.savefig(path, dpi=300)

    if save_plot_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, completed_deliveries_sum, fmt="%d")


def save_avg_stepcount(
    completion_steps: list[int],
    path: Path,
    filename: str,
    save_plot_data: bool = False,
    window_size: int = 20,
    start_eps: float = None,
    epsilon_decay: float = None,
):
    filename = filename.removesuffix(".pth")
    path = path / f"{filename}_avg_completion_steps_w{window_size}.png"
    n = len(completion_steps)
    avg_stepcount_sum = [max(completion_steps)] * n
    for i in range(n):
        cur_winsize = min(i + 1, window_size)
        avg_stepcount_sum[i] = (
            sum(completion_steps[i - cur_winsize + 1 : i + 1]) / cur_winsize
        )
    plt.figure(figsize=(10, 6))
    x = range(n)
    plt.plot(x, avg_stepcount_sum)
    # if start_eps is not None and epsilon_decay is not None:
    #     y = [start_eps * (epsilon_decay**i) for i in x]
    #     plt.plot(x, y, label="eps")
    plt.title(f"Average stepcount in last {window_size} episodes")
    plt.xlabel("episode")
    plt.ylabel("stepcount")
    plt.savefig(path, dpi=300)
    if save_plot_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, avg_stepcount_sum, fmt="%d")


def save_stepcount(
    completion_steps: list[int],
    path: Path,
    filename: str,
    save_plot_data: bool = False,
    start_eps: float = None,
    epsilon_decay: float = None,
):
    filename = filename.removesuffix(".pth")
    path = path / f"{filename}_completion_steps.png"
    plt.figure(figsize=(10, 6))
    x = range(len(completion_steps))
    plt.plot(x, completion_steps)
    # if start_eps is not None and epsilon_decay is not None:
    #     y = [start_eps * (epsilon_decay**i) for i in x]
    #     plt.plot(x, y, label="eps")
    plt.title("Stepcount")
    plt.xlabel("episode")
    plt.ylabel("stepcount")
    plt.savefig(path, dpi=300)
    if save_plot_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, completion_steps, fmt="%d")


def plot_delivery_quality():
    models = ["CNN1", "DQNet2"]
    view_sizes = [5, 7]
    data = defaultdict(list)
    data_path = Path(__file__).parent.parent / "data" / "times"
    for num_robots in range(10, 81, 10):
        for model in models:
            for view_size in view_sizes:
                filename = data_path / f"{model}_{num_robots}_{view_size}.json"
                with open(filename, "r", encoding="utf-8") as f:
                    data[f"{model}_{view_size}"].append(json.load(f))

    # Wykres longer_delivery
    plt.figure(figsize=(10, 6))
    for key, values in data.items():
        x = [v["num_robots"] for v in values]
        y = [round(v["longer_delivery"] * 100, 1) for v in values]

        plt.plot(x, y, marker="o", label=key)

    plt.xlabel("Number of Robots")
    plt.ylabel("Delivery length relative to optimum [%]")
    plt.title("Delivery Efficiency")
    plt.ylim(bottom=100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = (
        Path(__file__).parent.parent
        / "neural_networks"
        / "plots"
        / "delivery_efficiency.png"
    )
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    # Wykres throughput
    plt.figure(figsize=(10, 6))
    for key, values in data.items():
        x = [v["num_robots"] for v in values]
        y = [round(v["robot_throughput_per_100ticks"]) for v in values]

        plt.plot(x, y, marker="o", label=key)

    plt.xlabel("Number of Robots")
    plt.ylabel("Avg Robot throughput per 100 ticks")
    plt.title("Average Robot Throughput")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    path = (
        Path(__file__).parent.parent
        / "neural_networks"
        / "plots"
        / "delivery_throughput.png"
    )
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_delivery_quality()
