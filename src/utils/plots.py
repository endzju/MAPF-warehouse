import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from torch import nn


def plot_avg_completed_tasks_percentage(
    avg_completed_tasks: list[int],
    max_tasks: int,
    path: Path,
    filename: str,
    save_plot_data: bool = False,
    window_size: int = 20,
):
    filename = filename.removesuffix(".pth")
    path = path / f"{filename}_completed_tasks_percentage_w{window_size}.png"
    completed_tasks_percentage = [t / max_tasks * 100 for t in avg_completed_tasks]
    plt.figure(figsize=(10, 6))
    x = range(len(avg_completed_tasks))
    plt.plot(x, completed_tasks_percentage)
    plt.title(f"Completed tasks percentage last window ({window_size} episodes)")
    plt.xlabel("window")
    plt.ylabel("completed tasks [%]")
    plt.ylim(-5, 105)
    plt.yticks(range(0, 101, 10))
    plt.savefig(path, dpi=300)
    plt.close()

    if save_plot_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, completed_tasks_percentage, fmt="%d")


def plot_avg_stepcount(
    avg_completion_steps: list[int],
    path: Path,
    filename: str,
    window_size: int,
    save_plot_data: bool = False,
):
    filename = filename.removesuffix(".pth")
    path = path / f"{filename}_avg_completion_steps_w{window_size}.png"

    plt.figure(figsize=(10, 6))
    x = range(len(avg_completion_steps))
    plt.plot(x, avg_completion_steps)
    plt.title(f"Average stepcount last window ({window_size} episodes)")
    plt.xlabel("window")
    plt.ylabel("avg stepcount")
    plt.savefig(path, dpi=300)
    plt.close()
    if save_plot_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, avg_completion_steps, fmt="%d")


def plot_delivery_quality(models: list[nn.Module], view_sizes: list[int]):
    # TODO complete change
    data = defaultdict(list)
    data_path = Path(__file__).parent.parent / "data" / "times"
    for num_robots in range(10, 81, 10):
        for model, view_size in zip(models, view_sizes, strict=True):
            filename = (
                data_path / f"{model.__name__}_robots{num_robots}_view{view_size}.json"
            )
            with open(filename, "r", encoding="utf-8") as f:
                data[f"{model.__name__}_{view_size}"].append(json.load(f))

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
