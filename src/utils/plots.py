import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
from torch import nn

plot_colors = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#CC79A7",  # reddish purple
    "#332288",  # dark blue
    "#117733",  # green
    "#FBF2C4",  # sand
]

plot_lines = [
    "-",
    "--",
    ":",
    "-.",
]


def plot_avg_completed_tasks_percentage(
    avg_completed_tasks: list[int],
    max_tasks: int,
    path: Path,
    filename: str,
    window_size: int = 20,
):
    filename = filename.removesuffix(".pth")
    path = path / f"{filename}_completed_tasks_percentage_w{window_size}.png"
    completed_tasks_percentage = [t / max_tasks * 100 for t in avg_completed_tasks]
    plt.figure(figsize=(10, 6))
    x = range(len(avg_completed_tasks))
    plt.plot(x, completed_tasks_percentage)
    plt.grid(True)
    plt.title(f"Completed tasks percentage last window ({window_size} episodes)")
    plt.xlabel("window")
    plt.ylabel("completed tasks [%]")
    plt.ylim(-5, 105)
    plt.yticks(range(0, 101, 10))
    plt.savefig(path, dpi=300)
    plt.close()


def plot_avg_delivery_times(
    avg_delivery_times: list[int],
    avg_manhattan_times: list[int],
    path: Path,
    filename: str,
    window_size: int,
):
    if len(avg_delivery_times) != len(avg_manhattan_times):
        raise ValueError(
            "avg_delivery_times and avg_manhattan_times must have the same length"
        )
    filename = filename.removesuffix(".pth")
    path = path / f"{filename}_avg_delivery_times_w{window_size}.png"

    plt.figure(figsize=(10, 6))
    x = range(len(avg_delivery_times))
    plt.plot(x, avg_delivery_times, label="Delivery time")
    plt.plot(x, avg_manhattan_times, label="Manhattan lower bound")

    min_idx = min(range(len(avg_delivery_times)), key=lambda i: avg_delivery_times[i])
    min_value = avg_delivery_times[min_idx]
    plt.scatter(min_idx, min_value, zorder=5)
    plt.annotate(
        f"{min_value:.2f}",
        (min_idx, min_value),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
    )

    plt.title(f"Average Delivery Time (Last {window_size} Episodes)")
    plt.xlabel("window")
    plt.grid(True)
    plt.ylabel("Avg delivery time")
    plt.savefig(path, dpi=300)
    plt.close()


def read_model_data(
    model: nn.Module,
    batch_size: int,
    num_robots: int,
    view_size: int,
    update_episodes: int,
) -> dict:
    data_path = Path(__file__).parent.parent / "data" / "times" / model.display_name
    filename = (
        data_path
        / f"{model.display_name}_b{batch_size}_r{num_robots}_v{view_size}_u{update_episodes}.json"
    )
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def read_models_data(
    models: list[nn.Module],
    batch_sizes: list[int],
    num_robot_list: list[int],
    view_sizes: list[int],
    update_episodes_list: list[int],
):
    data = {}
    for (
        model,
        batch_size,
        num_robots,
        view_size,
        update_episodes,
    ) in list(
        product(models, batch_sizes, num_robot_list, view_sizes, update_episodes_list)
    ):
        data[
            f"{model.display_name}_b{batch_size}_r{num_robots}_v{view_size}_u{update_episodes}"
        ] = read_model_data(model, batch_size, num_robots, view_size, update_episodes)

    return data


def plot_delivery_efficiency(
    models: list[nn.Module],
    batch_sizes: list[int],
    num_robot_list: list[int],
    view_sizes: list[int],
    update_episodes_list: list[int],
    x_ticks: list[int | str],
):
    x_ticks_set = {str(tick) for tick in x_ticks}
    data = read_models_data(
        models=models,
        batch_sizes=batch_sizes,
        num_robot_list=num_robot_list,
        view_sizes=view_sizes,
        update_episodes_list=update_episodes_list,
    )
    plt.figure(figsize=(10, 6))
    for i, (model_name, evaluation) in enumerate(data.items()):
        evaluation = {k: v for k, v in evaluation.items() if k in x_ticks_set}
        evaluation = dict(sorted(evaluation.items(), key=lambda item: int(item[0])))
        x = [num_robots for num_robots in evaluation]
        y = [
            round(eval_info["movement_efficiency"] * 100, 2)
            for eval_info in evaluation.values()
        ]

        plt.plot(
            x,
            y,
            label=model_name,
            color=plot_colors[i % len(plot_colors)],
            linestyle=plot_lines[(i // len(plot_colors)) % len(plot_lines)],
            marker="o",
        )

    plt.xlabel("Number of Robots")
    plt.ylabel("Movement length relative to optimum [%]")
    plt.title("Movement Efficiency")
    plt.ylim(bottom=100)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    results_path = (
        Path(__file__).parent.parent / "neural_networks" / "plots" / "results"
    )
    results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / "delivery_efficiency.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_delivery_throughput(
    models: list[nn.Module],
    batch_sizes: list[int],
    num_robot_list: list[int],
    view_sizes: list[int],
    update_episodes_list: list[int],
    x_ticks: list[str | int],
):
    x_ticks_set = {str(tick) for tick in x_ticks}
    data = read_models_data(
        models,
        batch_sizes,
        num_robot_list,
        view_sizes,
        update_episodes_list,
    )
    plt.figure(figsize=(10, 6))
    for i, (model_name, evaluation) in enumerate(data.items()):
        evaluation = {k: v for k, v in evaluation.items() if k in x_ticks_set}
        evaluation = dict(sorted(evaluation.items(), key=lambda item: int(item[0])))
        x = [num_robots for num_robots in evaluation]
        y = [
            round(eval_info["robot_throughput_per_100ticks"], 2)
            for eval_info in evaluation.values()
        ]

        plt.plot(
            x,
            y,
            label=model_name,
            color=plot_colors[i % len(plot_colors)],
            linestyle=plot_lines[(i // len(plot_colors)) % len(plot_lines)],
            marker="o",
        )

    plt.xlabel("Number of Robots")
    plt.ylabel("Avg Robot throughput per 100 ticks")
    plt.title("Average Robot Throughput")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    results_path = (
        Path(__file__).parent.parent / "neural_networks" / "plots" / "results"
    )
    results_path.mkdir(parents=True, exist_ok=True)
    path = results_path / "delivery_throughput.png"
    plt.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    pass
