from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_completed_deliveries_plot(
    completed_deliveries: list[int],
    path: Path,
    filename: str,
    save_data: bool = False,
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

    if save_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, completed_deliveries_sum, fmt="%d")


def save_avg_stepcount(
    completion_steps: list[int],
    path: Path,
    filename: str,
    save_data: bool = False,
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
    if save_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, avg_stepcount_sum, fmt="%d")


def save_stepcount(
    completion_steps: list[int],
    path: Path,
    filename: str,
    save_data: bool = False,
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
    if save_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, completion_steps, fmt="%d")
