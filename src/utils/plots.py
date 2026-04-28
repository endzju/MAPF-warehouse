from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def save_completed_deliveries_plot(
    completed_deliveries: list[int],
    path: Path,
    save_data: bool = False,
    window_size: int = 20,
):
    n = len(completed_deliveries)
    completed_deliveries_sum = [0] * n
    for i in range(n):
        completed_deliveries_sum[i] = sum(
            completed_deliveries[i - window_size + 1 : i + 1]
        )
    plt.figure(figsize=(10, 6))
    x = range(n)
    plt.plot(x, completed_deliveries_sum)
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
    save_data: bool = False,
    window_size: int = 20,
):
    n = len(completion_steps)
    avg_stepcount_sum = [0] * n
    for i in range(n):
        cur_winsize = min(i + 1, window_size)
        avg_stepcount_sum[i] = (
            sum(completion_steps[i - window_size + 1 : i + 1]) / cur_winsize
        )
    plt.figure(figsize=(10, 6))
    x = range(n)
    plt.plot(x, avg_stepcount_sum)
    plt.title(f"Average stepcount in last {window_size} episodes")
    plt.xlabel("episode")
    plt.ylabel("stepcount")
    plt.savefig(path, dpi=300)
    if save_data:
        txt_path = path.with_suffix(".txt")
        np.savetxt(txt_path, avg_stepcount_sum, fmt="%d")
