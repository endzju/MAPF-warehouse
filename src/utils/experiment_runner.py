import math

import torch

from src.neural_networks.CNN.cnn import CNN1  # noqa: F401
from src.neural_networks.MLP.mlp import MLP1, MLP2, MLP3  # noqa: F401
from src.utils.fine_tune import run_fine_tuning
from src.utils.train import run_training


def run():

    print("Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_classes = [MLP2]
    num_robot_list = [10, 20, 30, 40, 50, 60, 70, 80]
    view_sizes = [5]

    params = {
        "num_episodes": 2000,
        "env_grid_size": (20, 20),
        "env_step_limit": 300,
        "env_task_length": 5,
        "device": device,
        "plot": True,
        "save_plot_data": False,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "epsilon_episodes": math.inf,
    }

    results = run_training(
        model_classes=model_classes,
        num_robot_list=num_robot_list,
        view_sizes=view_sizes,
        params=params,
    )

    for r in results:
        modules, avg_tasks, avg_steps = r

    # TODO: load best models
    best_models = []

    run_fine_tuning(
        model_classes=model_classes,
        num_robot_list=num_robot_list,
        view_sizes=view_sizes,
        params=params,
        best_models=best_models,
    )


if __name__ == "__main__":
    run()
