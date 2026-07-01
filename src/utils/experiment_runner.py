import math
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.neural_networks.CNN.cnn import (  # noqa: F401
    CNN_16k3_32k3,
    CNN_32k3_64k3,
    CNN_64k3_128k3,
    CNN_128k3_256k3,
)
from src.neural_networks.MLP.mlp import (  # noqa: F401
    MLP_64_32,
    MLP_256_128,
    MLP_512_256_128,
)
from src.utils.evaluate import run_evaluation
from src.utils.train import run_training


def get_best_models(
    results: list[tuple[list[nn.Module], list[float], list[float]]], step_limit: int
) -> list[nn.Module]:
    best_models = []
    for models, avg_tasks, avg_steps in results:
        if set(avg_steps) == {step_limit}:
            best_models.append(models[int(np.argmax(avg_tasks))])
        else:
            best_models.append(models[int(np.argmin(avg_steps))])
    return best_models


def run_experiments():
    print("Running training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # model_classes = [CNN_16k3_32k3, CNN_32k3_64k3, CNN_64k3_128k3, CNN_128k3_256k3]
    model_classes = [MLP_256_128]
    train_robot_list = [10, 20, 30, 40, 50, 60, 70, 80]
    train_view_sizes = [5]

    base_params = {
        "num_episodes": 1000,
        "env_grid_size": (20, 20),
        "env_step_limit": 800,
        "env_task_length": 3,
        "device": device,
        "plot": True,
        "save_plot_data": False,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "epsilon_episodes": math.inf,
    }

    model_configs = list(product(model_classes, train_robot_list, train_view_sizes))

    train_results = run_training(
        model_configs=model_configs,
        params=base_params.copy(),
    )

    best_trained_models = get_best_models(train_results, base_params["env_step_limit"])

    # tune_results = run_fine_tuning(
    #     model_configs=model_configs,
    #     params=base_params.copy(),
    #     best_models=best_trained_models,
    # )

    # best_tuned_models = get_best_models(tune_results, base_params["env_step_limit"])
    evaluate_robot_list = list(range(10, 81, 10))

    model_dir = Path(__file__).parent.parent / "neural_networks" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("Running evaluation...")

    for i, (model_class, num_robots, view_size) in enumerate(model_configs):
        for model, is_tuned in [
            (best_trained_models[i], False),
            # (best_tuned_models[i], True),
        ]:
            model = model.cpu()
            infix = "_tuned" if is_tuned else ""
            filename = (
                f"{model_class.__name__}{infix}_robots{num_robots}_view{view_size}.pth"
            )
            print(f"Evaluating model: {filename}")
            torch.save(model.state_dict(), model_dir / filename)
            run_evaluation(
                model=model,
                num_robot_list=evaluate_robot_list,
                train_num_robots=num_robots,
                view_size=view_size,
                params=base_params.copy(),
                is_tuned=is_tuned,
                num_simulations=10,
            )


if __name__ == "__main__":
    print("Running experiments...")
    run_experiments()
    print("Experiments completed.")
