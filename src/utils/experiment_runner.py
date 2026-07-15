import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.neural_networks.MLP.mlp import MLP
from src.utils.evaluate import run_evaluation
from src.utils.train import run_training


def get_best_model(
    result: tuple[list[nn.Module], list[float], list[float]], step_limit: int
):
    models, avg_tasks, avg_steps = result
    if set(avg_steps) == {step_limit}:
        return models[int(np.argmax(avg_tasks))]
    else:
        return models[int(np.argmin(avg_steps))]


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


def load_model(model: nn.Module, num_robots: int, view_size: int, num_batches: int):
    filename = f"{model.display_name}_b{num_batches}_r{num_robots}_v{view_size}.pth"
    path = (
        Path(__file__).parent.parent
        / "neural_networks"
        / "models"
        / model.display_name
        / filename
    )
    if not path.exists():
        return None
    weights_dict = torch.load(path, weights_only=True)
    model.load_state_dict(weights_dict)
    return model


def run_experiments(train: bool = True, eval: bool = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    view_shape = (4, 5, 5)
    goal_vec_size = 2

    model_params = {
        "input_size": view_shape[0] * view_shape[1] * view_shape[2] + goal_vec_size,
        "output_size": 5,
    }

    # CONFIG
    model_classes = [MLP]
    hidden_layers_list = [
        [16],
        [32],
        [64],
        [128],
        [128, 64],
        [256, 128],
        [512, 256],
        [1024, 512],
        [128, 64, 32],
        [256, 128, 64],
        [512, 256, 128],
        [1024, 512, 256],
    ]
    train_robot_list = [40]
    train_num_batches_list = [25]
    train_view_sizes = [5]

    eval_robot_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    models = [
        m(hidden_layers=h, **model_params)
        for m, h in product(model_classes, hidden_layers_list)
    ]
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

    model_configs = list(
        product(models, train_robot_list, train_view_sizes, train_num_batches_list)
    )

    for model, num_robots, view_size, num_batches in model_configs:
        filename = f"{model.display_name}_b{num_batches}_r{num_robots}_v{view_size}.pth"

        # TRAINING
        if not train:
            best_trained_model = load_model(
                model=model,
                num_robots=num_robots,
                view_size=view_size,
                num_batches=num_batches,
            )
        should_train = train or best_trained_model is None
        if should_train:
            tic = time.time()
            print(
                f"Training model: {model.display_name}, num robots: {num_robots}, view size: {view_size}, num batches: {num_batches}"
            )
            train_results = run_training(
                model=model,
                num_robots=num_robots,
                view_size=view_size,
                num_batches=num_batches,
                params=base_params.copy(),
            )
            print("model trained")

            best_trained_model = get_best_model(
                result=train_results, step_limit=base_params["env_step_limit"]
            )
            model_dir = (
                Path(__file__).parent.parent
                / "neural_networks"
                / "models"
                / model.display_name
            )
            model_dir.mkdir(parents=True, exist_ok=True)

            torch.save(best_trained_model.state_dict(), model_dir / filename)
            elapsed = time.time() - tic
            print(f"Training time: {elapsed / 60:.1f} min")

        # EVALUATION
        if eval:
            tic = time.time()
            model = best_trained_model.cpu()
            base_params["env_task_length"] = 5
            print(f"Evaluating model: {filename}")

            run_evaluation(
                model=model,
                num_robot_list=eval_robot_list,
                train_num_robots=num_robots,
                view_size=view_size,
                num_batches=num_batches,
                params=base_params.copy(),
                num_simulations=30,
                num_processes=3,
            )
            elapsed = time.time() - tic
            print(f"Evaluation time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    print("Running experiments...")
    run_experiments(train=False, eval=True)
    # try:
    #     run_experiments(train=False, eval=True)
    # except KeyboardInterrupt:
    #     print("Stopped experiments")
    # except Exception as e:
    #     print(e)
    # else:
    #     print("Experiments completed.")
