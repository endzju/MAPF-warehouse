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
    models, avg_completed_tasks, avg_delivery_times, _ = result
    if len(set(avg_delivery_times)) == 1:
        return models[int(np.argmax(avg_completed_tasks))]
    else:
        return models[int(np.argmin(avg_delivery_times))]


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


def load_model(
    model: nn.Module,
    num_robots: int,
    view_size: int,
    num_batches: int,
    update_episodes: int,
):
    filename = f"{model.display_name}_b{num_batches}_r{num_robots}_v{view_size}_u{update_episodes}.pth"
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

    view_dims = 4
    goal_vec_size = 2
    n_actions = 5

    # CONFIG
    model_classes = [MLP]
    hidden_layers_list = [
        # [16],
        # [32],
        # [64],
        # [128],
        # [128, 64],
        [256, 128],
        # [512, 256],
        # [1024, 512],
        # [128, 64, 32],
        # [256, 128, 64],
        # [512, 256, 128],
        # [1024, 512, 256],
    ]
    train_robot_list = [60]
    train_num_batches_list = [10]
    train_view_sizes = [5]
    train_update_episodes = [10]

    eval_robot_list = [n for n in range(10, 101, 5)]

    models = [
        m(
            hidden_layers=h,
            input_size=view_dims * v * v + goal_vec_size,
            output_size=n_actions,
        )
        for m, h, v in product(model_classes, hidden_layers_list, train_view_sizes)
    ]
    steps_per_robot = 20
    base_params = {
        "num_episodes": 1000,
        "env_grid_size": (20, 20),
        "env_step_limit": 800,
        "env_task_length": 3,
        "device": device,
        "plot": True,
        "epsilon": 1.0,
        "epsilon_min": 0.01,
        "epsilon_decay": 0.995,
        "epsilon_episodes": math.inf,
    }

    model_configs = list(
        product(
            models,
            train_robot_list,
            train_view_sizes,
            train_num_batches_list,
            train_update_episodes,
        )
    )

    for model, num_robots, view_size, num_batches, update_episodes in model_configs:
        filename = f"{model.display_name}_b{num_batches}_r{num_robots}_v{view_size}_u{update_episodes}.pth"

        # TRAINING
        if not train:
            best_trained_model = load_model(
                model=model,
                num_robots=num_robots,
                view_size=view_size,
                num_batches=num_batches,
                update_episodes=update_episodes,
            )
        should_train = train or best_trained_model is None
        if should_train:
            tic = time.time()
            print(
                f"Training model: {model.display_name}, num robots: {num_robots}, view size: {view_size}, num batches: {num_batches}, update episodes: {update_episodes}"
            )
            params = base_params.copy()
            params["env_step_limit"] = steps_per_robot * num_robots
            train_results = run_training(
                model=model,
                num_robots=num_robots,
                view_size=view_size,
                num_batches=num_batches,
                update_episodes=update_episodes,
                params=params.copy(),
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
                update_episodes=update_episodes,
                params=base_params.copy(),
                num_simulations=50,
                num_processes=3,
            )
            elapsed = time.time() - tic
            print(f"Evaluation time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    print("Running experiments...")
    run_experiments(train=True, eval=True)
    # try:
    #     run_experiments(train=False, eval=False)
    # except KeyboardInterrupt:
    #     print("Stopped experiments")
    # except Exception as e:
    #     print(e)
    # else:
    #     print("Experiments completed.")
