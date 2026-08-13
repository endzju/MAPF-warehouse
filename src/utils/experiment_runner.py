import math
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.neural_networks import observation_configs
from src.neural_networks.architectures.mlp import MLP
from src.utils.evaluate import run_evaluation
from src.utils.train import run_training


def get_best_model(result: tuple[list[nn.Module], list[float], list[float]]):
    models, _avg_completed_tasks, avg_delivery_times, avg_manhattan_times = result
    ratios = [h / d for d, h in zip(avg_delivery_times, avg_manhattan_times)]
    return models[int(np.argmax(ratios))]


def load_model(model: nn.Module):
    if not model.save_path.exists():
        return None
    weights_dict = torch.load(model.save_path, weights_only=True)
    model.load_state_dict(weights_dict)
    return model


def run_experiments(force_train: bool = True, eval: bool = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # CONFIG

    hidden_layers = [
        [256],
        [512],
        [1024],
        [32, 16],
        [64, 32],
        [128, 64],
        [256, 128],
        [512, 256],
        [1024, 512],
    ]
    view_sizes = [7, 9, 11, 13]

    models_settings = [
        {
            "class": MLP,
            "view_size": 7,
            "hidden_layers": {
                "mlp_layers": [512],
            },
            "model_config": "modulo2rewardxyfloatposxy",
        },
        {
            "class": MLP,
            "view_size": 7,
            "hidden_layers": {
                "mlp_layers": [512],
            },
            "model_config": "modulo2rewardxyfloatposx",
        },
        {
            "class": MLP,
            "view_size": 7,
            "hidden_layers": {
                "mlp_layers": [512],
            },
            "model_config": "modulo2rewardxyfloatposy",
        },
    ]

    for v, h in product(view_sizes, hidden_layers):
        models_settings.append(
            {
                "class": MLP,
                "view_size": v,
                "hidden_layers": {
                    "mlp_layers": h,
                },
                "model_config": "modulo2rewardxy",
            }
        )

    train_robot_list = [60]
    train_num_batches_list = [120]
    train_target_update_interval = [60]
    train_best_model_window = 10

    eval_robot_list = [n for n in range(10, 101, 5)]

    models = []

    for model_setting in models_settings:
        config = observation_configs.OBSERVATION_CONFIGS[model_setting["model_config"]]
        model = model_setting["class"](
            hidden_layers=model_setting["hidden_layers"],
            view_size=model_setting["view_size"],
            view_dims=config.view_dims,
            additional_input_size=config.get_additional_input_size(),
            output_size=config.get_output_size(),
            observation_config=config,
            observation_config_name=model_setting["model_config"],
        )
        models.append(model)

    base_params = {
        "num_episodes": 1000,
        "env_grid_size": (20, 20),
        "env_step_limit": 1500,
        "env_task_length": 5,
        "device": device,
        "plot": True,
        "epsilon": 1.0,
        "epsilon_min": 0,
        "epsilon_decay": 0.995,
        "epsilon_episodes": math.inf,
    }

    model_configs = list(
        product(
            models,
            train_robot_list,
            train_num_batches_list,
            train_target_update_interval,
        )
    )

    for model, num_robots, num_batches, target_update_interval in model_configs:
        model_dir = (
            Path(__file__).parent.parent
            / "neural_networks"
            / "models"
            / model.model_name
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_name = f"b{num_batches}_r{num_robots}_v{model.view_size}_u{target_update_interval}_{model.observation_config_name}.pth"
        model.save_path = model_dir / checkpoint_name
        model.checkpoint_name = checkpoint_name

        # TRAINING
        if not force_train:
            best_trained_model = load_model(model=model)
        should_train = force_train or best_trained_model is None
        if should_train:
            tic = time.time()
            print(f"Training model: {model.model_name}_{model.checkpoint_name}")
            params = base_params.copy()
            train_results = run_training(
                model=model,
                num_robots=num_robots,
                num_batches=num_batches,
                target_update_interval=target_update_interval,
                best_model_window=train_best_model_window,
                params=params.copy(),
            )
            print("model trained")

            best_trained_model, _, _, _ = train_results

            torch.save(best_trained_model.state_dict(), model.save_path)
            elapsed = time.time() - tic
            print(f"Training time: {elapsed / 60:.1f} min")

        # EVALUATION
        if eval:
            tic = time.time()
            print(f"Evaluating model: {model.model_name}_{model.checkpoint_name}")
            run_evaluation(
                model=best_trained_model,
                num_robot_list=eval_robot_list,
                params=base_params.copy(),
                num_simulations=50,
                num_processes=3,
            )
            elapsed = time.time() - tic
            print(f"Evaluation time: {elapsed / 60:.1f} min")


if __name__ == "__main__":
    print("Running experiments...")
    run_experiments(force_train=False, eval=True)
    # try:
    #     run_experiments(train=False, eval=False)
    # except KeyboardInterrupt:
    #     print("Stopped experiments")
    # except Exception as e:
    #     print(e)
    # else:
    #     print("Experiments completed.")
