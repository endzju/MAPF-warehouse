import math
from itertools import product

import torch

from src.neural_networks.architectures.mlp import MLP
from src.neural_networks.model_config import ModelConfig
from src.utils.evaluate import run_evaluation
from src.utils.train import run_training


def add_product(train_params_grid: dict, models_settings: list[dict]):
    keys = train_params_grid.keys()
    values = train_params_grid.values()
    new_settings = [dict(zip(keys, combo)) for combo in product(*values)]
    models_settings.extend(new_settings)


def run_experiments(force_train: bool = True, eval: bool = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # CONFIG

    # 4096 * 32 best num batches ~ 17, best result -> 72%, chyba tez niestabilne
    # 4096 * 16 best num batches ~ 30, best result -> 72%, a bit unstable
    # 4096 * 8  best num batches ~ 50 - 60, best result -> 72% very unstable
    # note for future me:

    models_settings = []
    eval_robot_list = [n for n in range(10, 101, 5)]
    train_workers = 5

    base_params = {
        "model_class": [MLP],
        "hidden_layers": [{"mlp_layers": [512]}],
        "view_size": [7],
        "buffer_length": [1_000_000],
        "num_robots": [60],
        "target_update_interval": [60],
        "suffix": ["sample1"],
    }
    variations = [
        {
            "batch_size": [2048],
            "num_batches": [150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250],
        },
        {
            "batch_size": [4096],
            "num_batches": [100],
            "suffix": ["sample2", "sample3", "sample4", "sample5"],
        },
        {
            "batch_size": [4096],
            "num_batches": [92, 96, 104, 108],
            "suffix": ["sample1", "sample2"],
        },
        {"batch_size": [4096 * 2], "num_batches": [80, 70, 60, 95, 105]},
        {"batch_size": [4096 * 4], "num_batches": [75, 65, 55, 85, 95]},
        {"batch_size": [4096 * 16], "num_batches": [70, 80, 90, 100]},
        {"batch_size": [4096 * 32], "num_batches": [40, 50, 60, 70, 80, 90]},
        {"batch_size": [4096 * 64], "num_batches": [30, 40, 50, 60, 70, 80]},
    ]

    # batch_size 512    	brak widocznego maksymalnego wyniku
    # batch_size 1024   	brak widocznego maksymalnego wyniku
    # batch_size 2048 in progress
    # batch_size 4096 in progress
    # batch_size 4096 * 2 	best num batches ~ 100, best result -> 74%, needs more epoch than 1000
    # batch_size 4096 * 4 	best num batches ~ 60-90, best result -> 73.5%, needs more epoch than 1000
    # batch_size 4096 * 8 	best num batches ~ 50-70, best result -> 74%, needs more epoch than 1000
    # batch_size 4096 * 16 	best num batches ~ 50-70, best result -> 73.8%, needs more epoch than 1000

    for var in variations:
        grid = base_params.copy()
        grid.update(var)
        add_product(grid, models_settings)

    # End of config

    model_configs = []
    for model_setting in models_settings:
        model_configs.append(ModelConfig(**model_setting))

    env_base_params = {
        "grid_size": (20, 20),
        "step_limit": 1500,
        "task_length": 5,
    }
    train_base_params = {
        "num_episodes": 1000,
        "device": device,
        "epsilon": 1.0,
        "epsilon_min": 0,
        "epsilon_decay": 0.995,
        "epsilon_episodes": math.inf,
        "best_model_window": 10,
    }

    # TRAINING
    run_training(
        model_configs=model_configs,
        env_base_params=env_base_params.copy(),
        train_base_params=train_base_params.copy(),
        force_train=force_train,
        num_processes=train_workers,
    )

    # EVALUATION
    if eval:
        run_evaluation(
            model_configs=model_configs,
            env_base_params=env_base_params.copy(),
            eval_robot_list=eval_robot_list,
            num_simulations=50,
            num_processes=3,
        )


if __name__ == "__main__":
    print("Running experiments...")
    run_experiments(force_train=False, eval=False)
    # try:
    #     run_experiments(train=False, eval=False)
    # except KeyboardInterrupt:
    #     print("Stopped experiments")
    # except Exception as e:
    #     print(e)
    # else:
    #     print("Experiments completed.")
