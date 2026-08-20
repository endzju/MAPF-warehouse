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

    model_class_list = [MLP]
    hidden_layers_list = [{"mlp_layers": [512]}]
    agent_view_size_list = [7]
    buffer_length_list = [1_000_000, 2_000_000]
    batch_size_list = [4096 * 4]
    num_robot_list = [60]
    num_batches_list = [100, 110, 90]
    target_update_interval_list = [60]
    suffix_list = ["sample1", "sample2", "sample3", "sample4", "sample5"]

    train_params_grid = {
        "model_class": model_class_list,
        "hidden_layers": hidden_layers_list,
        "view_size": agent_view_size_list,
        "buffer_length": buffer_length_list,
        "batch_size": batch_size_list,
        "num_robots": num_robot_list,
        "num_batches": num_batches_list,
        "target_update_interval": target_update_interval_list,
        "suffix": suffix_list,
    }
    models_settings = []
    train_workers = 4
    eval_robot_list = [n for n in range(10, 101, 5)]

    add_product(train_params_grid, models_settings)

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
