from itertools import product

import torch

from src.neural_networks.architectures.cnn import CNN
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
    model_classes = [MLP, CNN]
    models_settings = []
    eval_robot_list = [n for n in range(10, 101, 5)]
    train_workers = 3

    base_params = {
        "grid_size": [(20, 20)],
        "step_limit": [1500],
        "task_length": [5],
        "device": [device],
        "model_class": [MLP],
        "hidden_layers": [{"mlp_layers": [512]}],
        "view_size": [11],
        "buffer_length": [500_000],
        "num_robots": [60],
        "target_update_interval": [60],
        "suffix": ["sample1"],
        "batch_size": [4096 * 6],
        "num_batches": [45],
        "num_episodes": [1200],
    }
    variations = [
        {
            "model_class": [MLP],
            "hidden_layers": [
                {"mlp_layers": [512]},
            ],
            "view_size": [7],
            "num_batches": [50],
            "batch_size": [4096 * 6],
            "num_episodes": [1200],
            "suffix": ["retry2_no_f_vec"],
            "step_limit": [1000],
            "target_update_interval": [60],
            "buffer_length": [1_000_000],
            "float_goal_vector": [False],
        },
    ]

    # MLP 512 neurons
    # batch_size 512    	brak widocznego maksymalnego wyniku
    # batch_size 1024   	brak widocznego maksymalnego wyniku
    # batch_size 2048 		best num batches ~ 150, best result -> 73.8%
    # batch_size 4096 		best num batches ~ 104, best result -> 74.5$
    # batch_size 4096 * 2 	best num batches ~ 75, best result -> 74.71%
    # batch_size 4096 * 4 	best num batches ~ 65, best result -> 74.95%
    # batch_size 4096 * 6	best num batches ~ 45, best result -> 74.90%, very stable
    # batch_size 4096 * 8 	best num batches ~ 50, best result -> 75.15%
    # batch_size 4096 * 16 	best num batches ~ 45, best result -> 74.89%
    # batch_size 4096 * 32 	best num batches ~ 50, best result -> 74.93%

    # CNN k3c64p1
    # batch_size 1024   	best num batches ~ 30, best result -> 49.12%
    # batch_size 2048 		best num batches ~ 30, best result -> 58.70%
    # batch_size 4096 		best num batches ~ 40 or 90, best result -> 65.10%, quite stable
    # batch_size 4096 * 2 	best num batches ~ 20-30, best result -> ?
    # batch_size 4096 * 4 	best num batches ~ 30, best result -> 63.53%, stable
    # batch_size 4096 * 8 	best num batches ~ 20-30, best result -> ?

    # CNN k3c64p0
    # batch_size 1024   	best num batches ~ 30, best result -> 56.02%
    # batch_size 2048 		best num batches ~ 30, best result -> 60.64%
    # batch_size 4096 		best num batches ~ 30, best result -> 63.22%
    # batch_size 4096 * 2 	best num batches ~ 30, best result -> 65.26%, very stable
    # batch_size 4096 * 4 	best num batches ~ 20, best result -> 61.74%, unstable

    # CNN k3c32p1
    # batch_size 4096   	best num batches ~ +-30, best result -> 64.36%, needs more training
    # batch_size 4096 * 2 	best num batches ~ 22-26, best result -> 63.63%, a bit unstable
    # batch_size 4096 * 4 	best num batches ~ +-30, best result -> 65.81%, stable

    # CNN k3c32p0
    # batch_size 4096   	best num batches ~ 26, best result -> 64.76%, a bit unstable
    # batch_size 4096 * 2 	best num batches ~ 26, best result -> 65.31%, stable
    # batch_size 4096 * 4 	best num batches ~ 22, best result -> 64.03%, stable

    for var in variations:
        grid = base_params.copy()
        grid.update(var)
        add_product(grid, models_settings)

    # End of config

    model_configs = []
    for model_setting in models_settings:
        model_configs.append(ModelConfig(**model_setting))

    # TRAINING
    run_training(
        model_configs=model_configs,
        force_train=force_train,
        num_processes=train_workers,
    )

    # EVALUATION
    if eval:
        run_evaluation(
            model_configs=model_configs,
            eval_robot_list=eval_robot_list,
            num_simulations=50,
            num_processes=4,
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
