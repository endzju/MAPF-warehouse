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
    _model_classes = [MLP, CNN]
    models_settings = []
    eval_robot_list = [n for n in range(10, 101, 5)]
    train_workers = 3

    base_params = {
        "grid_size": [(20, 20)],
        "step_limit": [3000],
        "task_length": [5],
        "device": [device],
        "model_class": [MLP],
        "hidden_layers": [{"mlp_layers": [512]}],
        "view_size": [7],
        "buffer_length": [500_000],
        "num_robots": [60],
        "target_update_interval": [30],
        "suffix": ["sample1"],
        "batch_size": [4096 * 6],
        "num_batches": [10],
        "num_episodes": [1200],
        "gamma": [0.95],
    }
    variations = [
        {
            "model_class": [MLP],
            "hidden_layers": [
                {"mlp_layers": [1024]},
            ],
            "view_size": [9, 11],
            "num_batches": [8, 10, 12],
            "batch_size": [4096 * 8],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "float_goal_vector": [True],
        },
        {
            "model_class": [MLP],
            "hidden_layers": [
                {"mlp_layers": [1024]},
            ],
            "view_size": [7],
            "num_batches": [4, 5, 6, 7, 8, 9, 10, 11, 12],
            "batch_size": [4096 * 8],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "float_goal_vector": [True],
        },
        {
            "model_class": [MLP],
            "hidden_layers": [
                {"mlp_layers": [1024]},
            ],
            "view_size": [7],
            "num_batches": [10],
            "batch_size": [4096 * 8],
            "suffix": ["search1"],
            "target_update_interval": [25, 30, 35],
            "buffer_length": [500_000],
            "float_goal_vector": [True],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 32, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [10, 15, 20, 25, 30, 35, 40],
            "batch_size": [512, 1024, 2048, 4096, 4096 * 2, 4096 * 4, 4096 * 8],
            "num_episodes": [1000],
            "suffix": ["sample1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "float_goal_vector": [False],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 32, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [30],
            "batch_size": [4096],
            "num_episodes": [1200],
            "suffix": ["sample1"],
            "target_update_interval": [30],
            "buffer_length": [300_000, 400_000, 500_000, 600_000, 700_000],
            "float_goal_vector": [False],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 32, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [30],
            "batch_size": [4096],
            "num_episodes": [1200],
            "suffix": ["sample1"],
            "target_update_interval": [10, 20, 30, 40, 50, 60, 70, 80],
            "buffer_length": [500_000],
            "float_goal_vector": [False],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 32, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 64, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [30],
            "batch_size": [4096],
            "num_episodes": [1200],
            "suffix": ["sample1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "float_goal_vector": [False],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(5, 32, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [30],
            "batch_size": [4096],
            "num_episodes": [1200],
            "suffix": ["sample1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "float_goal_vector": [False],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(7, 32, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [30],
            "batch_size": [4096],
            "num_episodes": [1200],
            "suffix": ["sample1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "float_goal_vector": [False],
        },
    ]

    # MLP 512
    # batch_size 4096 * 8
    # best num batches ~ 10
    # best gamma ~ 0.950
    # best buffor length ~ 500_000
    # best step limit >= 3000
    #

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
    run_experiments(force_train=False, eval=True)
    # try:
    #     run_experiments(train=False, eval=False)
    # except KeyboardInterrupt:
    #     print("Stopped experiments")
    # except Exception as e:
    #     print(e)
    # else:
    #     print("Experiments completed.")
