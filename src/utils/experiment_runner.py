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
        "step_limit": [4000],
        "task_length": [5],
        "device": [device],
        "model_class": [CNN],
        "hidden_layers": [
            {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
        ],
        "view_size": [9],
        "buffer_length": [500_000],
        "num_robots": [60],
        "target_update_interval": [30],
        "suffix": ["sample1"],
        "batch_size": [4096 * 16],
        "num_batches": [30],
        "num_episodes": [1200],
        "gamma": [0.95],
        "float_goal_vector": [True],
    }
    variations = [
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "suffix": ["search1"],
            "target_update_interval": [10, 15, 20, 25, 30, 35, 40],
            "buffer_length": [500_000],
            "step_limit": [4000],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "suffix": ["search1"],
            "buffer_length": [500_000],
            "step_limit": [4000],
            "num_robots": [50, 55, 60, 65, 70],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [20],
            "batch_size": [4096 * 4],
            "suffix": ["search1"],
            "buffer_length": [300_000, 400_000, 500_000, 600_000, 700_000, 800_000],
            "step_limit": [4000],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [5, 10, 15, 20, 25],
            "batch_size": [4096 * 8, 4096 * 16, 4096 * 32, 4096 * 64],
            "suffix": ["search1"],
            "buffer_length": [500_000],
            "step_limit": [4000],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [128]},
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [512]},
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [128, 64]},
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256, 128]},
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [512, 256]},
            ],
            "view_size": [9],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "step_limit": [4000],
        },
    ]

    # MLP 512 -
    # batch_size 4096 * 8
    # best num batches ~ 10
    # best gamma ~ 0.950
    # best buffor length ~ 500_000
    # best step limit >= 3000
    #

    # best CNN params:
    # "cnn_layers": [(3, 16, 1)]        retrain with higher batch size
    # "mlp_layers": [256]               retrain with higher batch size
    # "target_update_interval": 30      retrain with higher batch size
    # "num_robots": 60                  retrain with higher batch size
    # best buffor length ~ 500_000      retrain with higher batch size
    # batch_size >= 4096 * 16

    # best models:
    # v9_r60_u30_l500000_s16384_b20_g0950_t4000_d0995_search1 -> 54.5 tasks / 100 ticks, very good with large number of robots

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
