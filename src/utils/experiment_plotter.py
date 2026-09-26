from itertools import product

import torch

from src.neural_networks.architectures.cnn import CNN
from src.neural_networks.architectures.mlp import MLP
from src.neural_networks.model_config import ModelConfig
from src.utils.plots import plot_delivery_efficiency, plot_delivery_throughput


def add_product(train_params_grid: dict, models_settings: list[dict]):
    keys = train_params_grid.keys()
    values = train_params_grid.values()
    new_settings = [dict(zip(keys, combo)) for combo in product(*values)]
    models_settings.extend(new_settings)


if __name__ == "__main__":
    model_classes = [MLP, CNN]
    models_settings = []
    eval_robot_list = [n for n in range(10, 101, 5)]
    train_workers = 4
    device = torch.device("cpu")

    base_params = {
        "grid_size": [(20, 20)],
        "step_limit": [4000],
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
        "float_goal_vector": [True],
    }
    variations = [
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [20],
            "batch_size": [4096 * 4],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000, 600_000, 700_000, 800_000],
            "step_limit": [4000],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [
                16,
                18,
                20,
                22,
                24,
                26,
                28,
                30,
                32,
                34,
                36,
                38,
                40,
                45,
                50,
                55,
                60,
                65,
                70,
            ],
            "batch_size": [4096 * 4],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "step_limit": [4000],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 2, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 2, 0)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 4, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 4, 0)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 8, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 8, 0)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 16, 0)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 32, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 32, 0)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 64, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 64, 0)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [20],
            "batch_size": [4096 * 4],
            "suffix": ["search1"],
            "target_update_interval": [30],
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
            "num_batches": [20],
            "batch_size": [4096 * 4],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "step_limit": [4000],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [11, 13],
            "num_batches": [20],
            "batch_size": [4096 * 4],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "step_limit": [4000],
        },
    ]

    for var in variations:
        grid = base_params.copy()
        grid.update(var)
        add_product(grid, models_settings)

    model_configs = []
    for model_setting in models_settings:
        model_configs.append(ModelConfig(**model_setting))

    x_ticks = [n for n in range(10, 101, 5)]

    plot_delivery_efficiency(model_configs=model_configs, x_ticks=x_ticks)
    plot_delivery_throughput(model_configs=model_configs, x_ticks=x_ticks)
