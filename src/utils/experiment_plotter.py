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
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [1024]},
        #     ],
        #     "view_size": [11],
        #     "num_batches": [10, 15, 20],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [1024]},
        #     ],
        #     "view_size": [11],
        #     "num_batches": [10],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "num_robots": [30, 40, 50, 60, 70, 80],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [1024]},
        #     ],
        #     "view_size": [11],
        #     "num_batches": [10],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        #     "buffer_length": [500_000],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [1024]},
        #     ],
        #     "view_size": [11],
        #     "num_batches": [10],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [300_000, 400_000, 500_000, 600_000, 700_000],
        # },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9, 11],
            "num_batches": [20],
            "batch_size": [4096 * 8],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [20],
            "batch_size": [4096 * 8, 4096 * 4, 4096 * 2],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
        },
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "num_batches": [20],
            "batch_size": [4096 * 8],
            "num_episodes": [1000],
            "suffix": ["sample1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "step_limit": [
                5000,
                6000,
            ],
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
