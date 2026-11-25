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
        "model_class": [CNN],
        "hidden_layers": [
            {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
        ],
        "view_size": [11],
        "buffer_length": [500_000],
        "num_robots": [60],
        "target_update_interval": [4],
        "suffix": ["sample1"],
        "batch_size": [4096 * 16],
        "num_batches": [25],
        "num_episodes": [1200],
        "gamma": [0.95],
        "float_goal_vector": [True],
    }
    variations = [
        # {
        #     "model_class": [CNN],
        #     "hidden_layers": [
        #         {"cnn_layers": [(3, 16, 0)], "mlp_layers": [256]},
        #         {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
        #         {"cnn_layers": [(3, 16, 0)], "mlp_layers": [512]},
        #         {"cnn_layers": [(3, 16, 1)], "mlp_layers": [512]},
        #         {"cnn_layers": [(5, 16, 0)], "mlp_layers": [256]},
        #         {"cnn_layers": [(5, 16, 1)], "mlp_layers": [256]},
        #         {"cnn_layers": [(5, 16, 0)], "mlp_layers": [512]},
        #         {"cnn_layers": [(5, 16, 1)], "mlp_layers": [512]},
        #     ],
        #     "suffix": ["search1"],
        #     "step_limit": [4000],
        # },
        # CNN_k3c16p1MLP_256_v9_r60_u4_l500000_s65536_b30_g0950_t4000_d0995
        {
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 16, 1)], "mlp_layers": [256]},
            ],
            "view_size": [9],
            "buffer_length": [500_000],
            "num_robots": [60],
            "target_update_interval": [4],
            "batch_size": [65_536],
            "num_batches": [30],
            "gamma": [0.95],
            "step_limit": [4000],
            "suffix": [""],
        },
        # MLP_512_v7_r60_u30_l500000_s32768_b10_g0990_t3000_d0995
        {
            "model_class": [MLP],
            "hidden_layers": [
                {"mlp_layers": [512]},
            ],
            "view_size": [7],
            "buffer_length": [500_000],
            "num_robots": [60],
            "target_update_interval": [30],
            "batch_size": [32_768],
            "num_batches": [10],
            "gamma": [0.99],
            "step_limit": [3000],
            "suffix": [""],
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
