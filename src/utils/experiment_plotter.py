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

    base_params = {
        "grid_size": [(20, 20)],
        "step_limit": [1000],
        "task_length": [5],
        "device": [torch.device("cpu")],
        "model_class": [MLP],
        "hidden_layers": [{"mlp_layers": [512]}],
        "view_size": [7],
        "buffer_length": [1_000_000],
        "num_robots": [60],
        "target_update_interval": [60],
        "suffix": ["sample1"],
        "batch_size": [4096 * 6],
        "num_batches": [45],
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
            "num_episodes": [1500],
            "suffix": ["fix5"],
            "step_limit": [900],
            "target_update_interval": [60],
            "buffer_length": [1_000_000],
            "float_goal_vector": [True],
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
