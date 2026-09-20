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
        "step_limit": [3000],
        "task_length": [5],
        "device": [torch.device("cpu")],
        "model_class": [MLP],
        "hidden_layers": [{"mlp_layers": [512]}],
        "view_size": [7],
        "buffer_length": [500_000],
        "num_robots": [60],
        "target_update_interval": [30],
        "suffix": ["sample1"],
        "batch_size": [4096 * 6],
        "num_batches": [45],
        "num_episodes": [1200],
    }
    variations = [
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "task_tsp": [True],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [2048]},
        #         {"mlp_layers": [1024]},
        #         {"mlp_layers": [512]},
        #         {"mlp_layers": [256]},
        #         {"mlp_layers": [128]},
        #         {"mlp_layers": [1024, 512]},
        #         {"mlp_layers": [512, 256]},
        #         {"mlp_layers": [256, 128]},
        #         {"mlp_layers": [128, 64]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [10],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "gamma": [0.95],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [7, 8, 9, 10, 11, 12, 13, 40, 50, 60],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [False],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "x_position_float": [True],
        #     "y_position_float": [True],
        # },
        {
            "model_class": [MLP],
            "hidden_layers": [
                {"mlp_layers": [512]},
            ],
            "view_size": [7],
            "num_batches": [50],
            "batch_size": [4096 * 8],
            "suffix": ["search1"],
            "target_update_interval": [30],
            "buffer_length": [500_000],
            "float_goal_vector": [True],
            "gamma": [0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98],
        },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "epsilon_decay": [
        #         0.99,
        #         0.98,
        #         0.991,
        #         0.992,
        #         0.993,
        #         0.994,
        #         0.995,
        #         0.996,
        #         0.997,
        #         0.998,
        #         0.999,
        #     ],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [
        #         50_000,
        #         100_000,
        #         200_000,
        #         300_000,
        #         400_000,
        #         500_000,
        #         600_000,
        #         700_000,
        #         800_000,
        #         900_000,
        #         1_000_000,
        #     ],
        #     "float_goal_vector": [True],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [10, 20, 25, 30, 35],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "modulo_reward_x": [(1, -1)],
        #     "modulo_reward_y": [(1, -1)],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "modulo_reward_x": [(2, -2)],
        #     "modulo_reward_y": [(2, -2)],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "modulo_reward_x": [(5, -5)],
        #     "modulo_reward_y": [(5, -5)],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "modulo_reward_x": [(10, -10)],
        #     "modulo_reward_y": [(10, -10)],
        # },
        # {
        #     "model_class": [MLP],
        #     "hidden_layers": [
        #         {"mlp_layers": [512]},
        #     ],
        #     "view_size": [7],
        #     "num_batches": [50],
        #     "batch_size": [4096 * 8],
        #     "suffix": ["search1"],
        #     "target_update_interval": [30],
        #     "buffer_length": [500_000],
        #     "float_goal_vector": [True],
        #     "modulo_reward_x": [(20, -20)],
        #     "modulo_reward_y": [(20, -20)],
        # },
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
