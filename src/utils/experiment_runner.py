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
        "step_limit": [1000],
        "task_length": [5],
        "device": [device],
        "model_class": [MLP],
        "hidden_layers": [{"mlp_layers": [512]}],
        "view_size": [11],
        "buffer_length": [1_000_000],
        "num_robots": [60],
        "target_update_interval": [60],
        "suffix": ["sample1"],
        "batch_size": [4096 * 6],
        "num_batches": [45],
        "num_episodes": [1000],
    }
    variations = [
        {
            "model_class": [MLP],
            "hidden_layers": [
                {"mlp_layers": [256]},
            ],
            "batch_size": [4096 * 2, 4096 * 4],
            "num_batches": [50, 60],
            "view_size": [7],
            "suffix": ["sample1"],
            "num_episodes": [1000],
        },
        {
            "model_class": [MLP],
            "hidden_layers": [
                {"mlp_layers": [512]},
                {"mlp_layers": [1024]},
                {"mlp_layers": [2048]},
            ],
            "batch_size": [4096 * 6],
            "num_batches": [45],
            "view_size": [7, 9, 11, 13],
            "suffix": ["sample1"],
            "num_episodes": [1000],
        },
        # With padding
        {
            "num_episodes": [1000],
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 32, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 32, 1)], "mlp_layers": [512]},
                {"cnn_layers": [(3, 32, 1)], "mlp_layers": [1024]},
            ],
            "suffix": ["sample1"],
        },
        {
            "num_episodes": [1000],
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(5, 32, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(5, 32, 1)], "mlp_layers": [512]},
                {"cnn_layers": [(5, 32, 1)], "mlp_layers": [1024]},
            ],
            "suffix": ["sample1"],
        },
        {
            "num_episodes": [1000],
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 32, 1), (3, 32, 1)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 32, 1), (3, 32, 1)], "mlp_layers": [512]},
                {"cnn_layers": [(3, 32, 1), (3, 32, 1)], "mlp_layers": [1024]},
            ],
            "suffix": ["sample1"],
        },
        # Without padding
        {
            "num_episodes": [1000],
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 32, 0)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 32, 0)], "mlp_layers": [512]},
                {"cnn_layers": [(3, 32, 0)], "mlp_layers": [1024]},
            ],
            "suffix": ["sample1"],
        },
        {
            "num_episodes": [1000],
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(5, 32, 0)], "mlp_layers": [256]},
                {"cnn_layers": [(5, 32, 0)], "mlp_layers": [512]},
                {"cnn_layers": [(5, 32, 0)], "mlp_layers": [1024]},
            ],
            "suffix": ["sample1"],
        },
        {
            "num_episodes": [1000],
            "model_class": [CNN],
            "hidden_layers": [
                {"cnn_layers": [(3, 32, 0), (3, 32, 0)], "mlp_layers": [256]},
                {"cnn_layers": [(3, 32, 0), (3, 32, 0)], "mlp_layers": [512]},
                {"cnn_layers": [(3, 32, 0), (3, 32, 0)], "mlp_layers": [1024]},
            ],
            "suffix": ["sample1"],
        },
    ]

    # batch_size 512    	brak widocznego maksymalnego wyniku
    # batch_size 1024   	brak widocznego maksymalnego wyniku
    # batch_size 2048 		best num batches ~ 150, best result -> 73.8%
    # batch_size 4096 		best num batches ~ 104, best result -> 74.5$
    # batch_size 4096 * 2 	best num batches ~ 75, best result -> 74.71%
    # batch_size 4096 * 4 	best num batches ~ 65, best result -> 74.95%
    # batch_size 4096 * 6	best num batches ~ 45, best result -> 74.9%, very stable
    # batch_size 4096 * 8 	best num batches ~ 50, best result -> 75.15%
    # batch_size 4096 * 16 	best num batches ~ 45, best result -> 74.89%
    # batch_size 4096 * 32 	best num batches ~ 50, best result -> 74.93%

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
