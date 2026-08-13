from itertools import product

from src.neural_networks.architectures.mlp import MLP
from src.utils.plots import plot_delivery_efficiency, plot_delivery_throughput

if __name__ == "__main__":
    hidden_layers = [
        [128],
        [256],
        [512],
        [16, 8],
        [32, 16],
        [64, 32],
        [128, 64],
        [256, 128],
        [512, 256],
        [1024, 512],
        [64, 32, 16],
        [128, 64, 32],
        [256, 128, 64],
        [512, 256, 128],
        [1024, 512, 256],
        [128, 64, 32, 16],
        [256, 128, 64, 32],
        [512, 256, 128, 64],
        [1024, 512, 256, 128],
    ]
    models_settings = []
    for h in hidden_layers:
        models_settings.append(
            {
                "class": MLP,
                "view_size": 7,
                "hidden_layers": {
                    "mlp_layers": h,
                },
                "model_config": "modulo2rewardxy",
            }
        )
    train_robot_list = [60]
    train_num_batches_list = [120]
    train_target_update_interval = [60]

    eval_robot_list = [n for n in range(10, 101, 5)]

    models = []

    for model_setting in models_settings:
        # config = observation_configs.OBSERVATION_CONFIGS[model_setting["model_config"]]
        model = model_setting["class"](
            hidden_layers=model_setting["hidden_layers"],
            view_size=model_setting["view_size"],
            view_dims=4,
            additional_input_size=2,
            output_size=5,
            observation_config=None,
            observation_config_name=model_setting["model_config"],
        )
        models.append(model)

    model_configs = list(
        product(
            models,
            train_robot_list,
            train_num_batches_list,
            train_target_update_interval,
        )
    )

    for model, num_robots, num_batches, target_update_interval in model_configs:
        checkpoint_name = f"b{num_batches}_r{num_robots}_v{model.view_size}_u{target_update_interval}_{model.observation_config_name}.pth"
        model.checkpoint_name = checkpoint_name

    x_ticks = [n for n in range(10, 101, 5)]

    plot_delivery_efficiency(models=models, x_ticks=x_ticks)
    plot_delivery_throughput(models=models, x_ticks=x_ticks)
