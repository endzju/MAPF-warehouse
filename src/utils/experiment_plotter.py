from src.neural_networks.MLP.mlp import MLP
from src.utils.plots import plot_delivery_efficiency, plot_delivery_throughput

if __name__ == "__main__":
    model_classes = [MLP]
    hidden_layers_list = [
        [16],
        [32],
        [64],
        [128],
        [128, 64],
        [256, 128],
        [512, 256],
        [1024, 512],
        [128, 64, 32],
        [256, 128, 64],
        [512, 256, 128],
        [1024, 512, 256],
    ]
    num_robot_list = [60]
    batch_sizes = [70]
    view_sizes = [5]
    update_episodes_list = [20]

    x_ticks = [n for n in range(10, 101, 5)]

    models = []
    for hidden_layers in hidden_layers_list:
        for model_class in model_classes:
            models.append(
                model_class(hidden_layers=hidden_layers, input_size=1, output_size=1)
            )

    plot_delivery_efficiency(
        models=models,
        batch_sizes=batch_sizes,
        num_robot_list=num_robot_list,
        view_sizes=view_sizes,
        update_episodes_list=update_episodes_list,
        x_ticks=x_ticks,
    )

    plot_delivery_throughput(
        models=models,
        batch_sizes=batch_sizes,
        num_robot_list=num_robot_list,
        view_sizes=view_sizes,
        update_episodes_list=update_episodes_list,
        x_ticks=x_ticks,
    )
