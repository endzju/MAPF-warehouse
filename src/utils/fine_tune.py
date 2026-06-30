from torch import nn

from src.neural_networks.CNN.cnn import CNN1  # noqa: F401
from src.neural_networks.MLP.mlp import MLP1, MLP2, MLP3  # noqa: F401
from src.utils.train import train


def run_fine_tuning(
    model_configs: list[tuple[type, int, int]],
    best_models: list[nn.Module],
    params: dict,
):
    print("Fine tuning...")
    params = params.copy()
    params["epsilon"] = 0
    params["epsilon_min"] = 0
    params["epsilon_decay"] = 0
    params["epsilon_episodes"] = 0
    params["is_tuned"] = True

    training_results = []

    for best_idx, (model_class, num_robots, view_size) in enumerate(model_configs):
        params["model_class"] = model_class
        params["env_max_robots"] = num_robots
        params["env_num_tasks"] = num_robots * 5
        params["env_agent_view_size"] = view_size
        params["in_model"] = best_models[best_idx]

        print(
            f"Training model: {model_class.__name__}, num robots: {num_robots}, view size: {view_size}"
        )
        training_results.append(
            train(
                **params,
            )
        )
        print("model trained")

    print("Training done")
    return training_results


if __name__ == "__main__":
    pass
