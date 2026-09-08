import time
from pathlib import Path

import torch

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.neural_networks.architectures.mlp import MLP
from src.neural_networks.model_config import ModelConfig


def get_hidden_layers(model_name):
    hidden_layers = {}
    mlp_name = model_name[model_name.index("MLP_") :]
    mlp_layers = mlp_name.removeprefix("MLP_").split("_")
    mlp_layers = [int(layer) for layer in mlp_layers]
    hidden_layers["mlp_layers"] = mlp_layers
    return hidden_layers


def main(
    model: Path,
    env: MultiRobotGridEnv,
    render=True,
):
    observations, _info = env.reset()
    if render:
        env.render()
    terminated = False
    truncated = False
    while not (terminated or truncated):
        quit_requested, pause_pressed = False, False
        if render:
            quit_requested, pause_pressed = env.handle_events()
        if quit_requested:
            break
        if pause_pressed:
            env.paused = not env.paused
        if env.paused and render:
            env.render(paused=True)
            time.sleep(0.1)
            continue
        # 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT, 4=WAIT
        model.cpu().eval()
        dqn_agent = ActionAgent(model, epsilon=0, epsilon_min=0, decay=0)
        actions = {
            agent_id: dqn_agent.get_action(obs, device="cpu")
            for agent_id, obs in observations.items()
        }
        observations, _rewards, terminated, truncated, _info = env.step(actions)
        if render:
            env.render_move(move_time=0.3, fps=30)
            # time.sleep(0.07)
    if render:
        time.sleep(1)
    return env.avg_manhattan_distance, env.avg_delivery_time


if __name__ == "__main__":
    models_path = Path(__file__).resolve().parent / "neural_networks" / "models"

    # CONFIG
    config_params = {
        "grid_size": (20, 20),
        "step_limit": 5000,
        "task_length": 5,
        "device": torch.device("cpu"),
        "model_class": MLP,
        "hidden_layers": {"mlp_layers": [512]},
        "view_size": 7,
        "num_robots": 60,
        "suffix": "sample1",
        "batch_size": 4096 * 8,
        "num_batches": 50,
        "num_tasks": 3000,
    }
    model_config = ModelConfig(**config_params)
    model = model_config.load_model().to("cpu")

    model_config.num_robots = 60

    manhattan_delivery_times = []
    delivery_times = []
    env = MultiRobotGridEnv(
        **model_config.get_env_params(),
    )
    # try:
    avg_manhattan_delivery_time, avg_delivery_time = main(
        model=model,
        env=env,
        render=True,
    )
    print()

    # except KeyboardInterrupt:
    #     print("Przerwano program")

    # except Exception as e:
    #     print(e)
