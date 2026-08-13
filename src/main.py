import time
from pathlib import Path

import torch

from src.agents.action_agent import ActionAgent
from src.core.MultiRobotGridEnv import MultiRobotGridEnv
from src.models.depot import Depot
from src.neural_networks.architectures.mlp import MLP
from src.neural_networks.observation_configs import OBSERVATION_CONFIGS


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
    model_dir = "MLP_512"
    model_filename = "b120_r60_v7_u60_modulo2rewardxy.pth"
    model_class = MLP

    model_path = models_path / model_dir / model_filename

    config_name = model_filename.split("_")[-1].split(".")[0]
    config_name = "modulo2"
    config = OBSERVATION_CONFIGS[config_name]
    hidden_layers = get_hidden_layers(model_dir)

    view_size = int(model_filename.split("v")[1].split("_")[0])
    n_actions = config.n_actions

    grid_size = (20, 20)

    depot1 = Depot((0, 0))
    depot2 = Depot((grid_size[0] - 1, 0))
    depot3 = Depot((0, grid_size[1] - 1))
    depot4 = Depot((grid_size[0] - 1, grid_size[1] - 1))

    model = model_class(
        hidden_layers=hidden_layers,
        view_size=view_size,
        view_dims=config.view_dims,
        additional_input_size=config.get_additional_input_size(),
        output_size=n_actions,
        observation_config=config,
        observation_config_name=config_name,
    ).to("cpu")
    weights_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(weights_dict)

    manhattan_delivery_times = []
    delivery_times = []
    env = MultiRobotGridEnv(
        grid_size=grid_size,
        obstacles=None,
        agent_view_size=view_size,
        input_depots=[depot1, depot2],
        output_depots=[depot3, depot4],
        step_limit=5000,
        task_length=5,
        max_robots=60,
        num_tasks=300,
        **config.to_dict(),
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
